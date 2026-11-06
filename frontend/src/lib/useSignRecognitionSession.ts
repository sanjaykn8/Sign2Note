import { useCallback, useEffect, useRef, useState } from "react";
import { getModelMeta, getModelOnnxBytes, recognizeFromFeatures, type ModelMeta } from "@/lib/api";
import { loadHolisticLandmarker, detectHolistic } from "@/lib/holisticLandmarker";
import { loadOnnxSession, resetOnnxSession, runInference } from "@/lib/onnxSession";
import { featureDimForSchema, toSchema } from "@/lib/featureSchema";
import {
  KeypointBuffer,
  SessionSmoother,
  buildFeatureVector,
  softmaxArgmax,
  DEFAULT_SMOOTHING,
  type PredictionEvent,
  type Landmark,
} from "@/lib/webcamPipeline";

export type CameraState = "idle" | "requesting" | "active" | "error";
export type ModelState = "idle" | "loading" | "ready" | "error";
export type CurrentSign = { label: string; confidence: number } | "uncertain" | "no_sign" | null;
/** "local": ONNX model running in this browser (only gloss words leave the
 * device). "server": keypoints extracted here are sent to the main
 * server's recognition model (still never video). */
export type RecognizerMode = "local" | "server";

/** Where the video frames come from. "webcam" is getUserMedia (the
 * camera). "screen" is getDisplayMedia -- the browser's own tab/window/
 * screen picker. What it can capture is up to the browser and OS; there is
 * no way for a web page to force system-wide capture, and this doesn't
 * pretend to. Unsupported on most mobile browsers. */
export type RecognitionSource = "webcam" | "screen";

const DEFAULT_SAMPLE_HZ = 7.5; // FDMSE-ISL 60 fps / frame_skip 8
const DEFAULT_WINDOW_FRAMES = 24;
const SERVER_EVERY_N_TICKS = 2; // server mode: one request per 2 samples, at most one in flight
const FAILURES_BEFORE_ERROR = 8;
const NO_HANDS_TICKS_FOR_NO_SIGN = 3;

const POSE_SKELETON: [number, number][] = [
  [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [23, 24],
  [11, 23], [12, 24],
];

function drawOverlay(
  canvas: HTMLCanvasElement | null,
  video: HTMLVideoElement | null,
  leftHand: Landmark[] | null,
  rightHand: Landmark[] | null,
  pose: Landmark[] | null
) {
  if (!canvas || !video) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#22c55e"; // hands
  for (const hand of [leftHand, rightHand]) {
    if (!hand) continue;
    for (const pt of hand) {
      ctx.beginPath();
      ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if (pose) {
    ctx.strokeStyle = "#60a5fa"; // pose skeleton
    ctx.lineWidth = 2;
    for (const [a, b] of POSE_SKELETON) {
      const pa = pose[a];
      const pb = pose[b];
      if (!pa || !pb) continue;
      ctx.beginPath();
      ctx.moveTo(pa.x * canvas.width, pa.y * canvas.height);
      ctx.lineTo(pb.x * canvas.width, pb.y * canvas.height);
      ctx.stroke();
    }
  }
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export interface UseSignRecognitionSession {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  cameraState: CameraState;
  cameraError: string | null;
  modelState: ModelState;
  modelError: string | null;
  /** Which recognizer is in use, and why (e.g. why local isn't). */
  recognizer: RecognizerMode | null;
  recognizerNote: string | null;
  modelMeta: ModelMeta | null;
  /** Set when recognition keeps failing during a session (e.g. server
   * unreachable); cleared on the next success. */
  recognitionError: string | null;
  sessionActive: boolean;
  paused: boolean;
  current: CurrentSign;
  history: PredictionEvent[];
  editingIndex: number | null;
  editDraft: string;
  setEditDraft: (v: string) => void;
  source: RecognitionSource;
  screenCaptureSupported: boolean;
  startSession: (source?: RecognitionSource) => Promise<void>;
  stopSession: () => void;
  pauseSession: () => void;
  resumeSession: () => void;
  clearSession: () => void;
  undoLast: () => void;
  deleteEvent: (index: number) => void;
  startEdit: (index: number) => void;
  saveEdit: (index: number) => void;
  cancelEdit: () => void;
  elapsedSessionSeconds: () => number;
  retryModelLoad: () => void;
}

/**
 * The live recognition pipeline shared by Live Webcam (Mode 2) and Live
 * Transcription (Mode 3): model selection (local vs server), camera/screen
 * lifecycle, per-sample Holistic detection -> canonical feature vector ->
 * recognition, the stability state machine (SessionSmoother), and history
 * editing (undo/delete/edit). Language generation is NOT done here -- the
 * pages call the notes/transcript service only when the user asks.
 */
export function useSignRecognitionSession(): UseSignRecognitionSession {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastDetectionRef = useRef<number>(0);
  const detectingRef = useRef(false);
  const bufferRef = useRef<KeypointBuffer | null>(null);
  const smootherRef = useRef<SessionSmoother>(new SessionSmoother(DEFAULT_SMOOTHING));
  const sessionStartRef = useRef<number>(0);
  const pausedRef = useRef(false);
  const pauseStartRef = useRef<number | null>(null);
  const totalPausedMsRef = useRef(0);
  const onnxSessionRef = useRef<Awaited<ReturnType<typeof loadOnnxSession>> | null>(null);
  const landmarkerRef = useRef<Awaited<ReturnType<typeof loadHolisticLandmarker>> | null>(null);
  const modelMetaRef = useRef<ModelMeta | null>(null);
  const recognizerRef = useRef<RecognizerMode | null>(null);
  const tickRef = useRef(0);
  const serverInFlightRef = useRef(false);
  const failuresRef = useRef(0);
  const noHandsTicksRef = useRef(0);
  const activeRef = useRef(false);
  const historyRef = useRef<PredictionEvent[]>([]);

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [modelError, setModelError] = useState<string | null>(null);
  const [recognizer, setRecognizer] = useState<RecognizerMode | null>(null);
  const [recognizerNote, setRecognizerNote] = useState<string | null>(null);
  const [modelMeta, setModelMeta] = useState<ModelMeta | null>(null);
  const [recognitionError, setRecognitionError] = useState<string | null>(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [paused, setPaused] = useState(false);
  const [current, setCurrent] = useState<CurrentSign>(null);
  const [history, setHistoryState] = useState<PredictionEvent[]>([]);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const [source, setSource] = useState<RecognitionSource>("webcam");
  const [loadAttempt, setLoadAttempt] = useState(0);

  // History lives in a ref too, so callbacks never act on a stale copy.
  const setHistory = useCallback((update: (h: PredictionEvent[]) => PredictionEvent[]) => {
    historyRef.current = update(historyRef.current);
    setHistoryState(historyRef.current);
  }, []);

  const screenCaptureSupported =
    typeof navigator !== "undefined" && !!navigator.mediaDevices && "getDisplayMedia" in navigator.mediaDevices;

  // Decide local vs server recognition once, up front, WITHOUT asking for
  // camera permission (that only happens on Start).
  useEffect(() => {
    let cancelled = false;
    setModelState("loading");
    setModelError(null);
    (async () => {
      let meta: ModelMeta;
      try {
        meta = await getModelMeta();
      } catch (err) {
        if (cancelled) return;
        setModelError(`Couldn't load the recognition model's details from the server: ${errorMessage(err)}`);
        setModelState("error");
        return;
      }
      if (cancelled) return;
      if (meta.ready === false) {
        setModelError(
          `The server has no usable recognition model: ${(meta.compatibility_warnings || []).join("; ") || "unknown reason"}`
        );
        setModelState("error");
        return;
      }
      const dim = featureDimForSchema(meta.feature_schema_version ?? "1.0");
      if (dim === null || dim !== meta.input_dim) {
        setModelError(
          `The server's model uses feature schema ${meta.feature_schema_version} (${meta.input_dim} dims), which this version of the app cannot produce. Update the app or serve a compatible model.`
        );
        setModelState("error");
        return;
      }
      modelMetaRef.current = meta;
      setModelMeta(meta);

      // Local recognizer: only if the ONNX model loads AND a test inference
      // returns the expected number of classes -- availability is checked,
      // not assumed from a file existing.
      let mode: RecognizerMode = "server";
      let note: string | null = null;
      if (meta.onnx_available === false) {
        note = "No ONNX export on the server, so recognition runs on the server.";
      } else {
        try {
          const bytes = await getModelOnnxBytes();
          if (cancelled) return;
          if (loadAttempt > 0) resetOnnxSession(); // pick up a replaced model on retry
          const session = await loadOnnxSession(bytes);
          const probe = await runInference(session, new Float32Array(meta.max_len * meta.input_dim), meta.max_len, meta.input_dim);
          if (probe.length !== meta.num_classes) {
            throw new Error(`model returned ${probe.length} scores, expected ${meta.num_classes}`);
          }
          onnxSessionRef.current = session;
          mode = "local";
        } catch (err) {
          note = `Local recognizer unavailable (${errorMessage(err)}); using server recognition instead.`;
        }
      }
      if (cancelled) return;
      recognizerRef.current = mode;
      setRecognizer(mode);
      setRecognizerNote(note);
      bufferRef.current = new KeypointBuffer(meta.max_len, meta.input_dim, meta.window_frames ?? DEFAULT_WINDOW_FRAMES);
      setModelState("ready");
    })();
    return () => {
      cancelled = true;
    };
  }, [loadAttempt]);

  const retryModelLoad = useCallback(() => setLoadAttempt((n) => n + 1), []);

  const stopCamera = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const elapsedSessionSeconds = useCallback(() => {
    const pausedNow = pauseStartRef.current !== null ? performance.now() - pauseStartRef.current : 0;
    return (performance.now() - sessionStartRef.current - totalPausedMsRef.current - pausedNow) / 1000;
  }, []);

  const handlePrediction = useCallback(
    (label: string, confidence: number, at: number) => {
      if (!activeRef.current || pausedRef.current) return;
      const result = smootherRef.current.update(label, confidence, at);
      if (result.status === "no_sign") setCurrent("no_sign");
      else if (result.status === "uncertain") setCurrent("uncertain");
      else setCurrent({ label, confidence });
      if (result.status === "committed") setHistory((h) => [...h, result.event]);
    },
    [setHistory]
  );

  const noteFailure = useCallback((err: unknown) => {
    failuresRef.current += 1;
    console.error("[recognition] tick failed:", err);
    if (failuresRef.current >= FAILURES_BEFORE_ERROR) {
      setRecognitionError(
        recognizerRef.current === "server"
          ? `Server recognition is failing: ${errorMessage(err)}. Your recognized glosses so far are kept.`
          : `Recognition is failing: ${errorMessage(err)}`
      );
    }
  }, []);

  const noteSuccess = useCallback(() => {
    if (failuresRef.current >= FAILURES_BEFORE_ERROR) setRecognitionError(null);
    failuresRef.current = 0;
  }, []);

  const detectLoop = useCallback(
    (timestampMs: number) => {
      rafRef.current = requestAnimationFrame(detectLoop);
      if (pausedRef.current) return;
      const meta = modelMetaRef.current;
      const intervalMs = 1000 / (meta?.sample_hz || DEFAULT_SAMPLE_HZ);
      if (timestampMs - lastDetectionRef.current < intervalMs) return;
      if (detectingRef.current) return;
      lastDetectionRef.current = timestampMs;

      const video = videoRef.current;
      const landmarker = landmarkerRef.current;
      const buffer = bufferRef.current;
      if (!video || !landmarker || !meta || !buffer) return;
      if (video.readyState < 2) return;

      detectingRef.current = true;
      (async () => {
        try {
          const { leftHand, rightHand, pose, face } = detectHolistic(landmarker, video, timestampMs);
          drawOverlay(canvasRef.current, video, leftHand, rightHand, pose);
          const vec = toSchema(buildFeatureVector(leftHand, rightHand, pose, face), meta.feature_schema_version ?? "1.0");
          buffer.push(vec);
          const at = elapsedSessionSeconds();

          // No hands for a few samples = not signing: report NO_SIGN
          // directly instead of asking the classifier to label noise.
          noHandsTicksRef.current = leftHand || rightHand ? 0 : noHandsTicksRef.current + 1;
          if (noHandsTicksRef.current >= NO_HANDS_TICKS_FOR_NO_SIGN) {
            handlePrediction("", 0, at);
            noteSuccess();
            return;
          }

          if (recognizerRef.current === "local" && onnxSessionRef.current) {
            const window = buffer.getNormalizedWindow();
            if (!window) return;
            const logits = await runInference(onnxSessionRef.current, window, meta.max_len, meta.input_dim);
            const { index, confidence } = softmaxArgmax(logits);
            handlePrediction(meta.id2label[String(index)] ?? `class_${index}`, confidence, at);
            noteSuccess();
          } else {
            tickRef.current += 1;
            if (tickRef.current % SERVER_EVERY_N_TICKS !== 0 || serverInFlightRef.current) return;
            serverInFlightRef.current = true;
            // Keypoints only (rounded to shrink the request) -- never frames.
            const frames = buffer.rawFrames().map((f) => Array.from(f, (v) => Math.round(v * 1e4) / 1e4));
            recognizeFromFeatures(frames, { threshold: 0, windowFrames: frames.length })
              .then((res) => {
                const seg = res.segments[res.segments.length - 1];
                if (seg) handlePrediction(seg.label, seg.confidence, at);
                noteSuccess();
              })
              .catch(noteFailure)
              .finally(() => {
                serverInFlightRef.current = false;
              });
          }
        } catch (err) {
          // One bad sample must not end the session.
          noteFailure(err);
        } finally {
          detectingRef.current = false;
        }
      })();
    },
    [elapsedSessionSeconds, handlePrediction, noteFailure, noteSuccess]
  );

  const stopSession = useCallback(() => {
    activeRef.current = false;
    stopCamera();
    setSessionActive(false);
    setPaused(false);
    pausedRef.current = false;
    pauseStartRef.current = null;
    setCameraState("idle");
    setCurrent(null);
  }, [stopCamera]);

  const startSession = async (requestedSource: RecognitionSource = "webcam") => {
    if (activeRef.current) return;
    setCameraError(null);
    setRecognitionError(null);
    setCameraState("requesting");
    setSource(requestedSource);
    try {
      if (typeof window !== "undefined" && window.isSecureContext === false) {
        throw new Error(
          "Camera and screen capture need a secure page (https:// or localhost). On another device, open the app over HTTPS -- see SETUP.md \"Connecting other devices\"."
        );
      }
      let stream: MediaStream;
      if (requestedSource === "screen") {
        if (!screenCaptureSupported) {
          throw new Error(
            "Screen/tab capture isn't supported in this browser (common on phones and tablets). Use the webcam instead, or a desktop browser such as Chrome, Edge or Firefox."
          );
        }
        stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
      } else {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser does not provide camera access.");
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraState("active");
      // The browser's own "Stop sharing" button ends the session too.
      stream.getVideoTracks()[0]?.addEventListener("ended", () => stopSession());

      if (!landmarkerRef.current) {
        landmarkerRef.current = await loadHolisticLandmarker();
      }

      bufferRef.current?.clear();
      smootherRef.current.reset();
      sessionStartRef.current = performance.now();
      totalPausedMsRef.current = 0;
      pauseStartRef.current = null;
      pausedRef.current = false;
      lastDetectionRef.current = 0;
      failuresRef.current = 0;
      noHandsTicksRef.current = 0;
      tickRef.current = 0;
      setPaused(false);
      setHistory(() => []);
      setEditingIndex(null);
      setCurrent(null);
      activeRef.current = true;
      setSessionActive(true);
      rafRef.current = requestAnimationFrame(detectLoop);
    } catch (err) {
      stopCamera();
      setCameraState("error");
      const e = err as { name?: string; message?: string };
      if (e?.name === "NotAllowedError") {
        setCameraError(
          requestedSource === "screen"
            ? "Screen/tab sharing permission was denied."
            : "Camera permission was denied. Allow camera access in your browser's site settings and try again."
        );
      } else if (e?.name === "NotFoundError") {
        setCameraError("No camera was found on this device.");
      } else if (e?.name === "NotReadableError") {
        setCameraError("The camera is in use by another application.");
      } else if (e?.message?.includes("landmark") || e?.message?.toLowerCase().includes("fetch")) {
        setCameraError(
          `The hand/face/pose landmark model could not be loaded (it is downloaded once from a CDN): ${e.message}`
        );
      } else {
        setCameraError(e?.message || `Couldn't start ${requestedSource === "screen" ? "screen capture" : "the camera"}.`);
      }
    }
  };

  const pauseSession = () => {
    if (!activeRef.current || pausedRef.current) return;
    pausedRef.current = true;
    pauseStartRef.current = performance.now();
    setPaused(true);
    setCurrent(null);
  };

  const resumeSession = () => {
    if (!activeRef.current || !pausedRef.current) return;
    if (pauseStartRef.current !== null) {
      totalPausedMsRef.current += performance.now() - pauseStartRef.current;
      pauseStartRef.current = null;
    }
    // Don't let frames from before the pause leak into the first window
    // after it.
    bufferRef.current?.clear();
    smootherRef.current.reset();
    const last = historyRef.current[historyRef.current.length - 1];
    smootherRef.current.forgetLastCommitted(last ? last.label : null);
    pausedRef.current = false;
    setPaused(false);
  };

  const clearSession = () => {
    setHistory(() => []);
    setEditingIndex(null);
    setCurrent(null);
    smootherRef.current.reset();
  };

  const undoLast = () => {
    setHistory((h) => {
      if (h.length === 0) return h;
      const next = h.slice(0, -1);
      smootherRef.current.forgetLastCommitted(next.length > 0 ? next[next.length - 1].label : null);
      return next;
    });
    setEditingIndex(null);
  };

  const deleteEvent = (index: number) => {
    setHistory((h) => {
      if (index < 0 || index >= h.length) return h;
      const next = h.filter((_, i) => i !== index);
      if (index === h.length - 1) {
        smootherRef.current.forgetLastCommitted(next.length > 0 ? next[next.length - 1].label : null);
      }
      return next;
    });
    setEditingIndex(null);
  };

  const startEdit = (index: number) => {
    const ev = historyRef.current[index];
    if (!ev) return;
    setEditingIndex(index);
    setEditDraft(ev.label);
  };

  const saveEdit = (index: number) => {
    // Kept as typed (trimmed): the model's own glosses are mixed-case
    // ("Whistle"), and forcing upper case made edited glosses inconsistent.
    const cleaned = editDraft.trim();
    if (cleaned) {
      setHistory((h) => h.map((e, i) => (i === index ? { ...e, label: cleaned, edited: true } : e)));
    }
    setEditingIndex(null);
  };

  const cancelEdit = () => setEditingIndex(null);

  return {
    videoRef,
    canvasRef,
    cameraState,
    cameraError,
    modelState,
    modelError,
    recognizer,
    recognizerNote,
    modelMeta,
    recognitionError,
    sessionActive,
    paused,
    current,
    history,
    editingIndex,
    editDraft,
    setEditDraft,
    source,
    screenCaptureSupported,
    startSession,
    stopSession,
    pauseSession,
    resumeSession,
    clearSession,
    undoLast,
    deleteEvent,
    startEdit,
    saveEdit,
    cancelEdit,
    elapsedSessionSeconds,
    retryModelLoad,
  };
}
