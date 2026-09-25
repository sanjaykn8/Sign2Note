import { useCallback, useEffect, useRef, useState } from "react";
import { getModelMeta, getModelOnnxBytes, type ModelMeta } from "@/lib/api";
import { loadHolisticLandmarker, detectHolistic } from "@/lib/holisticLandmarker";
import { loadOnnxSession, runInference } from "@/lib/onnxSession";
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

/** Where the video frames come from. "webcam" is getUserMedia (the
 * camera). "screen" is getDisplayMedia -- the browser's own tab/window/
 * screen picker (project brief section 22: "watching sign-language
 * content on a laptop/mobile/system screen"). This is the real,
 * browser-supported mechanism, not a fake -- what it actually captures
 * (a tab, a window, or the whole screen) is up to the user's choice in
 * the browser's native picker, and support varies by browser (broad in
 * Chrome/Edge/Firefox; limited/absent in some mobile browsers). There is
 * no way to force true OS-wide capture from a web page, and this doesn't
 * pretend to. */
export type RecognitionSource = "webcam" | "screen";

// How often to run landmark detection + inference, in ms. Chosen to
// roughly match the training-time sampling cadence: frame_skip=8 at a
// typical 25-30fps source video is one kept frame per ~270-320ms. This is
// an approximation (browsers don't give exact frame-count control the way
// offline video decoding does) -- documented in ARCHITECTURE.md.
const DETECTION_INTERVAL_MS = 280;

// A handful of MediaPipe Pose skeleton connections worth drawing for
// visual feedback -- shoulders/elbows/wrists/hips, matching the same
// subset featureSchema.ts feeds the model (see FEATURE_SCHEMA.md). Full
// face-mesh overlay is deliberately NOT drawn: 41+ dots across the face
// every ~280ms is visual noise, not useful feedback (do not over-engineer
// the UI).
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

  ctx.fillStyle = "#22c55e"; // green -- hands
  for (const hand of [leftHand, rightHand]) {
    if (!hand) continue;
    for (const pt of hand) {
      ctx.beginPath();
      ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (pose) {
    ctx.strokeStyle = "#60a5fa"; // blue -- pose skeleton
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

export interface UseSignRecognitionSession {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  cameraState: CameraState;
  cameraError: string | null;
  modelState: ModelState;
  modelError: string | null;
  sessionActive: boolean;
  paused: boolean;
  current: CurrentSign;
  history: PredictionEvent[];
  editingIndex: number | null;
  editDraft: string;
  setEditDraft: (v: string) => void;
  source: RecognitionSource;
  /** Whether getDisplayMedia (screen/tab/window capture) is available in
   * this browser at all -- use to hide/disable the "Screen/Tab" option
   * rather than letting the user pick it and fail (RULE 19). */
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
}

/**
 * Encapsulates the entire local recognition pipeline: model/ONNX loading,
 * camera (or screen-capture) lifecycle, per-frame Holistic detection +
 * feature-vector construction + ONNX inference, the confidence/stability
 * state machine (SessionSmoother), and session history editing
 * (undo/delete/edit). Shared by Webcam.tsx (Mode 2) and
 * LiveTranscription.tsx (Mode 3) so this logic -- camera setup, error
 * handling, the detection loop -- exists in exactly one place. What each
 * page does with `history` once it's collected (generate structured
 * notes, or generate a natural-language transcript, or both) is left to
 * the caller, matching the "recognition produces glosses; something else
 * turns them into notes/transcript" separation (project brief section 58).
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

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [modelError, setModelError] = useState<string | null>(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [paused, setPaused] = useState(false);
  const [current, setCurrent] = useState<CurrentSign>(null);
  const [history, setHistory] = useState<PredictionEvent[]>([]);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const [source, setSource] = useState<RecognitionSource>("webcam");

  const screenCaptureSupported =
    typeof navigator !== "undefined" && !!navigator.mediaDevices && "getDisplayMedia" in navigator.mediaDevices;

  // Load model metadata + ONNX weights + holistic landmarker once, up
  // front, WITHOUT requesting camera/screen access -- permission is only
  // requested when the user explicitly clicks Start Session.
  useEffect(() => {
    let cancelled = false;
    setModelState("loading");
    (async () => {
      try {
        const [meta, onnxBytes] = await Promise.all([getModelMeta(), getModelOnnxBytes()]);
        if (cancelled) return;
        modelMetaRef.current = meta;
        const session = await loadOnnxSession(onnxBytes);
        if (cancelled) return;
        onnxSessionRef.current = session;
        bufferRef.current = new KeypointBuffer(meta.max_len, meta.input_dim);
        setModelState("ready");
      } catch (err: any) {
        if (cancelled) return;
        setModelError(
          err?.message?.includes("404") || err?.message?.includes("No trained") || err?.message?.includes("No ONNX")
            ? "No trained model found on the backend. Run train.py to produce a checkpoint and ONNX export first."
            : `Couldn't load the recognition model: ${err?.message || err}`
        );
        setModelState("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const stopCamera = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const elapsedSessionSeconds = () => (performance.now() - sessionStartRef.current - totalPausedMsRef.current) / 1000;

  const detectLoop = useCallback((timestampMs: number) => {
    rafRef.current = requestAnimationFrame(detectLoop);
    if (pausedRef.current) return;
    if (timestampMs - lastDetectionRef.current < DETECTION_INTERVAL_MS) return;
    if (detectingRef.current) return;
    lastDetectionRef.current = timestampMs;

    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    const session = onnxSessionRef.current;
    const meta = modelMetaRef.current;
    const buffer = bufferRef.current;
    if (!video || !landmarker || !session || !meta || !buffer) return;
    if (video.readyState < 2) return;

    detectingRef.current = true;
    (async () => {
      try {
        const { leftHand, rightHand, pose, face } = detectHolistic(landmarker, video, timestampMs);
        drawOverlay(canvasRef.current, video, leftHand, rightHand, pose);
        const vec = buildFeatureVector(leftHand, rightHand, pose, face);
        buffer.push(vec);

        const window = buffer.getNormalizedWindow();
        if (!window) return;

        const logits = await runInference(session, window, meta.max_len, meta.input_dim);
        const { index, confidence } = softmaxArgmax(logits);
        const label = meta.id2label[String(index)] ?? `class_${index}`;

        const result = smootherRef.current.update(label, confidence, elapsedSessionSeconds());

        if (result.status === "no_sign") {
          setCurrent("no_sign");
        } else if (result.status === "uncertain") {
          setCurrent("uncertain");
        } else {
          setCurrent({ label, confidence });
        }
        if (result.status === "committed") {
          setHistory((h) => [...h, result.event]);
        }
      } catch (err) {
        // Swallow per-tick inference errors so a single bad frame doesn't
        // kill the whole session -- surface nothing to the user unless it
        // keeps happening (that would already show as "uncertain" forever).
        console.error("[recognition] detection tick failed:", err);
      } finally {
        detectingRef.current = false;
      }
    })();
  }, []);

  const startSession = async (requestedSource: RecognitionSource = "webcam") => {
    setCameraError(null);
    setCameraState("requesting");
    setSource(requestedSource);
    try {
      let stream: MediaStream;
      if (requestedSource === "screen") {
        if (!screenCaptureSupported) {
          throw new Error(
            "Screen/tab capture (getDisplayMedia) isn't supported in this browser. Use Webcam instead, or switch to a browser that supports it (current desktop Chrome, Edge, or Firefox)."
          );
        }
        // The browser's OWN picker lets the user choose a tab, a window,
        // or (if their OS/browser offers it) the entire screen -- there is
        // no way for a web page to force true whole-system capture, and
        // this doesn't claim to (project brief RULE 19 / section 22).
        stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
      } else {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraState("active");

      // If the user manually stops screen sharing via the browser's own
      // "Stop sharing" control, treat it the same as clicking Stop.
      stream.getVideoTracks()[0]?.addEventListener("ended", () => {
        stopSession();
      });

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
      setPaused(false);
      setHistory([]);
      setEditingIndex(null);
      setCurrent(null);
      setSessionActive(true);
      rafRef.current = requestAnimationFrame(detectLoop);
    } catch (err: any) {
      setCameraState("error");
      if (err?.name === "NotAllowedError") {
        setCameraError(
          requestedSource === "screen"
            ? "Screen/tab sharing permission was denied."
            : "Camera permission was denied. Allow camera access in your browser's site settings and try again."
        );
      } else if (err?.name === "NotFoundError") {
        setCameraError("No camera was found on this device.");
      } else if (err?.message?.includes("landmark") || err?.message?.toLowerCase().includes("fetch")) {
        // Matches the project's documented error wording (see
        // ARCHITECTURE.md / TROUBLESHOOTING.md): "Face/pose landmark model
        // could not be loaded."
        setCameraError(
          `Face/pose landmark model could not be loaded (check your internet connection for the one-time model download): ${err.message}`
        );
      } else {
        setCameraError(err?.message || `Couldn't start ${requestedSource === "screen" ? "screen capture" : "the camera"}.`);
      }
    }
  };

  const stopSession = () => {
    stopCamera();
    setSessionActive(false);
    setPaused(false);
    pausedRef.current = false;
    pauseStartRef.current = null;
    setCameraState("idle");
    setCurrent(null);
  };

  const pauseSession = () => {
    if (!sessionActive || pausedRef.current) return;
    pausedRef.current = true;
    pauseStartRef.current = performance.now();
    setPaused(true);
    setCurrent(null);
  };

  const resumeSession = () => {
    if (!sessionActive || !pausedRef.current) return;
    if (pauseStartRef.current !== null) {
      totalPausedMsRef.current += performance.now() - pauseStartRef.current;
      pauseStartRef.current = null;
    }
    pausedRef.current = false;
    setPaused(false);
  };

  const clearSession = () => {
    setHistory([]);
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
      const next = h.filter((_, i) => i !== index);
      if (index === h.length - 1) {
        smootherRef.current.forgetLastCommitted(next.length > 0 ? next[next.length - 1].label : null);
      }
      return next;
    });
    setEditingIndex(null);
  };

  const startEdit = (index: number) => {
    setEditingIndex(index);
    setEditDraft(history[index].label);
  };

  const saveEdit = (index: number) => {
    const cleaned = editDraft.trim().toUpperCase();
    if (cleaned) {
      setHistory((h) => h.map((e, i) => (i === index ? { ...e, label: cleaned } : e)));
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
  };
}
