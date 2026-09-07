import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import PrivacyBanner from "@/components/PrivacyBanner";
import { getModelMeta, getModelOnnxBytes, generateNotesFromGlosses, type ModelMeta, type NotesMode } from "@/lib/api";
import { loadHandLandmarker, detectHands } from "@/lib/handLandmarker";
import { loadOnnxSession, runInference } from "@/lib/onnxSession";
import { KeypointBuffer, SessionSmoother, keypointsFromLandmarks, softmaxArgmax, DEFAULT_SMOOTHING, type PredictionEvent } from "@/lib/webcamPipeline";
import { Camera, Square, Trash2, FileText, AlertTriangle, Hand } from "lucide-react";

type CameraState = "idle" | "requesting" | "active" | "error";
type ModelState = "idle" | "loading" | "ready" | "error";

// How often to run hand-landmark detection + inference, in ms. Chosen to
// roughly match the training-time sampling cadence: frame_skip=8 at a
// typical 25-30fps source video is one kept frame per ~270-320ms. This is
// an approximation (browsers don't give exact frame-count control the way
// offline video decoding does) -- documented in ARCHITECTURE.md.
const DETECTION_INTERVAL_MS = 280;

export default function Webcam() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastDetectionRef = useRef<number>(0);
  const detectingRef = useRef(false);
  const bufferRef = useRef<KeypointBuffer | null>(null);
  const smootherRef = useRef<SessionSmoother>(new SessionSmoother(DEFAULT_SMOOTHING));
  const sessionStartRef = useRef<number>(0);
  const onnxSessionRef = useRef<Awaited<ReturnType<typeof loadOnnxSession>> | null>(null);
  const landmarkerRef = useRef<Awaited<ReturnType<typeof loadHandLandmarker>> | null>(null);
  const modelMetaRef = useRef<ModelMeta | null>(null);

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [modelError, setModelError] = useState<string | null>(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [current, setCurrent] = useState<{ label: string; confidence: number } | "uncertain" | null>(null);
  const [history, setHistory] = useState<PredictionEvent[]>([]);
  const [notes, setNotes] = useState<string | null>(null);
  const [notesMode, setNotesMode] = useState<NotesMode>("template");
  const [generating, setGenerating] = useState(false);
  const [notesError, setNotesError] = useState<string | null>(null);

  // Load model metadata + ONNX weights + hand landmarker once, up front,
  // WITHOUT requesting camera access -- camera permission is only
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

  const drawOverlay = (hands: { x: number; y: number }[][]) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#22c55e";
    for (const hand of hands) {
      for (const pt of hand) {
        ctx.beginPath();
        ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };

  const detectLoop = useCallback((timestampMs: number) => {
    rafRef.current = requestAnimationFrame(detectLoop);
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
        const { hands } = detectHands(landmarker, video, timestampMs);
        drawOverlay(hands as any);
        const vec = keypointsFromLandmarks(hands);
        buffer.push(vec);

        const window = buffer.getNormalizedWindow();
        if (!window) return;

        const logits = await runInference(session, window, meta.max_len, meta.input_dim);
        const { index, confidence } = softmaxArgmax(logits);
        const label = meta.id2label[String(index)] ?? `class_${index}`;

        const elapsedSeconds = (performance.now() - sessionStartRef.current) / 1000;
        const result = smootherRef.current.update(label, confidence, elapsedSeconds);

        if (result.status === "uncertain") {
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
        console.error("[webcam] detection tick failed:", err);
      } finally {
        detectingRef.current = false;
      }
    })();
  }, []);

  const startSession = async () => {
    setCameraError(null);
    setCameraState("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraState("active");

      if (!landmarkerRef.current) {
        landmarkerRef.current = await loadHandLandmarker();
      }

      bufferRef.current?.clear();
      smootherRef.current.reset();
      sessionStartRef.current = performance.now();
      lastDetectionRef.current = 0;
      setHistory([]);
      setCurrent(null);
      setNotes(null);
      setNotesError(null);
      setSessionActive(true);
      rafRef.current = requestAnimationFrame(detectLoop);
    } catch (err: any) {
      setCameraState("error");
      if (err?.name === "NotAllowedError") {
        setCameraError("Camera permission was denied. Allow camera access in your browser's site settings and try again.");
      } else if (err?.name === "NotFoundError") {
        setCameraError("No camera was found on this device.");
      } else if (err?.message?.includes("hand landmark") || err?.message?.toLowerCase().includes("fetch")) {
        setCameraError(`Couldn't load the hand-tracking model (check your internet connection for the one-time model download): ${err.message}`);
      } else {
        setCameraError(`Couldn't start the camera: ${err?.message || err}`);
      }
    }
  };

  const stopSession = () => {
    stopCamera();
    setSessionActive(false);
    setCameraState("idle");
    setCurrent(null);
  };

  const clearSession = () => {
    setHistory([]);
    setCurrent(null);
    setNotes(null);
    setNotesError(null);
    smootherRef.current.reset();
  };

  const handleGenerateNotes = async () => {
    if (history.length === 0) return;
    setGenerating(true);
    setNotesError(null);
    try {
      const res = await generateNotesFromGlosses(history.map((e) => e.label), { notesMode });
      setNotes(res.notes_md);
    } catch (err: any) {
      setNotesError(err?.message || "Failed to generate notes.");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Live Webcam Session</h1>
        <p className="text-sm text-muted-foreground">
          Sign continuously in front of your webcam. Recognized signs are collected into a session; generate notes when you're done.
        </p>
      </div>

      <PrivacyBanner variant="webcam" />

      {modelState === "error" && (
        <Card className="border-destructive/40 bg-destructive/5">
          <CardContent className="flex items-start gap-2 py-4 text-sm">
            <AlertTriangle className="h-4 w-4 shrink-0 text-destructive mt-0.5" />
            <span>{modelError}</span>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6">
        <Card>
          <CardContent className="p-4 space-y-4">
            <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-black/90">
              <video ref={videoRef} className="h-full w-full object-cover -scale-x-100" muted playsInline />
              <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100" />
              {cameraState !== "active" && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-white/60">
                  {cameraState === "requesting" ? "Requesting camera access…" : "Camera is off"}
                </div>
              )}
            </div>

            {cameraError && (
              <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm">
                <AlertTriangle className="h-4 w-4 shrink-0 text-destructive mt-0.5" />
                <span>{cameraError}</span>
              </div>
            )}

            <div className="flex flex-wrap gap-2">
              {!sessionActive ? (
                <Button onClick={startSession} disabled={modelState !== "ready" || cameraState === "requesting"}>
                  <Camera className="h-4 w-4 mr-2" />
                  Start Session
                </Button>
              ) : (
                <Button onClick={stopSession} variant="secondary">
                  <Square className="h-4 w-4 mr-2" />
                  Stop Session
                </Button>
              )}
              <Button onClick={clearSession} variant="outline" disabled={history.length === 0 && !notes}>
                <Trash2 className="h-4 w-4 mr-2" />
                Clear Session
              </Button>
              <select
                value={notesMode}
                onChange={(e) => setNotesMode(e.target.value as NotesMode)}
                className="rounded-lg border bg-background px-3 py-2 text-sm"
              >
                <option value="template">Deterministic notes</option>
                <option value="llm">Local LLM</option>
              </select>
              <Button onClick={handleGenerateNotes} disabled={history.length === 0 || generating} variant="default">
                <FileText className="h-4 w-4 mr-2" />
                {generating ? "Generating…" : "Generate Notes"}
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Current Sign</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {current === null && <p className="text-sm text-muted-foreground">Start a session to begin recognizing signs.</p>}
              {current === "uncertain" && (
                <div className="flex items-center gap-2 text-amber-600 text-sm">
                  <AlertTriangle className="h-4 w-4" />
                  Low confidence — please repeat
                </div>
              )}
              {current && current !== "uncertain" && (
                <>
                  <div className="text-xl font-bold flex items-center gap-2">
                    <Hand className="h-5 w-5 text-accent" />
                    {current.label}
                  </div>
                  <Progress value={current.confidence * 100} />
                  <p className="text-xs text-muted-foreground">Confidence: {(current.confidence * 100).toFixed(0)}%</p>
                </>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Sign History ({history.length})</CardTitle>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <p className="text-sm text-muted-foreground">No signs recognized yet.</p>
              ) : (
                <ul className="space-y-1 max-h-64 overflow-y-auto text-sm">
                  {history.map((e, i) => (
                    <li key={i} className="flex items-center justify-between gap-2">
                      <span className="tabular-nums text-muted-foreground">{formatTime(e.timestamp)}</span>
                      <Badge variant="secondary">{e.label}</Badge>
                      <span className="text-xs text-muted-foreground">{(e.confidence * 100).toFixed(0)}%</span>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {notesError && (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm">
          <AlertTriangle className="h-4 w-4 shrink-0 text-destructive mt-0.5" />
          <span>{notesError}</span>
        </div>
      )}

      {notes && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              Generated Lecture Notes
            </CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="whitespace-pre-wrap text-sm font-sans">{notes}</pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
