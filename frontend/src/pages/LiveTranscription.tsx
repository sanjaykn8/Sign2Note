import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import PrivacyBanner from "@/components/PrivacyBanner";
import NotesPanel from "@/components/NotesPanel";
import { generateNotesFromGlosses, generateTranscriptFromGlosses, type NotesMode } from "@/lib/api";
import { useSignRecognitionSession, type RecognitionSource } from "@/lib/useSignRecognitionSession";
import {
  Camera, Monitor, Square, Play, Pause as PauseIcon, Trash2, AlertTriangle,
  Hand, Pencil, Check, X, Undo2, MessageSquareText, RotateCcw, Sparkles,
} from "lucide-react";

/**
 * Live Transcription mode (Mode 3): USER SIGNS -> recognition -> stable
 * gloss -> LIVE GLOSS TRANSCRIPT, updating continuously while signing.
 * Deliberately does NOT call the LLM per sign -- only once, when the user
 * clicks "Stop & Generate" (project brief section 21 / RULE 9). That one
 * action produces BOTH a natural-language transcript (flowing prose,
 * Layer 2) and structured notes (Layer 3), from the same frozen gloss
 * sequence (Layer 1) -- see the three-output-layers note in
 * ml_service/notes_generator.py.
 *
 * `genPhase` implements the STOPPING -> GENERATING part of the project
 * brief's state machine (section 56) explicitly, since that's exactly
 * where "don't allow duplicate requests on a rapid double-click" matters;
 * the live ACTIVE/PAUSED states are read directly from
 * useSignRecognitionSession's own sessionActive/paused rather than
 * duplicated into a second state enum.
 */
type GenPhase = "idle" | "stopping" | "generating" | "completed" | "error";

export default function LiveTranscription() {
  const session = useSignRecognitionSession();
  const {
    videoRef, canvasRef, cameraState, cameraError, modelState, modelError,
    sessionActive, paused, current, history, editingIndex, editDraft, setEditDraft,
    startSession, stopSession, pauseSession, resumeSession, clearSession,
    undoLast, deleteEvent, startEdit, saveEdit, cancelEdit, screenCaptureSupported,
  } = session;

  const [source, setSource] = useState<RecognitionSource>("webcam");
  const [notesMode, setNotesMode] = useState<NotesMode>("template");
  const [genPhase, setGenPhase] = useState<GenPhase>("idle");
  const [genError, setGenError] = useState<string | null>(null);
  const [frozenGlosses, setFrozenGlosses] = useState<string[]>([]);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [notes, setNotes] = useState<string | null>(null);

  const handleStart = async () => {
    setGenPhase("idle");
    setGenError(null);
    setTranscript(null);
    setNotes(null);
    await startSession(source);
  };

  const runGeneration = async (glosses: string[]) => {
    setGenPhase("generating");
    setGenError(null);
    const [transcriptResult, notesResult] = await Promise.allSettled([
      generateTranscriptFromGlosses(glosses, { notesMode }),
      generateNotesFromGlosses(glosses, { notesMode }),
    ]);

    if (transcriptResult.status === "fulfilled") setTranscript(transcriptResult.value.transcript);
    if (notesResult.status === "fulfilled") setNotes(notesResult.value.notes_md);

    if (transcriptResult.status === "rejected" && notesResult.status === "rejected") {
      // Section 31: keep the session, not lose it -- glosses stay in
      // frozenGlosses/history so the user can retry once the server's back.
      setGenError("Notes server unavailable. Your recognized glosses are preserved — you can retry below.");
      setGenPhase("error");
    } else {
      setGenPhase("completed");
    }
  };

  const handleStopAndGenerate = async () => {
    // Guard against a rapid double-click firing two generation requests
    // (project brief section 56).
    if (genPhase === "stopping" || genPhase === "generating") return;
    setGenPhase("stopping");
    setGenError(null);
    const glosses = history.map((e) => e.label);
    stopSession();
    setFrozenGlosses(glosses);

    if (glosses.length === 0) {
      setGenError("No signs were recognized during this session -- nothing to generate.");
      setGenPhase("error");
      return;
    }
    await runGeneration(glosses);
  };

  const handleRetry = async () => {
    if (genPhase === "generating" || frozenGlosses.length === 0) return;
    await runGeneration(frozenGlosses);
  };

  const handleStartNewSession = () => {
    clearSession();
    setGenPhase("idle");
    setGenError(null);
    setFrozenGlosses([]);
    setTranscript(null);
    setNotes(null);
  };

  const isLive = sessionActive;
  const canStart = !isLive && genPhase !== "stopping" && genPhase !== "generating" && modelState === "ready";
  const glossesToShow = genPhase === "completed" || genPhase === "error" ? frozenGlosses : history.map((e) => e.label);

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Live Transcription</h1>
        <p className="text-sm text-muted-foreground">
          Sign continuously and watch a live gloss transcript build up. When you're done, <strong>Stop & Generate</strong> turns it
          into a natural-language transcript and notes in one step — the LLM is never called while you're still signing.
        </p>
      </div>

      <PrivacyBanner variant="transcription" />

      {modelState === "error" && (
        <Card className="border-destructive/40 bg-destructive/5">
          <CardContent className="flex items-start gap-2 py-4 text-sm">
            <AlertTriangle className="h-4 w-4 shrink-0 text-destructive mt-0.5" />
            <span>{modelError}</span>
          </CardContent>
        </Card>
      )}

      {!isLive && genPhase === "idle" && (
        <Card>
          <CardContent className="flex flex-wrap items-center gap-4 py-4">
            <span className="text-sm font-medium">Source:</span>
            <div className="flex gap-2">
              <Button variant={source === "webcam" ? "default" : "outline"} size="sm" onClick={() => setSource("webcam")}>
                <Camera className="h-4 w-4 mr-2" />
                Webcam
              </Button>
              <Button
                variant={source === "screen" ? "default" : "outline"}
                size="sm"
                onClick={() => setSource("screen")}
                disabled={!screenCaptureSupported}
                title={
                  screenCaptureSupported
                    ? "Capture a tab, window, or screen you choose in your browser's own picker"
                    : "Not supported in this browser"
                }
              >
                <Monitor className="h-4 w-4 mr-2" />
                Screen / Tab
              </Button>
            </div>
            {source === "screen" && (
              <span className="text-xs text-muted-foreground">
                You'll pick exactly what to share (a tab, a window, or your screen) in the browser's own dialog — useful for
                transcribing sign-language content already playing on your device.
              </span>
            )}
            {!screenCaptureSupported && source === "screen" && (
              <span className="text-xs text-amber-600">Screen/tab capture isn't supported in this browser.</span>
            )}
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-6">
        {/* LIVE TRANSCRIPT -- the primary element of this mode */}
        <Card>
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-base flex items-center gap-2">
              <MessageSquareText className="h-5 w-5 text-primary" />
              {genPhase === "completed" || genPhase === "error" ? "Recognized Gloss Transcript" : "Live Transcript"}
            </CardTitle>
            {(genPhase === "idle") && (
              <Button onClick={undoLast} variant="ghost" size="sm" disabled={history.length === 0}>
                <Undo2 className="h-3.5 w-3.5 mr-1.5" />
                Undo
              </Button>
            )}
          </CardHeader>
          <CardContent>
            {glossesToShow.length === 0 ? (
              <p className="text-sm text-muted-foreground py-6 text-center">
                {isLive ? "Sign to begin building the transcript…" : "No signs recognized yet."}
              </p>
            ) : (
              <div className="flex flex-wrap items-center gap-2 min-h-[4rem]">
                {glossesToShow.map((label, i) => (
                  <span key={i} className="group flex items-center gap-1">
                    {editingIndex === i && genPhase === "idle" ? (
                      <span className="flex items-center gap-1">
                        <input
                          autoFocus
                          value={editDraft}
                          onChange={(ev) => setEditDraft(ev.target.value)}
                          onKeyDown={(ev) => {
                            if (ev.key === "Enter") saveEdit(i);
                            if (ev.key === "Escape") cancelEdit();
                          }}
                          className="w-28 rounded border bg-background px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
                        />
                        <button onClick={() => saveEdit(i)} className="text-emerald-600 hover:text-emerald-700" aria-label="Save">
                          <Check className="h-3.5 w-3.5" />
                        </button>
                        <button onClick={cancelEdit} className="text-muted-foreground hover:text-foreground" aria-label="Cancel">
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </span>
                    ) : (
                      <Badge
                        variant="secondary"
                        className={`text-sm px-3 py-1 ${genPhase === "idle" ? "cursor-pointer hover:bg-secondary/70" : ""}`}
                        onClick={() => genPhase === "idle" && startEdit(i)}
                      >
                        {label}
                      </Badge>
                    )}
                    {genPhase === "idle" && editingIndex !== i && (
                      <button
                        onClick={() => deleteEvent(i)}
                        className="opacity-0 transition-opacity group-hover:opacity-100 text-muted-foreground hover:text-destructive"
                        aria-label="Delete"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    )}
                    {i < glossesToShow.length - 1 && <span className="text-muted-foreground">→</span>}
                  </span>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Camera preview + current sign -- secondary to the transcript in this mode */}
        <div className="space-y-4">
          <Card>
            <CardContent className="p-3 space-y-3">
              <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-black/90">
                <video ref={videoRef} className="h-full w-full object-cover -scale-x-100" muted playsInline />
                <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100" />
                {!isLive && (
                  <div className="absolute inset-0 flex items-center justify-center text-xs text-white/60">
                    {cameraState === "requesting" ? "Requesting access…" : "Not active"}
                  </div>
                )}
                {paused && isLive && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/50 text-xs font-medium text-white">
                    Paused
                  </div>
                )}
              </div>
              {current && current !== "uncertain" && current !== "no_sign" && (
                <div className="text-sm font-semibold flex items-center gap-2">
                  <Hand className="h-4 w-4 text-accent" />
                  {current.label}
                  <span className="text-xs font-normal text-muted-foreground">{(current.confidence * 100).toFixed(0)}%</span>
                </div>
              )}
              {current === "uncertain" && (
                <div className="text-xs text-amber-600 flex items-center gap-1.5">
                  <AlertTriangle className="h-3.5 w-3.5" />
                  Uncertain — hold steady
                </div>
              )}
            </CardContent>
          </Card>

          {cameraError && (
            <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs">
              <AlertTriangle className="h-4 w-4 shrink-0 text-destructive mt-0.5" />
              <span>{cameraError}</span>
            </div>
          )}

          <div className="flex flex-wrap gap-2">
            {canStart && (
              <Button onClick={handleStart} className="flex-1">
                {source === "screen" ? <Monitor className="h-4 w-4 mr-2" /> : <Camera className="h-4 w-4 mr-2" />}
                Start
              </Button>
            )}
            {isLive && (
              <>
                {paused ? (
                  <Button onClick={resumeSession} variant="secondary" size="sm">
                    <Play className="h-3.5 w-3.5 mr-1.5" />
                    Resume
                  </Button>
                ) : (
                  <Button onClick={pauseSession} variant="secondary" size="sm">
                    <PauseIcon className="h-3.5 w-3.5 mr-1.5" />
                    Pause
                  </Button>
                )}
                <select
                  value={notesMode}
                  onChange={(e) => setNotesMode(e.target.value as NotesMode)}
                  className="rounded-lg border bg-background px-2 py-1 text-xs"
                >
                  <option value="template">Deterministic</option>
                  <option value="llm">Local LLM</option>
                </select>
              </>
            )}
          </div>

          {isLive && (
            <Button
              onClick={handleStopAndGenerate}
              disabled={genPhase === "stopping" || genPhase === "generating"}
              className="w-full"
              variant="default"
            >
              <Square className="h-4 w-4 mr-2" />
              Stop & Generate
            </Button>
          )}
        </div>
      </div>

      {genPhase === "generating" && (
        <Card>
          <CardContent className="flex items-center gap-2 py-4 text-sm text-muted-foreground">
            <Sparkles className="h-4 w-4 animate-pulse" />
            Converting the recognized signs into a transcript and notes…
          </CardContent>
        </Card>
      )}

      {genError && (
        <div className="flex items-start justify-between gap-2 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm">
          <div className="flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 shrink-0 text-destructive mt-0.5" />
            <span>{genError}</span>
          </div>
          {frozenGlosses.length > 0 && (
            <Button size="sm" variant="outline" onClick={handleRetry} disabled={genPhase === "generating"}>
              <RotateCcw className="h-3.5 w-3.5 mr-1.5" />
              Retry
            </Button>
          )}
        </div>
      )}

      {transcript && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <MessageSquareText className="h-5 w-5 text-primary" />
              Natural-Language Transcript
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm leading-relaxed whitespace-pre-line">{transcript}</p>
          </CardContent>
        </Card>
      )}

      {notes && <NotesPanel markdown={notes} title="Live Transcription Notes" />}

      {(genPhase === "completed" || genPhase === "error") && (
        <Button onClick={handleStartNewSession} variant="outline">
          <RotateCcw className="h-4 w-4 mr-2" />
          Start New Session
        </Button>
      )}
    </div>
  );
}
