import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import PrivacyBanner from "@/components/PrivacyBanner";
import NotesPanel from "@/components/NotesPanel";
import RecognizerStatus from "@/components/RecognizerStatus";
import { generateNotesFromGlosses, type GenerationMeta, type NoteStyle, type NotesMode } from "@/lib/api";
import { useSignRecognitionSession } from "@/lib/useSignRecognitionSession";
import { Camera, Square, Trash2, FileText, AlertTriangle, Hand, Pencil, Check, X, Undo2, Play, Pause as PauseIcon } from "lucide-react";

const actionBtn =
  "shrink-0 rounded p-1 text-muted-foreground transition-opacity hover:text-foreground focus:opacity-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:opacity-0 sm:group-hover:opacity-100";

export default function Webcam() {
  const session = useSignRecognitionSession();
  const {
    videoRef, canvasRef, cameraState, cameraError, modelState,
    sessionActive, paused, current, history, editingIndex, editDraft, setEditDraft,
    startSession, stopSession, pauseSession, resumeSession, clearSession,
    undoLast, deleteEvent, startEdit, saveEdit, cancelEdit, elapsedSessionSeconds,
  } = session;

  const [notes, setNotes] = useState<string | null>(null);
  const [generation, setGeneration] = useState<GenerationMeta | undefined>(undefined);
  const [notesDuration, setNotesDuration] = useState<number | undefined>(undefined);
  const [notesMode, setNotesMode] = useState<NotesMode>("template");
  const [style, setStyle] = useState<NoteStyle>("concise");
  const [generating, setGenerating] = useState(false);
  const [notesError, setNotesError] = useState<string | null>(null);

  const handleStartSession = async () => {
    setNotes(null);
    setNotesDuration(undefined);
    setNotesError(null);
    await startSession("webcam");
  };

  const handleClearSession = () => {
    clearSession();
    setNotes(null);
    setNotesDuration(undefined);
    setNotesError(null);
  };

  // Notes are generated ONLY on request -- never per recognized sign.
  const handleGenerateNotes = async () => {
    if (history.length === 0 || generating) return;
    setGenerating(true);
    setNotesError(null);
    try {
      const res = await generateNotesFromGlosses(
        history.map((e) => e.label),
        { notesMode, style, confidences: history.map((e) => (e.edited ? 1 : e.confidence)) }
      );
      setNotes(res.notes_md);
      setGeneration(res.generation);
      setNotesDuration(sessionActive ? elapsedSessionSeconds() : history[history.length - 1]?.timestamp);
    } catch (err) {
      setNotesError(err instanceof Error ? err.message : "Failed to generate notes.");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Live Webcam</h1>
        <p className="text-sm text-muted-foreground">
          Sign in front of your camera. Each stable recognized sign is added to the gloss history; edit it as you go and
          generate notes when you're ready.
        </p>
      </div>

      <PrivacyBanner variant="webcam" />
      <RecognizerStatus session={session} />

      <div className="grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6">
        <Card>
          <CardContent className="p-4 space-y-4">
            <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-black/90">
              <video ref={videoRef} className="h-full w-full object-cover -scale-x-100" muted playsInline aria-label="Camera preview" />
              <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100" aria-hidden />
              {cameraState !== "active" && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-white/60">
                  {cameraState === "requesting" ? "Requesting camera access…" : "Camera is off"}
                </div>
              )}
              {paused && cameraState === "active" && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 text-sm font-medium text-white">
                  <PauseIcon className="h-4 w-4 mr-2" aria-hidden />
                  Paused — recognition is not running
                </div>
              )}
            </div>

            {cameraError && (
              <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm" role="alert">
                <AlertTriangle className="h-4 w-4 shrink-0 text-destructive mt-0.5" aria-hidden />
                <span>{cameraError}</span>
              </div>
            )}

            <div className="flex flex-wrap gap-2">
              {!sessionActive ? (
                <Button onClick={handleStartSession} disabled={modelState !== "ready" || cameraState === "requesting"}>
                  <Camera className="h-4 w-4 mr-2" aria-hidden />
                  Start Session
                </Button>
              ) : (
                <>
                  <Button onClick={stopSession} variant="secondary">
                    <Square className="h-4 w-4 mr-2" aria-hidden />
                    Stop Session
                  </Button>
                  {paused ? (
                    <Button onClick={resumeSession} variant="secondary">
                      <Play className="h-4 w-4 mr-2" aria-hidden />
                      Resume
                    </Button>
                  ) : (
                    <Button onClick={pauseSession} variant="secondary">
                      <PauseIcon className="h-4 w-4 mr-2" aria-hidden />
                      Pause
                    </Button>
                  )}
                </>
              )}
              <Button onClick={undoLast} variant="outline" disabled={history.length === 0}>
                <Undo2 className="h-4 w-4 mr-2" aria-hidden />
                Undo
              </Button>
              <Button onClick={handleClearSession} variant="outline" disabled={history.length === 0 && !notes}>
                <Trash2 className="h-4 w-4 mr-2" aria-hidden />
                Clear Session
              </Button>
            </div>
            <div className="flex flex-wrap items-center gap-2 border-t border-border pt-3">
              <select
                aria-label="Notes generator"
                value={notesMode}
                onChange={(e) => setNotesMode(e.target.value as NotesMode)}
                className="rounded-lg border bg-background px-3 py-2 text-sm"
              >
                <option value="template">Deterministic notes</option>
                <option value="llm">Local LLM</option>
              </select>
              <select
                aria-label="Notes style"
                value={style}
                onChange={(e) => setStyle(e.target.value as NoteStyle)}
                className="rounded-lg border bg-background px-3 py-2 text-sm"
              >
                <option value="concise">Concise</option>
                <option value="detailed">Detailed</option>
                <option value="academic">Academic</option>
              </select>
              <Button onClick={handleGenerateNotes} disabled={history.length === 0 || generating} variant="default">
                <FileText className="h-4 w-4 mr-2" aria-hidden />
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
            <CardContent className="space-y-2" aria-live="polite">
              {current === null && (
                <p className="text-sm text-muted-foreground">{paused ? "Recognition paused." : "Start a session to begin recognizing signs."}</p>
              )}
              {current === "no_sign" && (
                <div className="flex items-center gap-2 text-muted-foreground text-sm">
                  <Hand className="h-4 w-4 opacity-40" aria-hidden />
                  No sign detected
                </div>
              )}
              {current === "uncertain" && (
                <div className="flex items-center gap-2 text-amber-700 text-sm">
                  <AlertTriangle className="h-4 w-4" aria-hidden />
                  Uncertain — hold the sign steady
                </div>
              )}
              {current && current !== "uncertain" && current !== "no_sign" && (
                <>
                  <div className="text-xl font-bold flex items-center gap-2">
                    <Hand className="h-5 w-5 text-accent" aria-hidden />
                    {current.label}
                  </div>
                  <Progress value={current.confidence * 100} aria-label="Model confidence" />
                  <p className="text-xs text-muted-foreground">
                    Model confidence: {(current.confidence * 100).toFixed(0)}% (a score, not a guarantee)
                  </p>
                </>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Gloss History ({history.length})</CardTitle>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <p className="text-sm text-muted-foreground">No signs recognized yet.</p>
              ) : (
                <ol className="space-y-1 max-h-64 overflow-y-auto text-sm">
                  {history.map((e, i) => (
                    <li key={i} className="group flex items-center gap-2">
                      <span className="tabular-nums text-muted-foreground shrink-0 w-12">{formatTime(e.timestamp)}</span>
                      {editingIndex === i ? (
                        <div className="flex flex-1 items-center gap-1">
                          <input
                            autoFocus
                            value={editDraft}
                            aria-label={`Edit gloss ${i + 1}`}
                            onChange={(ev) => setEditDraft(ev.target.value)}
                            onKeyDown={(ev) => {
                              if (ev.key === "Enter") saveEdit(i);
                              if (ev.key === "Escape") cancelEdit();
                            }}
                            className="flex-1 rounded border bg-background px-2 py-0.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
                          />
                          <button onClick={() => saveEdit(i)} className="text-emerald-600 hover:text-emerald-700" aria-label="Save gloss">
                            <Check className="h-3.5 w-3.5" />
                          </button>
                          <button onClick={cancelEdit} className="text-muted-foreground hover:text-foreground" aria-label="Cancel editing">
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      ) : (
                        <>
                          <Badge variant="secondary" className="flex-1 justify-center truncate">
                            {e.label}
                            {e.edited && <span className="ml-1 text-muted-foreground">(edited)</span>}
                          </Badge>
                          <span className="text-xs text-muted-foreground shrink-0 w-9 text-right">{(e.confidence * 100).toFixed(0)}%</span>
                          <button onClick={() => startEdit(i)} className={actionBtn} aria-label={`Edit gloss ${e.label}`}>
                            <Pencil className="h-3.5 w-3.5" />
                          </button>
                          <button onClick={() => deleteEvent(i)} className={`${actionBtn} hover:text-destructive`} aria-label={`Delete gloss ${e.label}`}>
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </>
                      )}
                    </li>
                  ))}
                </ol>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {notesError && (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm" role="alert">
          <AlertTriangle className="h-4 w-4 shrink-0 text-destructive mt-0.5" aria-hidden />
          <span>{notesError} Your gloss history is kept — try again.</span>
        </div>
      )}

      {notes && (
        <NotesPanel
          markdown={notes}
          title="Live Session Notes"
          durationSeconds={notesDuration}
          onRegenerate={handleGenerateNotes}
          regenerating={generating}
          generation={generation}
        />
      )}
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
