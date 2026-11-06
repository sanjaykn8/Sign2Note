import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import NotesPanel from "@/components/NotesPanel";
import GlossEditor, { type GlossItem } from "@/components/GlossEditor";
import { generateNotesFromGlosses, type GenerationMeta, type NoteStyle, type NotesMode, type ProcessResult } from "@/lib/api";
import { Hand, AlertTriangle, RefreshCw } from "lucide-react";

function initialGlosses(result: ProcessResult): GlossItem[] {
  if (result.events && result.events.length > 0) {
    return result.events.map((e) => ({ label: e.label, confidence: e.confidence, start_time: e.start_time, end_time: e.end_time }));
  }
  return result.gloss_list.map((label) => ({ label }));
}

export default function ResultsPanel({
  result,
  notesMode = "template",
  initialStyle = "concise",
}: {
  result: ProcessResult;
  notesMode?: NotesMode;
  initialStyle?: NoteStyle;
}) {
  const [glosses, setGlosses] = useState<GlossItem[]>(() => initialGlosses(result));
  const [notes, setNotes] = useState(result.notes_md);
  const [generation, setGeneration] = useState<GenerationMeta | undefined>(result.generation);
  const [style, setStyle] = useState<NoteStyle>(result.generation?.style ?? initialStyle);
  const [mode, setMode] = useState<NotesMode>(notesMode);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stale, setStale] = useState(false);

  const regenerate = async () => {
    if (glosses.length === 0) {
      setError("There are no glosses left to generate notes from. Undo a deletion first.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const conf = glosses.every((g) => g.confidence !== undefined)
        ? glosses.map((g) => (g.edited ? 1 : (g.confidence as number)))
        : undefined;
      const res = await generateNotesFromGlosses(
        glosses.map((g) => g.label),
        { style, notesMode: mode, confidences: conf }
      );
      setNotes(res.notes_md);
      setGeneration(res.generation);
      setStale(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate notes.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-5">
      <Card className="border-accent/30 bg-accent/5">
        <CardHeader className="pb-3">
          <CardTitle className="flex flex-wrap items-center gap-2 text-base">
            <Hand className="h-5 w-5 text-accent" aria-hidden />
            Recognized glosses
            {result.low_confidence && (
              <Badge variant="outline" className="ml-auto gap-1 text-xs font-normal text-amber-700 border-amber-500/40">
                <AlertTriangle className="h-3 w-3" aria-hidden />
                No confident sign — best guess shown
              </Badge>
            )}
          </CardTitle>
          <p className="text-xs text-muted-foreground">
            Isolated-sign recognizer applied to sliding windows; glosses are the model's best guesses, not a verified translation.
            {result.model_version ? ` Model: ${result.model_version}` : ""}
            {result.feature_schema_version ? ` (feature schema v${result.feature_schema_version})` : ""}
            {result.video_duration_seconds ? ` · Video length ${Math.round(result.video_duration_seconds)} s` : ""}
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          <GlossEditor
            items={glosses}
            disabled={busy}
            onChange={(next) => {
              setGlosses(next);
              setStale(true);
            }}
          />
          <div className="flex flex-wrap items-center gap-2 border-t border-border pt-3">
            <label className="text-xs font-medium" htmlFor="regen-style">
              Notes style
            </label>
            <select
              id="regen-style"
              value={style}
              onChange={(e) => {
                setStyle(e.target.value as NoteStyle);
                setStale(true);
              }}
              className="rounded-lg border bg-background px-2 py-1 text-xs"
            >
              <option value="concise">Concise</option>
              <option value="detailed">Detailed</option>
              <option value="academic">Academic</option>
            </select>
            <select
              aria-label="Notes generator"
              value={mode}
              onChange={(e) => {
                setMode(e.target.value as NotesMode);
                setStale(true);
              }}
              className="rounded-lg border bg-background px-2 py-1 text-xs"
            >
              <option value="template">Deterministic</option>
              <option value="llm">Local LLM</option>
            </select>
            <Button size="sm" onClick={regenerate} disabled={busy}>
              <RefreshCw className={`h-3.5 w-3.5 mr-1.5 ${busy ? "animate-spin" : ""}`} aria-hidden />
              {busy ? "Generating…" : "Regenerate notes"}
            </Button>
            {stale && !busy && <span className="text-xs text-amber-700">Glosses or style changed — regenerate to update the notes.</span>}
          </div>
          {error && (
            <p className="text-xs text-destructive" role="alert">
              {error}
            </p>
          )}
        </CardContent>
      </Card>

      {notes && <NotesPanel markdown={notes} title="Lecture Notes" generation={generation} durationSeconds={result.video_duration_seconds ?? undefined} />}
    </div>
  );
}
