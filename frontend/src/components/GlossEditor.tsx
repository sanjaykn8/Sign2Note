import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Check, Pencil, Trash2, Undo2, X } from "lucide-react";

export interface GlossItem {
  label: string;
  confidence?: number;
  start_time?: number;
  end_time?: number;
  edited?: boolean;
}

/** Below this, a gloss is flagged as uncertain in the UI (same cut-off the
 * notes service uses to mark "(?)"). Confidence is the model's own score,
 * not a guarantee of correctness. */
export const UNCERTAIN_BELOW = 0.6;

function fmt(t?: number) {
  if (t === undefined || t === null || !Number.isFinite(t)) return null;
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/**
 * Ordered gloss list with edit / delete / undo. Controlled: the parent owns
 * the list; every change goes through onChange, and undo restores the
 * previous list from a local history stack.
 */
export default function GlossEditor({
  items,
  onChange,
  disabled,
}: {
  items: GlossItem[];
  onChange: (next: GlossItem[]) => void;
  disabled?: boolean;
}) {
  const [undoStack, setUndoStack] = useState<GlossItem[][]>([]);
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState("");

  const commit = (next: GlossItem[]) => {
    setUndoStack((s) => [...s.slice(-49), items]);
    onChange(next);
  };
  const save = (i: number) => {
    const v = draft.trim();
    if (v && v !== items[i].label) commit(items.map((g, j) => (j === i ? { ...g, label: v, edited: true } : g)));
    setEditing(null);
  };
  const undo = () => {
    const prev = undoStack[undoStack.length - 1];
    if (!prev) return;
    setUndoStack((s) => s.slice(0, -1));
    setEditing(null);
    onChange(prev);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          {items.length} gloss{items.length === 1 ? "" : "es"} in recognized order. Edit or delete mistakes, then regenerate the notes.
        </p>
        <Button size="sm" variant="ghost" onClick={undo} disabled={disabled || undoStack.length === 0}>
          <Undo2 className="h-3.5 w-3.5 mr-1.5" aria-hidden />
          Undo
        </Button>
      </div>
      {items.length === 0 ? (
        <p className="text-sm text-muted-foreground py-2">No glosses. Undo to restore deleted ones.</p>
      ) : (
        <ol className="space-y-1 max-h-72 overflow-y-auto pr-1">
          {items.map((g, i) => {
            const t0 = fmt(g.start_time);
            const t1 = fmt(g.end_time);
            const uncertain = g.confidence !== undefined && g.confidence < UNCERTAIN_BELOW && !g.edited;
            return (
              <li key={i} className="flex items-center gap-2 text-sm">
                <span className="w-20 shrink-0 tabular-nums text-xs text-muted-foreground">{t0 ? `${t0}–${t1}` : `#${i + 1}`}</span>
                {editing === i ? (
                  <span className="flex flex-1 items-center gap-1">
                    <input
                      autoFocus
                      value={draft}
                      aria-label={`Edit gloss ${i + 1}`}
                      onChange={(e) => setDraft(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") save(i);
                        if (e.key === "Escape") setEditing(null);
                      }}
                      className="flex-1 rounded border bg-background px-2 py-0.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
                    />
                    <button onClick={() => save(i)} className="text-emerald-600 hover:text-emerald-700" aria-label="Save gloss">
                      <Check className="h-4 w-4" />
                    </button>
                    <button onClick={() => setEditing(null)} className="text-muted-foreground hover:text-foreground" aria-label="Cancel editing">
                      <X className="h-4 w-4" />
                    </button>
                  </span>
                ) : (
                  <>
                    <Badge variant="secondary" className={`flex-1 justify-start truncate ${uncertain ? "border border-amber-500/50" : ""}`}>
                      {g.label}
                      {uncertain && <span className="ml-1 text-amber-700" title="Low recognition confidence">(?)</span>}
                      {g.edited && <span className="ml-1 text-muted-foreground">(edited)</span>}
                    </Badge>
                    {g.confidence !== undefined && (
                      <span className="w-10 shrink-0 text-right text-xs text-muted-foreground" title="Model confidence (not a guarantee of correctness)">
                        {(g.confidence * 100).toFixed(0)}%
                      </span>
                    )}
                    <button
                      disabled={disabled}
                      onClick={() => {
                        setEditing(i);
                        setDraft(g.label);
                      }}
                      className="shrink-0 rounded p-1 text-muted-foreground hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-40"
                      aria-label={`Edit gloss ${g.label}`}
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                    <button
                      disabled={disabled}
                      onClick={() => commit(items.filter((_, j) => j !== i))}
                      className="shrink-0 rounded p-1 text-muted-foreground hover:text-destructive focus:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-40"
                      aria-label={`Delete gloss ${g.label}`}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
