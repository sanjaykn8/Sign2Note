import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FileText, Pencil, Check, X, RefreshCw, Download } from "lucide-react";
import { renderMarkdown } from "@/lib/markdown";
import { downloadMarkdown, downloadPlainText, downloadPdf, slugify } from "@/lib/notesExport";

interface NotesPanelProps {
  /** The generated Markdown. When this prop changes (e.g. after a
   * regenerate), the panel's editable copy re-syncs to it. */
  markdown: string;
  title?: string;
  durationSeconds?: number;
  onRegenerate?: () => void;
  regenerating?: boolean;
}

/**
 * "Preview -> Edit -> Export" notes panel (workplan section 7/8): renders
 * the notes as Markdown, lets the user fix misrecognized signs inline
 * before committing to a download, and offers Markdown/Text/PDF export.
 * Deliberately NOT "Generate -> immediately download" -- the student
 * reviews first.
 */
export default function NotesPanel({ markdown, title = "Lecture Notes", durationSeconds, onRegenerate, regenerating }: NotesPanelProps) {
  const [current, setCurrent] = useState(markdown);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(markdown);
  const [pdfBusy, setPdfBusy] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  // Re-sync when the parent hands us freshly (re)generated notes.
  useEffect(() => {
    setCurrent(markdown);
    setDraft(markdown);
    setEditing(false);
  }, [markdown]);

  const slug = slugify(title);

  const saveEdit = () => {
    setCurrent(draft);
    setEditing(false);
  };
  const cancelEdit = () => {
    setDraft(current);
    setEditing(false);
  };

  const handlePdf = async () => {
    setPdfBusy(true);
    setPdfError(null);
    try {
      await downloadPdf(`${slug}.pdf`, current, { title, durationSeconds });
    } catch (err) {
      setPdfError(err instanceof Error ? err.message : "Couldn't generate the PDF.");
    } finally {
      setPdfBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3 flex flex-row flex-wrap items-center justify-between gap-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <FileText className="h-5 w-5 text-primary" />
          {title}
        </CardTitle>
        <div className="flex flex-wrap gap-2">
          {!editing ? (
            <Button size="sm" variant="outline" onClick={() => setEditing(true)}>
              <Pencil className="h-3.5 w-3.5 mr-1.5" />
              Edit
            </Button>
          ) : (
            <>
              <Button size="sm" variant="outline" onClick={cancelEdit}>
                <X className="h-3.5 w-3.5 mr-1.5" />
                Cancel
              </Button>
              <Button size="sm" onClick={saveEdit}>
                <Check className="h-3.5 w-3.5 mr-1.5" />
                Save
              </Button>
            </>
          )}
          {onRegenerate && (
            <Button size="sm" variant="outline" onClick={onRegenerate} disabled={regenerating}>
              <RefreshCw className={`h-3.5 w-3.5 mr-1.5 ${regenerating ? "animate-spin" : ""}`} />
              {regenerating ? "Regenerating…" : "Regenerate"}
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {editing ? (
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={14}
            className="w-full rounded-lg border bg-background px-3 py-2 text-sm font-mono leading-relaxed focus:outline-none focus:ring-2 focus:ring-ring"
            autoFocus
          />
        ) : (
          <div
            className="prose prose-sm max-w-none text-foreground leading-relaxed"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(current) }}
          />
        )}

        <div className="flex flex-wrap items-center gap-2 border-t border-border pt-3">
          <span className="text-xs font-medium text-muted-foreground mr-1">Export:</span>
          <Button size="sm" variant="secondary" onClick={() => downloadMarkdown(`${slug}.md`, current)}>
            <Download className="h-3.5 w-3.5 mr-1.5" />
            Markdown
          </Button>
          <Button size="sm" variant="secondary" onClick={() => downloadPlainText(`${slug}.txt`, current, { title, durationSeconds })}>
            <Download className="h-3.5 w-3.5 mr-1.5" />
            Text
          </Button>
          <Button size="sm" variant="secondary" onClick={handlePdf} disabled={pdfBusy}>
            <Download className="h-3.5 w-3.5 mr-1.5" />
            {pdfBusy ? "Preparing…" : "PDF"}
          </Button>
        </div>
        {pdfError && <p className="text-xs text-destructive">{pdfError}</p>}
      </CardContent>
    </Card>
  );
}
