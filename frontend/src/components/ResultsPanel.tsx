import type { ProcessResult } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FileText, Hand } from "lucide-react";

function renderMarkdown(md: string): string {
  return md
    .replace(/^### (.+)$/gm, "<h3 class='text-base font-semibold mt-4 mb-1'>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2 class='text-lg font-bold mt-5 mb-2'>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1 class='text-xl font-bold mt-6 mb-2'>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/^- (.+)$/gm, "<li class='ml-4 list-disc'>$1</li>")
    .replace(/\n{2,}/g, "<br/><br/>")
    .replace(/\n/g, "<br/>");
}

export default function ResultsPanel({ result }: { result: ProcessResult }) {
  return (
    <div className="space-y-5">
      {/* Glosses */}
      {result.gloss_list.length > 0 && (
        <Card className="border-accent/30 bg-accent/5">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Hand className="h-5 w-5 text-accent" />
              Detected Signs
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {result.gloss_list.map((g, i) => (
              <Badge key={i} variant="secondary" className="text-sm font-medium">
                {g}
              </Badge>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Notes */}
      {result.notes_md && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <FileText className="h-5 w-5 text-primary" />
              Generated Notes
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className="prose prose-sm max-w-none text-foreground leading-relaxed"
              dangerouslySetInnerHTML={{ __html: renderMarkdown(result.notes_md) }}
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
