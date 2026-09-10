import type { ProcessResult } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import NotesPanel from "@/components/NotesPanel";
import { Hand, AlertTriangle } from "lucide-react";

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
              {result.low_confidence && (
                <Badge variant="outline" className="ml-auto gap-1 text-xs font-normal text-amber-600 border-amber-500/40">
                  <AlertTriangle className="h-3 w-3" />
                  Low confidence
                </Badge>
              )}
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

      {/* Notes -- preview, edit, and export as Markdown/Text/PDF */}
      {result.notes_md && <NotesPanel markdown={result.notes_md} title="Lecture Notes" />}
    </div>
  );
}
