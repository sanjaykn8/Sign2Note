import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Cpu, Server, Loader2 } from "lucide-react";
import type { UseSignRecognitionSession } from "@/lib/useSignRecognitionSession";

/** Model loading state, which recognizer is active (and what data leaves
 * the device because of it), and recognition errors -- shared by both live
 * modes so the wording can't drift. */
export default function RecognizerStatus({ session }: { session: UseSignRecognitionSession }) {
  const { modelState, modelError, recognizer, recognizerNote, modelMeta, recognitionError, retryModelLoad } = session;

  if (modelState === "loading" || modelState === "idle") {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
        Loading the recognition model…
      </div>
    );
  }
  if (modelState === "error") {
    return (
      <Card className="border-destructive/40 bg-destructive/5">
        <CardContent className="flex flex-wrap items-start justify-between gap-2 py-4 text-sm" role="alert">
          <span className="flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 shrink-0 text-destructive mt-0.5" aria-hidden />
            {modelError}
          </span>
          <Button size="sm" variant="outline" onClick={retryModelLoad}>
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground" role="status">
        {recognizer === "local" ? <Cpu className="h-3.5 w-3.5" aria-hidden /> : <Server className="h-3.5 w-3.5" aria-hidden />}
        <span>
          {recognizer === "local"
            ? "Recognition runs in this browser. Only recognized gloss words are sent to the server, when you generate notes."
            : "Recognition runs on the server: landmark coordinates (not video) are sent to it while you sign."}
          {modelMeta?.model_version ? ` Model ${modelMeta.model_version}` : ""}
          {modelMeta?.feature_schema_version ? `, feature schema v${modelMeta.feature_schema_version}` : ""}
          {modelMeta?.is_legacy_schema ? " (legacy hand-only model)" : ""}.
        </span>
      </div>
      {recognizerNote && <p className="text-xs text-amber-700">{recognizerNote}</p>}
      {recognitionError && (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs" role="alert">
          <AlertTriangle className="h-4 w-4 shrink-0 text-destructive" aria-hidden />
          {recognitionError}
        </div>
      )}
    </div>
  );
}
