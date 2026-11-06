import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";

export type ProcessingPhase = "uploading" | "processing";

/**
 * Honest progress: real byte progress while uploading; after that the
 * server does not report fine-grained progress, so we show an elapsed
 * timer and what the server is doing rather than a fake percentage.
 */
export default function ProcessingState({
  phase = "processing",
  uploadFraction = 0,
  onCancel,
}: {
  phase?: ProcessingPhase;
  uploadFraction?: number;
  onCancel?: () => void;
}) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    setElapsed(0);
    const start = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, [phase]);

  return (
    <div className="flex flex-col items-center gap-6 py-10" role="status" aria-live="polite">
      <div className="relative">
        <div className="h-16 w-16 rounded-full hero-gradient animate-pulse-slow" />
        <Loader2 className="absolute inset-0 m-auto h-8 w-8 text-primary-foreground animate-spin" aria-hidden />
      </div>
      {phase === "uploading" ? (
        <div className="w-full max-w-sm space-y-2 text-center">
          <p className="text-lg font-semibold text-foreground">Uploading video… {Math.round(uploadFraction * 100)}%</p>
          <Progress value={uploadFraction * 100} aria-label="Upload progress" />
        </div>
      ) : (
        <div className="text-center space-y-1">
          <p className="text-lg font-semibold text-foreground">Processing on the server</p>
          <p className="text-sm text-muted-foreground">
            Extracting hand, body and face landmarks → recognizing signs → generating notes
          </p>
          <p className="text-xs text-muted-foreground tabular-nums">
            {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")} elapsed · longer videos take proportionally longer
          </p>
        </div>
      )}
      {onCancel && (
        <Button variant="outline" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      )}
    </div>
  );
}
