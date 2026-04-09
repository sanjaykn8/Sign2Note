import { Loader2 } from "lucide-react";

const steps = [
  "Uploading video…",
  "Extracting hand keypoints…",
  "Recognizing signs…",
  "Generating notes…",
];

export default function ProcessingState() {
  return (
    <div className="flex flex-col items-center gap-6 py-12">
      <div className="relative">
        <div className="h-16 w-16 rounded-full hero-gradient animate-pulse-slow" />
        <Loader2 className="absolute inset-0 m-auto h-8 w-8 text-primary-foreground animate-spin" />
      </div>
      <div className="text-center">
        <p className="text-lg font-semibold text-foreground">Processing your video</p>
        <p className="mt-1 text-sm text-muted-foreground">This may take a minute or two</p>
      </div>
      <div className="flex flex-col gap-2 text-sm text-muted-foreground">
        {steps.map((s, i) => (
          <div key={i} className="flex items-center gap-2 animate-pulse-slow" style={{ animationDelay: `${i * 0.5}s` }}>
            <span className="h-1.5 w-1.5 rounded-full bg-primary/60" />
            {s}
          </div>
        ))}
      </div>
    </div>
  );
}
