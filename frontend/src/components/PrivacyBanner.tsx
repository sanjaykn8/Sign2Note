import { ShieldCheck } from "lucide-react";

/**
 * Shared privacy notice, used on both the upload page and the live webcam
 * page. Kept as one component so the wording can't drift between the two
 * places it's shown -- see PRIVACY.md for the full policy this summarizes.
 */
export default function PrivacyBanner({ variant = "upload" }: { variant?: "upload" | "webcam" }) {
  const text =
    variant === "webcam"
      ? "Privacy: your webcam video never leaves this browser tab. Hand landmarks and recognition run locally; only the recognized gloss words (not video or images) are sent to the backend when you click Generate Notes."
      : "Privacy: your video is processed locally and is not permanently stored or uploaded to a remote server. The uploaded file and extracted keypoints are deleted immediately after processing.";

  return (
    <div className="flex items-start gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-3 py-2 text-xs text-muted-foreground">
      <ShieldCheck className="h-4 w-4 shrink-0 text-emerald-600 mt-0.5" />
      <span>{text}</span>
    </div>
  );
}
