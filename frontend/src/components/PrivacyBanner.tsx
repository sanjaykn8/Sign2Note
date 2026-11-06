import { ShieldCheck } from "lucide-react";

/**
 * Shared privacy notice, used on the upload, live webcam, and live
 * transcription pages. Kept as one component so the wording can't drift
 * between the places it's shown -- see PRIVACY.md for the full policy
 * this summarizes.
 */
export default function PrivacyBanner({ variant = "upload" }: { variant?: "upload" | "webcam" | "transcription" }) {
  const text =
    variant === "webcam"
      ? "Privacy: camera video never leaves this browser tab. Landmarks are extracted here; recognition runs here when possible, otherwise only landmark coordinates (not video) go to the main laptop — the line below says which. Only gloss words are sent when you click Generate Notes."
      : variant === "transcription"
        ? "Privacy: whether you use the webcam or share a tab/window/screen, the video never leaves this browser tab. Landmarks are extracted here; recognition runs here when possible, otherwise only landmark coordinates go to the main laptop. Only gloss words are sent when you click Stop & Generate."
        : "Privacy: the video is sent to the Sign2Notes server (the main laptop), processed, and deleted immediately afterwards together with its extracted landmarks. It is not stored or sent anywhere else.";

  return (
    <div className="flex items-start gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-3 py-2 text-xs text-muted-foreground">
      <ShieldCheck className="h-4 w-4 shrink-0 text-emerald-600 mt-0.5" />
      <span>{text}</span>
    </div>
  );
}
