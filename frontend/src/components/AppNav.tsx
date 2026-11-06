import { NavLink } from "@/components/NavLink";
import ConnectionStatus from "@/components/ConnectionStatus";
import { Upload, Video, MessageSquareText } from "lucide-react";

export default function AppNav() {
  const linkClass =
    "flex items-center gap-1.5 rounded-md px-2 sm:px-3 py-1.5 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring";
  const activeClass = "bg-muted text-foreground";

  return (
    <nav className="border-b bg-background/80 backdrop-blur sticky top-0 z-10" aria-label="Modes">
      <div className="mx-auto max-w-6xl px-4 py-3 flex flex-wrap items-center justify-between gap-2">
        <span className="font-bold text-lg">Sign2Notes</span>
        <div className="flex flex-wrap items-center gap-1">
          <NavLink to="/" end className={linkClass} activeClassName={activeClass}>
            <Upload className="h-4 w-4" aria-hidden />
            <span>Video Upload</span>
          </NavLink>
          <NavLink to="/webcam" className={linkClass} activeClassName={activeClass}>
            <Video className="h-4 w-4" aria-hidden />
            <span>Live Webcam</span>
          </NavLink>
          <NavLink to="/live-transcription" className={linkClass} activeClassName={activeClass}>
            <MessageSquareText className="h-4 w-4" aria-hidden />
            <span>Live Transcription</span>
          </NavLink>
          <ConnectionStatus />
        </div>
      </div>
    </nav>
  );
}
