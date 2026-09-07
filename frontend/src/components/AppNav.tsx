import { NavLink } from "@/components/NavLink";
import { Upload, Video } from "lucide-react";

export default function AppNav() {
  const linkClass =
    "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors";
  const activeClass = "bg-muted text-foreground";

  return (
    <nav className="border-b bg-background/80 backdrop-blur sticky top-0 z-10">
      <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
        <span className="font-bold text-lg">Sign2Notes</span>
        <div className="flex gap-1">
          <NavLink to="/" end className={linkClass} activeClassName={activeClass}>
            <Upload className="h-4 w-4" />
            Upload Video
          </NavLink>
          <NavLink to="/webcam" className={linkClass} activeClassName={activeClass}>
            <Video className="h-4 w-4" />
            Live Webcam
          </NavLink>
        </div>
      </div>
    </nav>
  );
}
