import { useCallback, useState } from "react";
import { Upload, FileVideo, X } from "lucide-react";

interface Props {
  file: File | null;
  onFileSelect: (file: File | null) => void;
  disabled?: boolean;
}

export default function VideoDropzone({ file, onFileSelect, disabled }: Props) {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const f = e.dataTransfer.files[0];
      if (f?.type.startsWith("video/")) onFileSelect(f);
    },
    [disabled, onFileSelect]
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) onFileSelect(f);
  };

  if (file) {
    return (
      <div className="relative flex items-center gap-4 rounded-xl border-2 border-primary/30 bg-primary/5 p-5">
        <FileVideo className="h-10 w-10 text-primary shrink-0" />
        <div className="min-w-0 flex-1">
          <p className="truncate font-semibold text-foreground">{file.name}</p>
          <p className="text-sm text-muted-foreground">
            {(file.size / 1024 / 1024).toFixed(1)} MB
          </p>
        </div>
        {!disabled && (
          <button
            onClick={() => onFileSelect(null)}
            className="rounded-full p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        )}
      </div>
    );
  }

  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 transition-all ${
        dragOver
          ? "border-primary bg-primary/10 scale-[1.01]"
          : "border-border hover:border-primary/50 hover:bg-muted/50"
      } ${disabled ? "pointer-events-none opacity-50" : ""}`}
    >
      <div className="rounded-full bg-primary/10 p-4">
        <Upload className="h-7 w-7 text-primary" />
      </div>
      <div className="text-center">
        <p className="font-semibold text-foreground">
          Drop your sign-language video here
        </p>
        <p className="mt-1 text-sm text-muted-foreground">
          or click to browse — MP4, MOV, AVI supported
        </p>
      </div>
      <input
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleFileInput}
        disabled={disabled}
      />
    </label>
  );
}
