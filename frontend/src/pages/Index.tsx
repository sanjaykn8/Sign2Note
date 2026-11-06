import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import VideoDropzone from "@/components/VideoDropzone";
import ProcessingState, { type ProcessingPhase } from "@/components/ProcessingState";
import ResultsPanel from "@/components/ResultsPanel";
import PrivacyBanner from "@/components/PrivacyBanner";
import { uploadVideo, validateVideoFile, type ProcessResult, type NotesMode, type NoteStyle } from "@/lib/api";
import { ApiError } from "@/lib/serverConfig";
import { ArrowRight, Sparkles, Video, FileText, Hand, ChevronDown } from "lucide-react";

export default function Index() {
  const [file, setFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<ProcessingPhase | null>(null);
  const [uploadFraction, setUploadFraction] = useState(0);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [error, setError] = useState("");
  const [notesMode, setNotesMode] = useState<NotesMode>("template");
  const [style, setStyle] = useState<NoteStyle>("concise");
  const [threshold, setThreshold] = useState(0.55);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const selectFile = (f: File | null) => {
    setError("");
    if (f) {
      const problem = validateVideoFile(f);
      if (problem) {
        setError(problem);
        setFile(null);
        return;
      }
    }
    setFile(f);
  };

  const handleUpload = async () => {
    if (!file || phase) return; // no duplicate submissions
    setPhase("uploading");
    setUploadFraction(0);
    setError("");
    setResult(null);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const res = await uploadVideo(file, {
        notesMode,
        style,
        threshold,
        signal: controller.signal,
        onUploadProgress: (f) => {
          setUploadFraction(f);
          if (f >= 1) setPhase("processing");
        },
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof ApiError || e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setPhase(null);
      abortRef.current = null;
    }
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setError("");
  };

  return (
    <div className="min-h-screen">
      <header className="hero-gradient py-12 sm:py-16 px-4 text-center">
        <div className="mx-auto max-w-2xl space-y-4">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary-foreground/20 px-4 py-1.5 text-sm font-medium text-primary-foreground backdrop-blur-sm">
            <Sparkles className="h-4 w-4" aria-hidden />
            Indian Sign Language recognition → notes
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-primary-foreground sm:text-5xl">Sign2Notes</h1>
          <p className="text-lg text-primary-foreground/80">
            Upload a sign-language video: recognized signs (glosses) are turned into structured, editable notes.
          </p>
        </div>
      </header>

      <section className="border-b border-border bg-card py-8 px-4">
        <div className="mx-auto flex max-w-3xl flex-wrap items-center justify-center gap-8 text-center text-sm text-muted-foreground">
          {[
            { icon: Video, label: "Upload video" },
            { icon: Hand, label: "Recognize glosses" },
            { icon: FileText, label: "Generate notes" },
          ].map(({ icon: Icon, label }, i) => (
            <div key={label} className="flex items-center gap-3">
              {i > 0 && <ArrowRight className="h-4 w-4 text-border hidden sm:block" aria-hidden />}
              <div className="flex flex-col items-center gap-2">
                <div className="rounded-xl bg-muted p-3">
                  <Icon className="h-5 w-5 text-primary" aria-hidden />
                </div>
                <span className="font-medium text-foreground">{label}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <main className="mx-auto max-w-2xl px-4 py-10">
        <Card className="shadow-lg">
          <CardContent className="p-4 sm:p-6 space-y-6">
            {phase ? (
              <ProcessingState
                phase={phase}
                uploadFraction={uploadFraction}
                onCancel={phase === "uploading" ? () => abortRef.current?.abort() : undefined}
              />
            ) : result ? (
              <>
                <ResultsPanel result={result} notesMode={notesMode} initialStyle={style} />
                <Button variant="outline" onClick={reset} className="w-full">
                  Process another video
                </Button>
              </>
            ) : (
              <>
                <VideoDropzone file={file} onFileSelect={selectFile} onReject={(msg) => setError(msg)} />

                {error && (
                  <div className="rounded-lg bg-destructive/10 px-4 py-3 text-sm text-destructive" role="alert">
                    {error}
                  </div>
                )}

                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="space-y-1 text-xs font-medium">
                    Notes generator
                    <select
                      value={notesMode}
                      onChange={(e) => setNotesMode(e.target.value as NotesMode)}
                      className="block w-full rounded-lg border bg-background px-3 py-2 text-sm font-normal"
                    >
                      <option value="template">Deterministic (no LLM)</option>
                      <option value="llm">Local LLM (Ollama / llama.cpp)</option>
                    </select>
                  </label>
                  <label className="space-y-1 text-xs font-medium">
                    Notes style
                    <select
                      value={style}
                      onChange={(e) => setStyle(e.target.value as NoteStyle)}
                      className="block w-full rounded-lg border bg-background px-3 py-2 text-sm font-normal"
                    >
                      <option value="concise">Concise — quick revision</option>
                      <option value="detailed">Detailed — study notes</option>
                      <option value="academic">Academic — formal notes</option>
                    </select>
                  </label>
                </div>

                {notesMode === "llm" && (
                  <p className="text-xs text-muted-foreground">
                    Uses the language-generation service configured on the server. If it isn't reachable, the deterministic
                    notes are used instead and labeled as such.
                  </p>
                )}

                <div>
                  <button
                    type="button"
                    onClick={() => setShowAdvanced((v) => !v)}
                    aria-expanded={showAdvanced}
                    className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
                  >
                    <ChevronDown className={`h-3.5 w-3.5 transition-transform ${showAdvanced ? "rotate-180" : ""}`} aria-hidden />
                    Advanced settings
                  </button>
                  {showAdvanced && (
                    <div className="mt-2 space-y-2 rounded-lg border border-border p-3">
                      <label htmlFor="threshold" className="text-xs">
                        Confidence threshold: <strong>{threshold.toFixed(2)}</strong>
                      </label>
                      <input
                        id="threshold"
                        type="range"
                        min={0.1}
                        max={0.9}
                        step={0.05}
                        value={threshold}
                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                        className="w-full"
                      />
                      <p className="text-xs text-muted-foreground">
                        Windows below this confidence are not turned into glosses. Lower it if nothing is recognized; raise it to
                        reduce false glosses.
                      </p>
                    </div>
                  )}
                </div>

                <PrivacyBanner variant="upload" />

                <Button
                  onClick={handleUpload}
                  disabled={!file}
                  className="w-full hero-gradient text-primary-foreground font-semibold glow-primary"
                  size="lg"
                >
                  <Sparkles className="mr-2 h-4 w-4" aria-hidden />
                  Upload &amp; Generate Notes
                </Button>
              </>
            )}
          </CardContent>
        </Card>
      </main>

      <footer className="border-t border-border py-6 text-center text-sm text-muted-foreground">
        Sign2Notes — isolated-sign recognition with MediaPipe landmarks; not a complete ISL translator.
      </footer>
    </div>
  );
}
