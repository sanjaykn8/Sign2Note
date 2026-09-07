import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import VideoDropzone from "@/components/VideoDropzone";
import ProcessingState from "@/components/ProcessingState";
import ResultsPanel from "@/components/ResultsPanel";
import PrivacyBanner from "@/components/PrivacyBanner";
import { uploadVideo, type ProcessResult, type NotesMode } from "@/lib/api";
import { ArrowRight, Sparkles, Video, FileText, Hand, ChevronDown } from "lucide-react";

export default function Index() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [error, setError] = useState("");
  const [notesMode, setNotesMode] = useState<NotesMode>("template");
  const [style, setStyle] = useState<"concise" | "detailed" | "academic">("concise");
  const [threshold, setThreshold] = useState(0.55);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await uploadVideo(file, { notesMode, style, threshold });
      setResult(res);
    } catch (e: any) {
      setError(e.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setError("");
  };

  return (
    <div className="min-h-screen">
      {/* Hero */}
      <header className="hero-gradient py-16 px-4 text-center">
        <div className="mx-auto max-w-2xl space-y-4">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary-foreground/20 px-4 py-1.5 text-sm font-medium text-primary-foreground backdrop-blur-sm">
            <Sparkles className="h-4 w-4" />
            AI-Powered Sign Language Recognition
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-primary-foreground sm:text-5xl">
            Sign2Notes
          </h1>
          <p className="text-lg text-primary-foreground/80">
            Upload a sign-language video and get structured, readable notes in seconds.
          </p>
        </div>
      </header>

      {/* How it works */}
      <section className="border-b border-border bg-card py-10 px-4">
        <div className="mx-auto flex max-w-3xl flex-wrap items-center justify-center gap-8 text-center text-sm text-muted-foreground">
          {[
            { icon: Video, label: "Upload Video" },
            { icon: Hand, label: "Detect Signs" },
            { icon: FileText, label: "Generate Notes" },
          ].map(({ icon: Icon, label }, i) => (
            <div key={label} className="flex items-center gap-3">
              {i > 0 && <ArrowRight className="h-4 w-4 text-border hidden sm:block" />}
              <div className="flex flex-col items-center gap-2">
                <div className="rounded-xl bg-muted p-3">
                  <Icon className="h-5 w-5 text-primary" />
                </div>
                <span className="font-medium text-foreground">{label}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Main */}
      <main className="mx-auto max-w-2xl px-4 py-12">
        <Card className="shadow-lg">
          <CardContent className="p-6 space-y-6">
            {loading ? (
              <ProcessingState />
            ) : result ? (
              <>
                <ResultsPanel result={result} />
                <Button variant="outline" onClick={reset} className="w-full">
                  Process another video
                </Button>
              </>
            ) : (
              <>
                <VideoDropzone file={file} onFileSelect={setFile} />

                {error && (
                  <div className="rounded-lg bg-destructive/10 px-4 py-3 text-sm text-destructive">
                    {error}
                  </div>
                )}

                <div className="grid gap-3 sm:grid-cols-2">
                  <select value={notesMode} onChange={(e) => setNotesMode(e.target.value as any)} className="rounded-lg border bg-background px-3 py-2 text-sm">
                    <option value="template">Deterministic notes</option>
                    <option value="llm">Local LLM (Ollama / llama.cpp)</option>
                  </select>
                  <select value={style} onChange={(e) => setStyle(e.target.value as any)} className="rounded-lg border bg-background px-3 py-2 text-sm">
                    <option value="concise">Concise</option>
                    <option value="detailed">Detailed</option>
                    <option value="academic">Academic</option>
                  </select>
                </div>

                {notesMode === "llm" && (
                  <p className="text-xs text-muted-foreground">
                    Requires a local Ollama or llama.cpp server (see .env for LLM_PROVIDER/LLM_MODEL/LLM_BASE_URL). If it isn't reachable, notes fall back to the deterministic mode automatically.
                  </p>
                )}

                <div>
                  <button
                    type="button"
                    onClick={() => setShowAdvanced((v) => !v)}
                    className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
                  >
                    <ChevronDown className={`h-3.5 w-3.5 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
                    Advanced settings
                  </button>
                  {showAdvanced && (
                    <div className="mt-2 space-y-2 rounded-lg border border-border p-3">
                      <div className="flex items-center justify-between text-xs">
                        <label htmlFor="threshold">
                          Confidence threshold: <strong>{threshold.toFixed(2)}</strong>
                        </label>
                      </div>
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
                        Lower this if you're getting "No confident signs detected" — a small demo-scale model often predicts correctly with confidence below the 0.55 default.
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
                  <Sparkles className="mr-2 h-4 w-4" />
                  Upload &amp; Generate Notes
                </Button>
              </>
            )}
          </CardContent>
        </Card>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-6 text-center text-sm text-muted-foreground">
        Sign2Notes — Powered by MediaPipe &amp; Deep Learning
      </footer>
    </div>
  );
}
