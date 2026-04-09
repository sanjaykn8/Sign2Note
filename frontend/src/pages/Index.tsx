import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import VideoDropzone from "@/components/VideoDropzone";
import ProcessingState from "@/components/ProcessingState";
import ResultsPanel from "@/components/ResultsPanel";
import { uploadVideo, type ProcessResult } from "@/lib/api";
import { ArrowRight, Sparkles, Video, FileText, Hand } from "lucide-react";

export default function Index() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [error, setError] = useState("");

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await uploadVideo(file);
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
