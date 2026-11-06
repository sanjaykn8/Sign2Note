import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import JSZip from "jszip";
import { buildDocxDocument, parseNoteLines, unsupportedPdfCharacters, exportFilename, markdownToPlainText } from "../notesExport";
import { renderMarkdown } from "../markdown";
import { KeypointBuffer, SessionSmoother } from "../webcamPipeline";
import { buildFeatureVector, featureDimForSchema, legacyV1FromV2, toSchema, FEATURE_DIM, LEGACY_FEATURE_DIM } from "../featureSchema";
import { apiFetch, defaultServerUrl, getServerUrl, setApiToken, setServerUrl, ApiError } from "../serverConfig";

async function docxXml(markdown: string): Promise<string> {
  const { Packer } = await import("docx");
  const buf = await Packer.toBuffer(await buildDocxDocument(markdown, { title: "T", durationSeconds: 60 }));
  const zip = await JSZip.loadAsync(buf);
  return zip.file("word/document.xml")!.async("string");
}

function texts(xml: string): string[] {
  return [...xml.matchAll(/<w:t(?: [^>]*)?>([^<]*)<\/w:t>/g)].map((m) => m[1]);
}

describe("DOCX export contains exactly the displayed notes", () => {
  const notes = [
    "# Lecture Notes",
    "",
    "## Key Concepts",
    "- **Definition** discussed",
    "1. first",
    "2. second",
    "### Detail",
    "Plain paragraph with *italic*.",
  ].join("\n");

  it("adds no extra title or date paragraph (regression: a second title + 'Date:' line was prepended)", async () => {
    const xml = await docxXml(notes);
    const t = texts(xml);
    expect(t.join("|")).not.toMatch(/Date:/);
    expect(t.filter((x) => x === "Lecture Notes")).toHaveLength(1);
    expect(t[0]).toBe("Lecture Notes");
  });

  it("maps heading levels and keeps lists", async () => {
    const xml = await docxXml(notes);
    expect(xml).toContain('w:val="Heading1"');
    expect(xml).toContain('w:val="Heading2"');
    expect(xml).toContain('w:val="Heading3"');
    expect(xml).toMatch(/<w:numPr>/); // bullets + numbered list use numbering
    expect(texts(xml)).toEqual(expect.arrayContaining(["first", "second", "Definition", " discussed"]));
  });

  it("preserves Unicode text", async () => {
    const xml = await docxXml("# नोट्स\n- café — naïve → ✓ 漢字");
    expect(texts(xml).join("")).toContain("नोट्स");
    expect(texts(xml).join("")).toContain("café — naïve → ✓ 漢字");
  });

  it("handles empty and very long content", async () => {
    expect(texts(await docxXml(""))).toEqual([]);
    const long = Array.from({ length: 400 }, (_, i) => `- point ${i}`).join("\n");
    expect(texts(await docxXml(long))).toHaveLength(400);
  });

  it("concise/detailed/academic outputs all export", async () => {
    for (const md of [
      "# Session\n- **Questions:** Question\n- **Detected Signs:** Whistle",
      "# Session\n\n## Key Concepts\n\nConcepts recognized during this session:\n- Definition\n\n---\n_disclaimer_",
      "# Lecture Notes\n\n*Session summary: 2 recognized sign(s).*\n\n## Core Concepts\n- Definition",
    ]) {
      expect((await docxXml(md)).length).toBeGreaterThan(100);
    }
  });
});

describe("shared Markdown parsing (display == export)", () => {
  it("parses heading levels, nested bullets, numbered items and rules", () => {
    const lines = parseNoteLines("# A\n#### B\n- x\n  - y\n3. z\n---\ntext", false);
    expect(lines.map((l) => l.kind)).toEqual(["heading", "heading", "bullet", "bullet", "numbered", "rule", "para"]);
    expect(lines[1]).toMatchObject({ level: 4 });
    expect(lines[3]).toMatchObject({ indent: 1 });
    expect(lines[4]).toMatchObject({ number: "3", text: "z" });
  });

  it("renders lists inside ul/ol and escapes HTML", () => {
    const html = renderMarkdown("- a\n- b\n1. c\n<script>x</script>");
    expect(html).toMatch(/<ul[^>]*><li>a<\/li><li>b<\/li><\/ul><ol[^>]*><li>c<\/li><\/ol>/);
    expect(html).not.toContain("<script>");
  });

  it("plain text renders numbered items and keeps text", () => {
    const txt = markdownToPlainText("# T\n1. one\n- two");
    expect(txt).toContain("1. one");
    expect(txt).toContain("• two");
  });

  it("flags characters the PDF font cannot draw instead of exporting garbage", () => {
    expect(unsupportedPdfCharacters("café — ok → done")).toEqual([]);
    expect(unsupportedPdfCharacters("नमस्ते").length).toBeGreaterThan(0);
  });

  it("builds dated, safe filenames", () => {
    expect(exportFilename("Live Session Notes", "docx", new Date(2026, 10, 6))).toBe("live-session-notes-2026-11-06.docx");
  });
});

describe("live pipeline fixes", () => {
  it("a repeated sign after a pause is committed again (regression)", () => {
    const s = new SessionSmoother({ stableCount: 2, repeatGapTicks: 3 });
    s.update("HELLO", 0.9, 0);
    expect(s.update("HELLO", 0.9, 1).status).toBe("committed");
    for (let t = 2; t < 5; t++) expect(s.update("", 0, t).status).toBe("no_sign");
    s.update("HELLO", 0.9, 5);
    expect(s.update("HELLO", 0.9, 6).status).toBe("committed");
  });

  it("a held sign with only a short dip is still ONE event", () => {
    const s = new SessionSmoother({ stableCount: 2, repeatGapTicks: 3 });
    s.update("HELLO", 0.9, 0);
    expect(s.update("HELLO", 0.9, 1).status).toBe("committed");
    s.update("", 0, 2); // one no-sign tick < repeatGapTicks
    s.update("HELLO", 0.9, 3);
    expect(s.update("HELLO", 0.9, 4).status).toBe("pending");
  });

  it("KeypointBuffer keeps windowFrames real frames padded to maxLen, and rejects wrong dims", () => {
    const buf = new KeypointBuffer(8, 2, 3);
    for (let i = 0; i < 5; i++) buf.push(new Float32Array([i, i]));
    expect(buf.length).toBe(3);
    expect(buf.rawFrames().map((f) => f[0])).toEqual([2, 3, 4]);
    expect(buf.getNormalizedWindow()!.length).toBe(16);
    expect(() => buf.push(new Float32Array(3))).toThrow();
  });
});

describe("legacy schema (v1) conversion mirrors feature_schema.py", () => {
  const hand = (x: number) => Array.from({ length: 21 }, (_, i) => ({ x: x + i * 0.01, y: 0.5, z: 0 }));

  it("dims per schema", () => {
    expect(featureDimForSchema("2.0")).toBe(FEATURE_DIM);
    expect(featureDimForSchema("1.0")).toBe(LEGACY_FEATURE_DIM);
    expect(featureDimForSchema("9")).toBeNull();
  });

  it("single right hand lands in v1 slot 0; both hands keep (left, right) order", () => {
    const right = legacyV1FromV2(buildFeatureVector(null, hand(0.3), null, null));
    expect(right.length).toBe(126);
    expect(right[0]).toBeCloseTo(0.3);
    expect(right.slice(63).every((v) => v === 0)).toBe(true);
    const both = toSchema(buildFeatureVector(hand(0.1), hand(0.6), null, null), "1.0");
    expect(both[0]).toBeCloseTo(0.1);
    expect(both[63]).toBeCloseTo(0.6);
    expect(() => toSchema(new Float32Array(FEATURE_DIM), "7.0")).toThrow();
  });
});

describe("server address, token, timeouts", () => {
  beforeEach(() => {
    localStorage.clear();
    setServerUrl("");
    setApiToken("");
  });
  afterEach(() => vi.restoreAllMocks());

  it("defaults to this page's host on port 3001 (regression: was hard-coded localhost)", () => {
    expect(defaultServerUrl()).toBe(`http://${window.location.hostname}:3001`);
    expect(getServerUrl()).toBe(defaultServerUrl());
  });

  it("stores a custom URL and rejects non-http URLs", () => {
    setServerUrl("http://192.168.1.20:3001/");
    expect(getServerUrl()).toBe("http://192.168.1.20:3001");
    expect(() => setServerUrl("ftp://x")).toThrow();
    expect(() => setServerUrl("not a url")).toThrow();
  });

  it("sends the token and retries a bounded number of times on 503", async () => {
    setApiToken("abc");
    const calls: RequestInit[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_u, init) => {
      calls.push(init as RequestInit);
      return new Response("{}", { status: 503 });
    });
    const res = await apiFetch("/model/meta", {}, { retries: 2, retryDelayMs: 1 });
    expect(res.status).toBe(503);
    expect(calls).toHaveLength(3);
    expect(new Headers(calls[0].headers).get("Authorization")).toBe("Bearer abc");
  });

  it("times out with a clear message instead of hanging", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (_u, init) =>
        new Promise((_res, rej) => {
          (init as RequestInit).signal?.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError")));
        })
    );
    await expect(apiFetch("/health", {}, { timeoutMs: 20 })).rejects.toBeInstanceOf(ApiError);
    await expect(apiFetch("/health", {}, { timeoutMs: 20 })).rejects.toThrow(/did not respond/);
  });
});
