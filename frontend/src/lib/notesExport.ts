/**
 * Turns generated Markdown notes into downloadable files -- Markdown, a
 * styled plain-text version, and PDF -- entirely client-side. No backend
 * round-trip: by the time notes are on screen the full text already lives
 * in the browser, so there's nothing to fetch.
 *
 * The plain-text/PDF layout mirrors the "Suggested generated document"
 * format from the workplan (SIGN2NOTES header, date/duration, uppercase
 * section headers, bullet points, footer disclaimer).
 */

export interface ExportMeta {
  /** Lecture / session title, e.g. "DBMS Lecture 04". Defaults to "Lecture Notes". */
  title?: string;
  date?: Date;
  durationSeconds?: number;
}

function formatDuration(seconds?: number): string | null {
  if (seconds === undefined || seconds === null || !Number.isFinite(seconds)) return null;
  const total = Math.max(0, Math.round(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function formatDate(date?: Date): string {
  return (date ?? new Date()).toLocaleDateString(undefined, {
    day: "2-digit",
    month: "long",
    year: "numeric",
  });
}

/** One structured line of the notes, used by both the plain-text formatter
 * and the PDF renderer so the two stay visually consistent. */
type NoteLine =
  | { kind: "heading"; text: string; level: number }
  | { kind: "bullet"; text: string; indent: number }
  | { kind: "numbered"; text: string; number: string; indent: number }
  | { kind: "para"; text: string }
  | { kind: "rule" }
  | { kind: "blank" };

/** Line-level Markdown parse shared by every exporter (and mirrored by
 * markdown.ts's on-screen renderer) so exports match what is displayed. */
export function parseNoteLines(markdown: string, stripEmphasis = true): NoteLine[] {
  const lines: NoteLine[] = [];
  const strip = (s: string) =>
    stripEmphasis ? s.replace(/\*\*(.+?)\*\*/g, "$1").replace(/\*(.+?)\*/g, "$1").replace(/_(.+?)_/g, "$1") : s;
  for (const raw of markdown.replace(/\r\n?/g, "\n").split("\n")) {
    const line = raw.trimEnd();
    const indent = Math.floor((line.length - line.trimStart().length) / 2);
    const t = line.trimStart();
    let m: RegExpMatchArray | null;
    if ((m = t.match(/^(#{1,6})\s+(.*)$/))) {
      lines.push({ kind: "heading", level: m[1].length, text: strip(m[2]) });
    } else if (/^(-{3,}|\*{3,}|_{3,})$/.test(t)) {
      lines.push({ kind: "rule" });
    } else if ((m = t.match(/^[-*+]\s+(.*)$/))) {
      lines.push({ kind: "bullet", indent, text: strip(m[1]) });
    } else if ((m = t.match(/^(\d+)[.)]\s+(.*)$/))) {
      lines.push({ kind: "numbered", indent, number: m[1], text: strip(m[2]) });
    } else if (t.length === 0) {
      lines.push({ kind: "blank" });
    } else {
      lines.push({ kind: "para", text: strip(t) });
    }
  }
  return lines;
}

/** Splits a line of text into runs of plain/bold/italic segments, for
 * renderers (DOCX) that can represent emphasis natively instead of
 * stripping the `**bold**`/`*italic*` markers like the plain-text/PDF
 * renderers do. Doesn't handle nested/overlapping emphasis -- the notes
 * this renders are LLM- or template-generated Markdown, not arbitrary
 * user Markdown, so that's not a realistic input. Exported for direct
 * unit testing. */
export function parseInlineRuns(text: string): { text: string; bold?: boolean; italics?: boolean }[] {
  const runs: { text: string; bold?: boolean; italics?: boolean }[] = [];
  const pattern = /\*\*(.+?)\*\*|\*(.+?)\*/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) runs.push({ text: text.slice(lastIndex, match.index) });
    if (match[1] !== undefined) runs.push({ text: match[1], bold: true });
    else if (match[2] !== undefined) runs.push({ text: match[2], italics: true });
    lastIndex = pattern.lastIndex;
  }
  if (lastIndex < text.length) runs.push({ text: text.slice(lastIndex) });
  return runs.length > 0 ? runs : [{ text }];
}

/** Renders the SIGN2NOTES-styled plain-text document (see workplan section 7). */
export function markdownToPlainText(markdown: string, meta: ExportMeta = {}): string {
  const divider = "\u2500".repeat(40);
  const duration = formatDuration(meta.durationSeconds);

  const out: string[] = [
    "SIGN2NOTES",
    meta.title || "Lecture Notes",
    divider,
    `Date: ${formatDate(meta.date)}`,
  ];
  if (duration) out.push(`Duration: ${duration}`);
  out.push("");

  for (const line of parseNoteLines(markdown)) {
    if (line.kind === "heading") out.push("", line.text.toUpperCase());
    else if (line.kind === "bullet") out.push(`${"  ".repeat(line.indent)}\u2022 ${line.text}`);
    else if (line.kind === "numbered") out.push(`${"  ".repeat(line.indent)}${line.number}. ${line.text}`);
    else if (line.kind === "rule") out.push("\u2500".repeat(20));
    else if (line.kind === "blank") out.push("");
    else out.push(line.text);
  }

  out.push("", divider, "Generated by Sign2Notes", "Prototype \u2014 limited vocabulary");
  return out.join("\n");
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoking synchronously can cancel the download in Firefox/Safari.
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

export function downloadTextFile(filename: string, content: string, mimeType = "text/plain") {
  // UTF-8 BOM so Windows editors (Notepad) detect non-ASCII text correctly.
  triggerDownload(new Blob(["\ufeff", content], { type: `${mimeType};charset=utf-8` }), filename);
}

export function downloadMarkdown(filename: string, markdown: string) {
  downloadTextFile(filename, markdown, "text/markdown");
}

export function downloadPlainText(filename: string, markdown: string, meta: ExportMeta = {}) {
  downloadTextFile(filename, markdownToPlainText(markdown, meta), "text/plain");
}

/** Renders notes to a simple PDF via jsPDF, dynamically imported so the
 * library only loads for users who actually export to PDF. */
/** Characters jsPDF's built-in Helvetica can draw (WinAnsi / Latin-1 plus
 * a few typographic marks). Anything else would come out as garbage. */
export function unsupportedPdfCharacters(text: string): string[] {
  const bad = new Set<string>();
  for (const ch of text) {
    const c = ch.codePointAt(0)!;
    const ok = c === 10 || c === 13 || c === 9 || (c >= 0x20 && c <= 0x7e) || (c >= 0xa0 && c <= 0xff) ||
      "\u2013\u2014\u2018\u2019\u201c\u201d\u2022\u2026\u2192\u2500\u20ac".includes(ch);
    if (!ok) bad.add(ch);
  }
  return [...bad];
}

export async function downloadPdf(filename: string, markdown: string, meta: ExportMeta = {}) {
  const unsupported = unsupportedPdfCharacters(`${markdown}${meta.title ?? ""}`);
  if (unsupported.length > 0) {
    throw new Error(
      `PDF export uses a Latin-only font and can't show: ${unsupported.slice(0, 8).join(" ")}. Use Word (.docx), Markdown or Text for this content.`
    );
  }
  const { jsPDF } = await import("jspdf");
  const doc = new jsPDF({ unit: "pt", format: "a4" });

  const marginX = 48;
  const pageHeight = doc.internal.pageSize.getHeight();
  const pageWidth = doc.internal.pageSize.getWidth();
  const maxWidth = pageWidth - marginX * 2;
  let y = 56;

  const ensureSpace = (need: number) => {
    if (y + need > pageHeight - 48) {
      doc.addPage();
      y = 56;
    }
  };

  doc.setFont("helvetica", "bold");
  doc.setFontSize(20);
  doc.text("Sign2Notes", marginX, y);
  y += 26;

  doc.setFontSize(14);
  doc.text(meta.title || "Lecture Notes", marginX, y);
  y += 20;

  const duration = formatDuration(meta.durationSeconds);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(110);
  doc.text(`Date: ${formatDate(meta.date)}${duration ? `    Duration: ${duration}` : ""}`, marginX, y);
  doc.setTextColor(20);
  y += 20;
  doc.setDrawColor(210);
  doc.line(marginX, y, pageWidth - marginX, y);
  y += 18;

  for (const line of parseNoteLines(markdown)) {
    if (line.kind === "blank") {
      y += 8;
      continue;
    }
    if (line.kind === "rule") {
      ensureSpace(12);
      doc.setDrawColor(210);
      doc.line(marginX, y, pageWidth - marginX, y);
      y += 10;
      continue;
    }
    if (line.kind === "heading") {
      ensureSpace(24);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(12.5);
      y += 8;
      doc.text(line.text.toUpperCase(), marginX, y);
      y += 16;
      continue;
    }
    const isBullet = line.kind === "bullet";
    const isNumbered = line.kind === "numbered";
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10.5);
    const prefix = isBullet ? "\u2022  " : isNumbered ? `${line.number}.  ` : "";
    const indent = isBullet || isNumbered ? 12 + 12 * line.indent : 0;
    const wrapped = doc.splitTextToSize(prefix + line.text, maxWidth - indent);
    for (const w of wrapped) {
      ensureSpace(14);
      doc.text(w, marginX + indent, y);
      y += 14;
    }
  }

  ensureSpace(30);
  y += 10;
  doc.setDrawColor(210);
  doc.line(marginX, y, pageWidth - marginX, y);
  y += 16;
  doc.setFontSize(8.5);
  doc.setTextColor(140);
  doc.text("Generated by Sign2Notes \u2014 Prototype, limited vocabulary", marginX, y);

  doc.save(filename);
}

/** Slugifies a title into a safe filename stem, e.g. "DBMS Lecture 04" -> "dbms-lecture-04". */
export function slugify(title: string, fallback = "sign2notes"): string {
  const slug = title
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
  return slug || fallback;
}

/** Builds the `docx` Document object for a set of notes -- separated out
 * from downloadDocx() so tests can construct it and inspect the output.
 *
 * The document body is EXACTLY the notes as shown in the app (headings,
 * paragraphs, bullet and numbered lists, bold/italic), with nothing added:
 * no cover page, no logo/author, and no extra title or date block (the
 * notes already start with their own "# Title"; the previous version
 * prepended a second title plus a date line that the app never showed).
 * `meta.title` only sets the file's document-properties title.
 */
export async function buildDocxDocument(markdown: string, meta: ExportMeta = {}) {
  const { Document, Paragraph, TextRun, HeadingLevel, LevelFormat, AlignmentType } = await import("docx");

  const headingFor = [
    HeadingLevel.HEADING_1,
    HeadingLevel.HEADING_1,
    HeadingLevel.HEADING_2,
    HeadingLevel.HEADING_3,
    HeadingLevel.HEADING_4,
    HeadingLevel.HEADING_5,
    HeadingLevel.HEADING_6,
  ];
  const runs = (text: string) => parseInlineRuns(text).map((r) => new TextRun(r));
  const children: InstanceType<typeof Paragraph>[] = [];
  let numberedInstance = 0;
  let inNumbered = false;

  for (const line of parseNoteLines(markdown, /* stripEmphasis */ false)) {
    if (line.kind !== "numbered") inNumbered = false;
    if (line.kind === "blank") continue; // paragraph spacing comes from styles
    if (line.kind === "rule") {
      children.push(new Paragraph({ border: { bottom: { style: "single", size: 6, color: "BBBBBB", space: 1 } } }));
    } else if (line.kind === "heading") {
      children.push(new Paragraph({ children: runs(line.text), heading: headingFor[line.level] }));
    } else if (line.kind === "bullet") {
      children.push(new Paragraph({ children: runs(line.text), bullet: { level: Math.min(line.indent, 8) } }));
    } else if (line.kind === "numbered") {
      if (!inNumbered) {
        numberedInstance += 1; // restart numbering for each separate list
        inNumbered = true;
      }
      children.push(
        new Paragraph({
          children: runs(line.text),
          numbering: { reference: "notes-numbering", level: Math.min(line.indent, 8), instance: numberedInstance },
        })
      );
    } else {
      children.push(new Paragraph({ children: runs(line.text) }));
    }
  }
  if (children.length === 0) children.push(new Paragraph({ text: "" }));

  return new Document({
    title: meta.title || "Sign2Notes notes",
    creator: "Sign2Notes",
    numbering: {
      config: [
        {
          reference: "notes-numbering",
          levels: Array.from({ length: 9 }, (_, level) => ({
            level,
            format: LevelFormat.DECIMAL,
            text: `%${level + 1}.`,
            alignment: AlignmentType.START,
            style: { paragraph: { indent: { left: 720 * (level + 1), hanging: 360 } } },
          })),
        },
      ],
    },
    sections: [{ children }],
  });
}

/** Renders notes to a simple .docx file via the `docx` library, entirely
 * client-side (dynamically imported -- only loads for users who export). */
export async function downloadDocx(filename: string, markdown: string, meta: ExportMeta = {}) {
  const { Packer } = await import("docx");
  const doc = await buildDocxDocument(markdown, meta);
  triggerDownload(await Packer.toBlob(doc), filename);
}

/** Filename for an export: "<slug>-YYYY-MM-DD.<ext>". */
export function exportFilename(title: string, ext: string, date = new Date()): string {
  const d = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
  return `${slugify(title)}-${d}.${ext}`;
}
