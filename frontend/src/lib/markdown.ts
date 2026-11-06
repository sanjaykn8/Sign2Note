/**
 * Small Markdown-subset -> HTML renderer for generated notes: headings
 * (#..######), bold/italic, bullet and numbered lists (one nesting level per
 * two spaces), horizontal rules and paragraphs. It uses the SAME line parser
 * as the exporters (notesExport.parseNoteLines), so what is displayed is
 * what gets exported. Not a general Markdown parser. All text is
 * HTML-escaped before any markup is added, so generated notes can't inject
 * HTML.
 */
import { parseNoteLines } from "./notesExport";

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function inline(text: string): string {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/(^|\W)_(.+?)_(?=\W|$)/g, "$1<em>$2</em>");
}

const HEADING_CLASS: Record<number, string> = {
  1: "text-xl font-bold mt-6 mb-2",
  2: "text-lg font-bold mt-5 mb-2",
  3: "text-base font-semibold mt-4 mb-1",
};

export function renderMarkdown(md: string): string {
  const out: string[] = [];
  let openList: "ul" | "ol" | null = null;
  const closeList = () => {
    if (openList) out.push(`</${openList}>`);
    openList = null;
  };
  for (const line of parseNoteLines(md, false)) {
    if (line.kind === "bullet" || line.kind === "numbered") {
      const tag = line.kind === "bullet" ? "ul" : "ol";
      if (openList !== tag) {
        closeList();
        out.push(tag === "ul" ? "<ul class='list-disc pl-5 my-1'>" : "<ol class='list-decimal pl-5 my-1'>");
        openList = tag;
      }
      const style = line.indent ? ` style='margin-left:${line.indent * 1.25}rem'` : "";
      out.push(`<li${style}>${inline(line.text)}</li>`);
      continue;
    }
    closeList();
    if (line.kind === "heading") {
      const level = Math.min(line.level, 6);
      out.push(`<h${level} class='${HEADING_CLASS[level] ?? "text-sm font-semibold mt-3 mb-1"}'>${inline(line.text)}</h${level}>`);
    } else if (line.kind === "rule") {
      out.push("<hr class='my-3 border-border'/>");
    } else if (line.kind === "para") {
      out.push(`<p class='my-1'>${inline(line.text)}</p>`);
    }
  }
  closeList();
  return out.join("");
}
