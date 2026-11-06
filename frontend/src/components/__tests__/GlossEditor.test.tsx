import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { useState } from "react";
import GlossEditor, { type GlossItem } from "../GlossEditor";
import ResultsPanel from "../ResultsPanel";
import type { ProcessResult } from "@/lib/api";

function Harness({ initial }: { initial: GlossItem[] }) {
  const [items, setItems] = useState(initial);
  return (
    <>
      <GlossEditor items={items} onChange={setItems} />
      <output data-testid="labels">{items.map((g) => g.label).join(",")}</output>
    </>
  );
}

describe("GlossEditor", () => {
  it("edits, deletes and undoes in order", () => {
    render(<Harness initial={[{ label: "Hello", confidence: 0.9 }, { label: "Book", confidence: 0.4 }]} />);
    const labels = () => screen.getByTestId("labels").textContent;
    expect(screen.getByText("(?)")).toBeInTheDocument(); // low-confidence flag

    fireEvent.click(screen.getByRole("button", { name: "Edit gloss Book" }));
    const input = screen.getByRole("textbox", { name: "Edit gloss 2" });
    fireEvent.change(input, { target: { value: "Pen" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(labels()).toBe("Hello,Pen");

    fireEvent.click(screen.getByRole("button", { name: "Delete gloss Hello" }));
    expect(labels()).toBe("Pen");

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(labels()).toBe("Hello,Pen");
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(labels()).toBe("Hello,Book");
  });
});

const generateNotesFromGlosses = vi.fn();
vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return { ...actual, generateNotesFromGlosses: (...a: unknown[]) => generateNotesFromGlosses(...a) };
});

describe("ResultsPanel (Video Upload results)", () => {
  const result: ProcessResult = {
    notes_md: "# Lecture Notes\n- Hello",
    gloss_list: ["Hello", "Book"],
    events: [
      { label: "Hello", confidence: 0.9, start_time: 0, end_time: 1.2 },
      { label: "Book", confidence: 0.8, start_time: 1.5, end_time: 3 },
    ],
    segments: [],
    confidence: 0.85,
    backend: "onnx",
    generation: { generator: "template", fallback_reason: null, style: "concise" },
  };

  it("regenerates notes from the EDITED glosses with the chosen style", async () => {
    generateNotesFromGlosses.mockResolvedValue({ notes_md: "# Academic\n- Book", generation: { generator: "template", fallback_reason: null } });
    render(<ResultsPanel result={result} />);
    expect(screen.getByText("Deterministic notes (no LLM)")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete gloss Hello" }));
    fireEvent.change(screen.getByLabelText("Notes style"), { target: { value: "academic" } });
    fireEvent.click(screen.getByRole("button", { name: /regenerate notes/i }));

    await waitFor(() => expect(screen.getByText("Academic")).toBeInTheDocument());
    expect(generateNotesFromGlosses).toHaveBeenCalledWith(["Book"], expect.objectContaining({ style: "academic", confidences: [0.8] }));
  });

  it("labels a deterministic fallback when the LLM was unavailable", () => {
    render(<ResultsPanel result={{ ...result, generation: { generator: "template", fallback_reason: "APIConnectionError" } }} />);
    expect(screen.getByText(/LLM unavailable — deterministic fallback used/)).toBeInTheDocument();
  });
});
