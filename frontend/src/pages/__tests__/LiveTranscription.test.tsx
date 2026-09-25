import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import LiveTranscription from "../LiveTranscription";

// A minimal fake of useSignRecognitionSession, driven by a shared mutable
// object the tests set directly rather than real React state -- the hook's
// OWN internals (camera lifecycle, detection loop, smoothing) are already
// covered by webcamPipeline.test.ts / featureSchema.test.ts; what this
// file needs to verify is how the PAGE reacts to session state and, most
// importantly, that it doesn't fire duplicate generation requests.
const mockState = vi.hoisted(() => ({
  sessionActive: true,
  history: [] as { label: string; confidence: number; timestamp: number }[],
}));

vi.mock("@/lib/useSignRecognitionSession", () => ({
  useSignRecognitionSession: () => ({
    videoRef: { current: null },
    canvasRef: { current: null },
    cameraState: "active",
    cameraError: null,
    modelState: "ready",
    modelError: null,
    sessionActive: mockState.sessionActive,
    paused: false,
    current: null,
    history: mockState.history,
    editingIndex: null,
    editDraft: "",
    setEditDraft: () => {},
    source: "webcam",
    screenCaptureSupported: true,
    startSession: vi.fn(async () => {
      mockState.sessionActive = true;
    }),
    stopSession: vi.fn(() => {
      mockState.sessionActive = false;
    }),
    pauseSession: vi.fn(),
    resumeSession: vi.fn(),
    clearSession: vi.fn(() => {
      mockState.history = [];
    }),
    undoLast: vi.fn(),
    deleteEvent: vi.fn(),
    startEdit: vi.fn(),
    saveEdit: vi.fn(),
    cancelEdit: vi.fn(),
    elapsedSessionSeconds: () => 12,
  }),
}));

const generateTranscriptFromGlosses = vi.fn();
const generateNotesFromGlosses = vi.fn();
vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    generateTranscriptFromGlosses: (...args: unknown[]) => generateTranscriptFromGlosses(...args),
    generateNotesFromGlosses: (...args: unknown[]) => generateNotesFromGlosses(...args),
  };
});

function seedHistory(labels: string[]) {
  mockState.history = labels.map((label, i) => ({ label, confidence: 0.9, timestamp: i }));
}

beforeEach(() => {
  mockState.sessionActive = true;
  mockState.history = [];
  generateTranscriptFromGlosses.mockReset();
  generateNotesFromGlosses.mockReset();
});

describe("LiveTranscription -- Stop & Generate", () => {
  it("calls the transcript and notes generators exactly once each, even on a rapid double-click", async () => {
    seedHistory(["QUESTION", "IMPORTANT", "EXAM"]);
    generateTranscriptFromGlosses.mockResolvedValue({ transcript: "A question was raised about an important exam." });
    generateNotesFromGlosses.mockResolvedValue({ notes_md: "# Notes\n- Exam mentioned" });

    render(<LiveTranscription />);
    const button = screen.getByRole("button", { name: /stop & generate/i });

    // Simulate a rapid double-click -- the second click should be a no-op
    // while the first is still resolving (project brief section 56: "Do
    // not allow duplicate requests if the user clicks Stop/Generate
    // rapidly").
    fireEvent.click(button);
    fireEvent.click(button);

    await waitFor(() => {
      expect(screen.getByText(/A question was raised/i)).toBeInTheDocument();
    });

    expect(generateTranscriptFromGlosses).toHaveBeenCalledTimes(1);
    expect(generateNotesFromGlosses).toHaveBeenCalledTimes(1);
    expect(generateTranscriptFromGlosses).toHaveBeenCalledWith(
      ["QUESTION", "IMPORTANT", "EXAM"],
      expect.objectContaining({ notesMode: "template" })
    );
  });

  it("shows both the natural-language transcript AND the notes panel after generation", async () => {
    seedHistory(["DEFINITION"]);
    generateTranscriptFromGlosses.mockResolvedValue({ transcript: "A definition was given." });
    generateNotesFromGlosses.mockResolvedValue({ notes_md: "# Lecture Notes\n- Definition discussed" });

    render(<LiveTranscription />);
    fireEvent.click(screen.getByRole("button", { name: /stop & generate/i }));

    await waitFor(() => expect(screen.getByText(/A definition was given/i)).toBeInTheDocument());
    expect(screen.getByText(/Live Transcription Notes/i)).toBeInTheDocument();
  });

  it("does not call either generator when the session has no recognized glosses", async () => {
    seedHistory([]);
    render(<LiveTranscription />);
    fireEvent.click(screen.getByRole("button", { name: /stop & generate/i }));

    await waitFor(() => {
      expect(screen.getByText(/nothing to generate/i)).toBeInTheDocument();
    });
    expect(generateTranscriptFromGlosses).not.toHaveBeenCalled();
    expect(generateNotesFromGlosses).not.toHaveBeenCalled();
  });

  it("shows an error with a Retry option when both generators fail, and Retry re-attempts using the preserved glosses", async () => {
    seedHistory(["QUESTION"]);
    generateTranscriptFromGlosses.mockRejectedValueOnce(new Error("network down"));
    generateNotesFromGlosses.mockRejectedValueOnce(new Error("network down"));

    render(<LiveTranscription />);
    fireEvent.click(screen.getByRole("button", { name: /stop & generate/i }));

    await waitFor(() => {
      expect(screen.getByText(/preserved.*retry/i)).toBeInTheDocument();
    });
    expect(generateTranscriptFromGlosses).toHaveBeenCalledTimes(1);

    // Now let the retry succeed.
    generateTranscriptFromGlosses.mockResolvedValueOnce({ transcript: "Recovered transcript." });
    generateNotesFromGlosses.mockResolvedValueOnce({ notes_md: "# Notes" });

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    await waitFor(() => expect(screen.getByText(/Recovered transcript/i)).toBeInTheDocument());
    // Retried with the SAME frozen glosses, not an empty/different list.
    expect(generateTranscriptFromGlosses).toHaveBeenLastCalledWith(
      ["QUESTION"],
      expect.objectContaining({ notesMode: "template" })
    );
  });

  it("partial success (one generator fails) still shows the successful output, no error banner", async () => {
    seedHistory(["QUESTION"]);
    generateTranscriptFromGlosses.mockResolvedValue({ transcript: "It worked." });
    generateNotesFromGlosses.mockRejectedValue(new Error("LLM down"));

    render(<LiveTranscription />);
    fireEvent.click(screen.getByRole("button", { name: /stop & generate/i }));

    await waitFor(() => expect(screen.getByText(/It worked\./i)).toBeInTheDocument());
    expect(screen.queryByText(/preserved.*retry/i)).not.toBeInTheDocument();
  });
});
