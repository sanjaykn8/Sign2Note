"""Deterministic (LLM-free) and LLM-prompt note generation from gloss sequences.

template_notes_from_tokens() is the fallback mode: it must always produce
usable notes with zero external dependencies, because it's what the app
falls back to whenever the LLM is unavailable (Test 3 in the acceptance
tests). It opportunistically groups glosses into lecture-style sections
(Key Concepts / Questions / Tasks / Important) when it recognizes
structuring keywords, and otherwise falls back to a flat "Detected Signs"
list -- which is what actually happens with the current FDMSE-ISL 400-word
vocabulary, since it's everyday ISL words, not lecture-structure signs like
"QUESTION" or "HOMEWORK". Both paths are demonstrated in TESTING/tests.
"""

from typing import List

# Keyword -> section mapping for the deterministic template. Matching is
# case-insensitive substring matching against each gloss, so e.g. a gloss
# literally named "Question" or "Homework" (as in the walkthrough examples
# in the spec) is grouped, while arbitrary vocabulary (e.g. FDMSE-ISL's
# "Whistle", "Radish") falls through to the generic bucket untouched.
_SECTION_KEYWORDS = {
    "Key Concepts": ["definition", "example", "concept", "important", "key"],
    "Questions": ["question", "ask", "doubt"],
    "Tasks": ["homework", "assignment", "task", "deadline"],
}
_SECTION_ORDER = ["Key Concepts", "Questions", "Tasks", "Detected Signs"]

_SECTION_INTRO = {
    "Key Concepts": "Concepts discussed:",
    "Questions": "Questions raised during the session:",
    "Tasks": "Tasks or assignments mentioned:",
    "Detected Signs": "Signs recognized during this session:",
}


def _categorize(tokens: List[str]) -> dict:
    buckets = {name: [] for name in _SECTION_ORDER}
    for token in tokens:
        lowered = token.lower()
        placed = False
        for section, keywords in _SECTION_KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                buckets[section].append(token)
                placed = True
                break
        if not placed:
            buckets["Detected Signs"].append(token)
    return buckets


def template_notes_from_tokens(tokens: List[str], title: str = "Lecture Notes") -> str:
    """Deterministic, LLM-free notes generator. Always available -- no
    external service, no network call -- so this is the safe fallback for
    every failure mode (LLM down, model missing, network error, etc)."""
    if not tokens:
        return f"# {title}\n\nNo confident signs were detected. Please repeat the gesture."

    buckets = _categorize(tokens)
    lines = [f"# {title}", ""]
    any_section = False
    for section in _SECTION_ORDER:
        items = buckets[section]
        if not items:
            continue
        any_section = True
        lines.append(f"## {section}")
        lines.append("")
        lines.append(_SECTION_INTRO[section])
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    if not any_section:
        # Shouldn't happen (Detected Signs always catches leftovers), but
        # keep a hard fallback so this function can never return an empty body.
        lines.append("## Detected Signs")
        lines.append("")
        for item in tokens:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("---")
    lines.append(
        "_Notes generated from recognized sign-language glosses using a "
        "constrained-vocabulary recognizer. Verify low-confidence or "
        "ambiguous content before relying on it._"
    )
    return "\n".join(lines)


def build_notes_prompt(tokens: List[str], style: str = "concise") -> str:
    """Prompt for the LLM-backed notes generator in infer.py. Follows the
    rules from the spec: treat glosses as imperfect recognition output,
    never invent facts, preserve meaning, group related concepts, surface
    questions/assignments, and return Markdown only."""
    style_text = {
        "concise": "Use concise headings and bullet points.",
        "detailed": "Use slightly expanded explanations, but do not invent unsupported facts.",
        "academic": "Use a formal academic structure with clear headings and concise bullets.",
    }.get(style, "Use concise headings and bullet points.")
    return f"""You are a lecture note assistant for Sign2Notes.
Convert the following recognized sign-language gloss sequence into clear,
concise lecture notes. Treat the gloss sequence as imperfect recognition
output from a constrained-vocabulary sign recognizer, not as a transcript
of everything that was said.

Glosses (in recognized order):
{', '.join(tokens)}

Style: {style_text}

Rules:
1. Preserve the meaning of the provided glosses. Do not invent facts,
   examples, dates, names, or content that is not supported by the input.
2. Group related concepts together rather than listing every gloss flat.
3. Highlight questions and assignments/tasks in their own sections if any
   glosses suggest them.
4. Do not mention internal model details, confidence scores, or that this
   came from an AI/recognizer -- write as if these are lecture notes.
5. Return Markdown only -- no commentary before or after the notes.
"""
