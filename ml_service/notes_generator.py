from typing import List


def template_notes_from_tokens(tokens: List[str], title: str = "Generated Notes") -> str:
    """Deterministic, LLM-free notes generator. Always available — no external
    service required, so this is the safe fallback for every failure mode."""
    if not tokens:
        return "# Generated Notes\n\nNo confident signs were detected. Please repeat the gesture."
    lines = [f"# {title}", "", "## Detected Content", ""]
    for token in tokens:
        lines.append(f"- **{token}**")
    lines += ["", "## Review", "", "These notes were generated from recognized sign glosses. Verify low-confidence or ambiguous content before use."]
    return "\n".join(lines)


def build_notes_prompt(tokens: List[str], style: str = "concise") -> str:
    """Shared prompt builder used by the llama.cpp-backed notes generator in infer.py."""
    style_text = {
        "concise": "Use concise headings and bullet points.",
        "detailed": "Use slightly expanded explanations, but do not invent unsupported facts.",
        "academic": "Use a formal academic structure with clear headings and concise bullets.",
    }.get(style, "Use concise headings and bullet points.")
    return f"""You are Sign2Notes, a classroom note-taking assistant.
Convert ONLY the following ASL gloss sequence into readable Markdown notes.
{style_text}
Rules:
1. Preserve the meaning of the glosses.
2. Do not invent lecture facts, examples, dates, formulas, or names that are not supported by the input.
3. Group repeated/related glosses when sensible.
4. Output Markdown only.

Gloss sequence:
{', '.join(tokens)}
"""
