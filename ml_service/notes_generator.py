from typing import List

import requests


def template_notes_from_tokens(tokens: List[str], title: str = "Generated Notes") -> str:
    if not tokens:
        return "# Generated Notes\n\nNo confident signs were detected. Please repeat the gesture."
    lines = [f"# {title}", "", "## Detected Content", ""]
    for token in tokens:
        lines.append(f"- **{token}**")
    lines += ["", "## Review", "", "These notes were generated from recognized sign glosses. Verify low-confidence or ambiguous content before use."]
    return "\n".join(lines)


def _prompt(tokens: List[str], style: str) -> str:
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


def generate_ollama_notes(tokens: List[str], model="llama3.2:3b", style="concise") -> str:
    r = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": model, "prompt": _prompt(tokens, style), "stream": False,
              "options": {"temperature": 0.1}},
        timeout=90,
    )
    r.raise_for_status()
    text = r.json().get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response")
    return text


def run_llama_cpp_prompt(gguf_path: str, prompt: str,
                         llama_bin_path: str, n_predict: int = 512) -> str:
    import subprocess
    cmd = [llama_bin_path, "-m", gguf_path, "-p", prompt, "--n_predict", str(n_predict)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
    return out.stdout.strip()
