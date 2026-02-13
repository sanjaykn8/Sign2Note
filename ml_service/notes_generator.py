# ml_service/notes_generator.py
import json
import subprocess
from pathlib import Path
from typing import List, Optional

def template_notes_from_tokens(tokens: List[str], title: Optional[str]=None) -> str:
    """
    Simple deterministic notes generator for demo.
    """
    if title is None:
        title = "Generated Notes"
    md = []
    md.append(f"# {title}\n")
    md.append("**Summary:**\n")
    md.append("This document summarizes the content detected from the sign language video. Below are expanded sections and key points.\n\n")
    # group tokens into pseudo sections of 6 tokens
    chunk_size = 6
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size]
        heading = chunk[0] if chunk else "Section"
        md.append(f"## **{heading}**\n")
        for t in chunk:
            md.append(f"- *{t}* — Expanded explanation about **{t}**. Add more detail here based on context.\n")
        md.append("\n")
    md.append("**Conclusion:**\n")
    md.append("The above notes are auto-generated. Use them as a baseline and edit for accuracy.\n")
    return "".join(md)

def run_llama_cpp_prompt(gguf_path: str, prompt: str, llama_bin_path: str = "./llama.cpp/main", n_predict:int=512) -> str:
    """
    Optional: call llama.cpp `main` binary. Returns raw output text.
    User must build llama.cpp with CUDA or CPU and provide path to binary.
    """
    cmd = [llama_bin_path, "-m", gguf_path, "-p", prompt, "--n_predict", str(n_predict)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
        return out.stdout
    except Exception as e:
        return f"[llama.cpp call failed: {e}]"

def tokens_to_markdown(tokens_list, use_llama=False, gguf_path=None, llama_bin_path=None):
    """
    tokens_list: list of token strings
    If use_llama and gguf_path provided, will assemble a prompt and call llama.cpp; otherwise use template_notes_from_tokens.
    """
    if use_llama and gguf_path:
        token_text = " ".join(tokens_list)
        prompt = f"You are a note-taking assistant. Convert this token sequence into well-structured Markdown notes:\n\n{token_text}\n\nOutput only Markdown."
        out = run_llama_cpp_prompt(gguf_path, prompt, llama_bin_path or "./llama.cpp/main", n_predict=512)
        return out
    else:
        return template_notes_from_tokens(tokens_list)
