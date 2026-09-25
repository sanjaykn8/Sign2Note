"""generate_notes()'s LLM-failure fallback -- infer.py. No LLM server is
running in the test environment, so every "llm" mode call here must fall
back to template notes rather than raising."""
from infer import generate_notes, generate_transcript


def test_template_mode_never_touches_the_network():
    out = generate_notes(["DEFINITION", "QUESTION"], mode="template")
    assert "DEFINITION" in out


def test_llm_mode_falls_back_to_template_when_server_unreachable():
    out = generate_notes(["DEFINITION", "QUESTION"], mode="llm")
    # falls back to the same deterministic template output. Default style
    # is "concise", which is deliberately flat bullets (no "## " headers --
    # see notes_generator.py's _template_concise) rather than assuming any
    # particular Markdown structure, just that it's the template engine's
    # output and not raw/empty.
    assert "DEFINITION" in out
    assert "fallback" in out.lower()


def test_legacy_ollama_model_kwarg_still_works():
    # older callers used ollama_model=; generate_notes() must not crash
    # when called this way even though the parameter is now llm_model
    out = generate_notes(["DEFINITION"], mode="template", ollama_model="some-model")
    assert "DEFINITION" in out


def test_llm_fallback_honors_the_requested_style_not_always_concise():
    # Regression test: generate_notes()'s LLM-failure fallback used to call
    # template_notes_from_tokens() without passing style through, silently
    # ignoring whatever style the caller asked for.
    out = generate_notes(["DEFINITION", "QUESTION", "HOMEWORK"], mode="llm", style="academic")
    assert "## Core Concepts" in out  # academic-specific header, not concise's flat bullets


# ---------------------------------------------------------------------------
# generate_transcript() -- Layer 2 (natural-language transcript), separate
# from generate_notes() (Layer 3, structured notes)
# ---------------------------------------------------------------------------

def test_transcript_template_mode_never_touches_the_network():
    out = generate_transcript(["DEFINITION", "QUESTION"], mode="template")
    assert "DEFINITION" in out
    assert "QUESTION" in out


def test_transcript_llm_mode_falls_back_to_template_when_server_unreachable():
    out = generate_transcript(["DEFINITION", "QUESTION"], mode="llm")
    assert "DEFINITION" in out
    assert "no llm was used" in out.lower()


def test_transcript_output_is_distinct_from_notes_output_for_the_same_glosses():
    tokens = ["DEFINITION", "EXAMPLE", "QUESTION"]
    transcript = generate_transcript(tokens, mode="template")
    notes = generate_notes(tokens, mode="template", style="concise")
    assert transcript != notes
    assert "**Key Concepts:**" not in transcript  # not reusing the notes template
