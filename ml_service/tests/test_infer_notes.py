"""generate_notes()'s LLM-failure fallback -- infer.py. No LLM server is
running in the test environment, so every "llm" mode call here must fall
back to template notes rather than raising."""
from infer import generate_notes


def test_template_mode_never_touches_the_network():
    out = generate_notes(["DEFINITION", "QUESTION"], mode="template")
    assert "DEFINITION" in out


def test_llm_mode_falls_back_to_template_when_server_unreachable():
    out = generate_notes(["DEFINITION", "QUESTION"], mode="llm")
    # falls back to the same deterministic template output
    assert "DEFINITION" in out
    assert "## " in out  # markdown headers from the template engine


def test_legacy_ollama_model_kwarg_still_works():
    # older callers used ollama_model=; generate_notes() must not crash
    # when called this way even though the parameter is now llm_model
    out = generate_notes(["DEFINITION"], mode="template", ollama_model="some-model")
    assert "DEFINITION" in out
