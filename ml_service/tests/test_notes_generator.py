"""Deterministic template notes -- notes_generator.py. Must always produce
usable Markdown with zero external dependencies (this is the fallback for
every LLM failure mode)."""
from notes_generator import template_notes_from_tokens, build_notes_prompt


def test_empty_gloss_list_produces_a_readable_message_not_a_crash():
    out = template_notes_from_tokens([])
    assert "No confident signs" in out


def test_spec_vocabulary_groups_into_expected_sections():
    out = template_notes_from_tokens(["DEFINITION", "EXAMPLE", "QUESTION", "HOMEWORK", "IMPORTANT"])
    assert "## Key Concepts" in out
    assert "## Questions" in out
    assert "## Tasks" in out
    assert "DEFINITION" in out
    assert "QUESTION" in out
    assert "HOMEWORK" in out


def test_real_vocabulary_without_structure_keywords_falls_back_to_detected_signs():
    out = template_notes_from_tokens(["Whistle", "Radish", "Market"])
    assert "## Detected Signs" in out
    assert "Whistle" in out
    assert "## Key Concepts" not in out


def test_output_is_never_empty_string():
    for tokens in ([], ["X"], ["QUESTION", "QUESTION", "QUESTION"]):
        assert len(template_notes_from_tokens(tokens)) > 0


def test_build_notes_prompt_includes_glosses_and_no_fact_invention_rule():
    prompt = build_notes_prompt(["DEFINITION", "EXAMPLE"], style="concise")
    assert "DEFINITION" in prompt
    assert "EXAMPLE" in prompt
    assert "invent" in prompt.lower()
