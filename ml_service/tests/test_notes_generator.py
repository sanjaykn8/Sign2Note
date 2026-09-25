"""notes_generator.py -- deterministic template notes (must always produce
usable Markdown with zero external dependencies, for every LLM failure
mode) and the three genuinely different LLM prompt styles."""
from notes_generator import (
    SUPPORTED_STYLES,
    build_notes_prompt,
    build_transcript_prompt,
    template_notes_from_tokens,
    template_transcript_from_tokens,
)


# ---------------------------------------------------------------------------
# Deterministic fallback -- shared behavior
# ---------------------------------------------------------------------------

def test_empty_gloss_list_produces_a_readable_message_not_a_crash():
    for style in SUPPORTED_STYLES:
        out = template_notes_from_tokens([], style=style)
        assert "No confident signs" in out


def test_output_is_never_empty_string_for_any_style():
    for style in SUPPORTED_STYLES:
        for tokens in ([], ["X"], ["QUESTION", "QUESTION", "QUESTION"]):
            assert len(template_notes_from_tokens(tokens, style=style)) > 0


def test_unknown_style_falls_back_to_concise_instead_of_crashing():
    out = template_notes_from_tokens(["QUESTION"], style="not_a_real_style")
    assert out == template_notes_from_tokens(["QUESTION"], style="concise")


def test_fallback_output_labels_itself_as_the_deterministic_fallback():
    # RULE: "do not claim the fallback is equivalent to LLM quality."
    for style in SUPPORTED_STYLES:
        out = template_notes_from_tokens(["QUESTION"], style=style)
        assert "fallback" in out.lower()
        assert "no llm was used" in out.lower()


# ---------------------------------------------------------------------------
# Deterministic fallback -- per-style structural differences (not just one
# generic template with a style word swapped in)
# ---------------------------------------------------------------------------

def test_concise_fallback_is_flat_bullets_not_full_headers():
    out = template_notes_from_tokens(["DEFINITION", "QUESTION", "HOMEWORK"], style="concise")
    assert "## Key Concepts" not in out
    assert "## Questions" not in out
    assert "**Key Concepts:**" in out
    assert "**Questions:**" in out
    assert "DEFINITION" in out and "QUESTION" in out and "HOMEWORK" in out


def test_detailed_fallback_uses_full_section_headers_and_intro_sentences():
    out = template_notes_from_tokens(["DEFINITION", "EXAMPLE", "QUESTION", "HOMEWORK", "IMPORTANT"], style="detailed")
    assert "## Key Concepts" in out
    assert "## Questions" in out
    assert "## Tasks" in out
    assert "Concepts recognized during this session:" in out
    assert "- DEFINITION" in out  # one bullet per gloss, not a comma list


def test_academic_fallback_uses_formal_headers_distinct_from_detailed():
    out = template_notes_from_tokens(["DEFINITION", "QUESTION", "HOMEWORK"], style="academic")
    assert "## Core Concepts" in out
    assert "## Questions & Clarifications" in out
    assert "## Tasks & Follow-up" in out
    assert "Session summary:" in out
    # Academic's headers are genuinely different strings from detailed's.
    detailed = template_notes_from_tokens(["DEFINITION", "QUESTION", "HOMEWORK"], style="detailed")
    assert "## Core Concepts" not in detailed
    assert "## Key Concepts" not in out


def test_real_vocabulary_without_structure_keywords_falls_back_to_detected_signs():
    for style in SUPPORTED_STYLES:
        out = template_notes_from_tokens(["Whistle", "Radish", "Market"], style=style)
        assert "Whistle" in out
        assert "Key Concepts" not in out


def test_all_three_style_outputs_are_different_from_each_other():
    tokens = ["DEFINITION", "EXAMPLE", "QUESTION", "HOMEWORK"]
    outputs = {style: template_notes_from_tokens(tokens, style=style) for style in SUPPORTED_STYLES}
    assert outputs["concise"] != outputs["detailed"] != outputs["academic"]
    assert outputs["concise"] != outputs["academic"]


# ---------------------------------------------------------------------------
# LLM prompts -- genuinely different structure per style, not one prompt
# with a style word swapped in
# ---------------------------------------------------------------------------

def test_all_three_prompts_include_glosses_and_the_full_anti_hallucination_ruleset():
    for style in SUPPORTED_STYLES:
        prompt = build_notes_prompt(["DEFINITION", "EXAMPLE"], style=style)
        assert "DEFINITION" in prompt
        assert "EXAMPLE" in prompt
        assert "invent" in prompt.lower()
        assert "confidence scores" in prompt.lower()  # "don't mention confidence" rule present


def test_prompts_are_not_the_same_template_with_a_word_swapped():
    tokens = ["DEFINITION", "EXAMPLE"]
    concise = build_notes_prompt(tokens, style="concise")
    detailed = build_notes_prompt(tokens, style="detailed")
    academic = build_notes_prompt(tokens, style="academic")
    # If these were "one generic prompt, one word changed," the diff
    # between them would be tiny. Require a large structural difference.
    assert len(set(concise.split()) ^ set(detailed.split())) > 15
    assert len(set(detailed.split()) ^ set(academic.split())) > 15


def test_concise_prompt_asks_for_compactness_specifically():
    prompt = build_notes_prompt(["DEFINITION"], style="concise")
    assert "compact" in prompt.lower() or "quick" in prompt.lower()


def test_detailed_prompt_mentions_explanation_section_and_relationship_evidence_rule():
    prompt = build_notes_prompt(["DEFINITION", "EXAMPLE"], style="detailed")
    assert "## Explanation" in prompt
    assert "relationship" in prompt.lower()


def test_academic_prompt_uses_formal_hierarchical_sections():
    prompt = build_notes_prompt(["DEFINITION"], style="academic")
    assert "## Definitions" in prompt
    assert "## Conceptual Relationships" in prompt
    assert "formal" in prompt.lower()


def test_unknown_style_prompt_falls_back_to_concise():
    tokens = ["DEFINITION"]
    assert build_notes_prompt(tokens, style="not_a_real_style") == build_notes_prompt(tokens, style="concise")


# ---------------------------------------------------------------------------
# Natural-language TRANSCRIPT (Layer 2) -- genuinely different from notes
# (Layer 3), see module docstring's "three output layers"
# ---------------------------------------------------------------------------

def test_transcript_template_handles_empty_gloss_list():
    out = template_transcript_from_tokens([])
    assert "No confident signs" in out


def test_transcript_template_is_literal_not_grammatical_invention():
    tokens = ["DEFINITION", "EXAMPLE", "QUESTION"]
    out = template_transcript_from_tokens(tokens)
    for t in tokens:
        assert t in out
    # order preserved
    assert out.index("DEFINITION") < out.index("EXAMPLE") < out.index("QUESTION")


def test_transcript_template_labels_itself_as_the_fallback():
    out = template_transcript_from_tokens(["QUESTION"])
    assert "no llm was used" in out.lower()


def test_transcript_template_output_differs_from_notes_output():
    # Not the same rendering with a different label -- genuinely different
    # structure (prose-ish literal sequence vs. categorized bullets).
    tokens = ["DEFINITION", "EXAMPLE", "QUESTION", "HOMEWORK"]
    transcript = template_transcript_from_tokens(tokens)
    notes = template_notes_from_tokens(tokens, style="concise")
    assert transcript != notes
    assert "**Key Concepts:**" not in transcript


def test_transcript_prompt_asks_for_prose_not_structure():
    prompt = build_transcript_prompt(["DEFINITION", "EXAMPLE"])
    assert "DEFINITION" in prompt
    assert "EXAMPLE" in prompt
    assert "heading" in prompt.lower() or "bullet" in prompt.lower()  # explicitly told not to use them
    assert "invent" in prompt.lower()  # anti-hallucination rules present


def test_transcript_prompt_is_not_the_notes_prompt_with_a_word_swapped():
    tokens = ["DEFINITION", "EXAMPLE"]
    transcript_prompt = build_transcript_prompt(tokens)
    notes_prompt = build_notes_prompt(tokens, style="concise")
    assert len(set(transcript_prompt.split()) ^ set(notes_prompt.split())) > 15
