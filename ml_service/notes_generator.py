"""Deterministic (LLM-free) and LLM-prompt note generation from gloss
sequences, in three genuinely different styles: concise / detailed /
academic.

Two independent generators exist for each style, and BOTH must exist for
every style -- not just one generic implementation with a style word
swapped in:

  1. build_notes_prompt() -- the LLM prompt. Each style gets its own
     prompt-building function with a different structure, not the same
     paragraph with one adjective changed (see PROMPT REQUIREMENTS below).
  2. template_notes_from_tokens() -- the deterministic, LLM-free fallback
     (Test 3 in the acceptance tests: the app must remain usable with the
     LLM off, timing out, or returning garbage). This is NOT claimed to be
     LLM-quality -- it's a fixed template that reorganizes recognized
     glosses using keyword matching, and every style's fallback says so.

Both generators treat the gloss sequence as imperfect, constrained-
vocabulary recognizer output, never as a verbatim transcript -- see the
shared anti-hallucination rules baked into every LLM prompt below (project
brief section 29): no invented facts, names, dates, formulas, examples;
no inferring a whole lecture from one generic word; preserve recognized
terminology; stay conservative when ambiguous.
"""

from typing import Dict, List, Optional, Sequence

SUPPORTED_STYLES = ("concise", "detailed", "academic")
DEFAULT_STYLE = "concise"

# Prompt versions -- bump the relevant one whenever a prompt's wording or
# required structure changes, so generated notes can be traced back to the
# exact prompt that produced them (returned as `prompt_version` by the
# /notes and /transcript APIs).
PROMPT_VERSIONS = {
    "concise": "concise-v2",
    "detailed": "detailed-v2",
    "academic": "academic-v2",
}
TRANSCRIPT_PROMPT_VERSION = "transcript-v2"

# A gloss whose recognition confidence is below this is marked "(?)" in
# prompts and in the deterministic fallback, so the uncertainty is carried
# through to the notes instead of being stated as fact.
UNCERTAIN_CONFIDENCE = 0.6
MAX_CONTEXT_CHARS = 300


def _collapse_consecutive(tokens: Sequence[str]) -> List[str]:
    """X X X Y -> X Y. Consecutive repeats are usually one held sign
    recognized over several windows, not separate points."""
    out: List[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def _mark_uncertain(tokens: Sequence[str], confidences: Optional[Sequence[float]]) -> List[str]:
    if not confidences or len(confidences) != len(tokens):
        return list(tokens)
    return [
        f"{t} (?)" if (c is not None and float(c) < UNCERTAIN_CONFIDENCE) else t
        for t, c in zip(tokens, confidences)
    ]


def _gloss_line(tokens: Sequence[str], confidences: Optional[Sequence[float]] = None) -> str:
    return ", ".join(_mark_uncertain(tokens, confidences))


def _context_block(context: Optional[str]) -> str:
    if not context or not context.strip():
        return ""
    c = " ".join(context.split())[:MAX_CONTEXT_CHARS]
    return (
        "\nUser-provided context (typed by the user, NOT recognized from signing): "
        f"\"{c}\"\nYou may use it only to choose a title and to disambiguate a gloss's "
        "wording. It is NOT a source of facts: do not add any content from it that the "
        "glosses do not support.\n"
    )

# ---------------------------------------------------------------------------
# Shared, style-independent step: bucket glosses by keyword. Both the
# template fallback AND (indirectly, by informing what's genuinely in the
# input) the LLM prompts rely on this -- it doesn't invent structure the
# glosses don't support, it just recognizes lecture-relevant keywords when
# they're literally present in a gloss.
# ---------------------------------------------------------------------------
_SECTION_KEYWORDS = {
    "Key Concepts": ["definition", "example", "concept", "important", "key"],
    "Questions": ["question", "ask", "doubt"],
    "Tasks": ["homework", "assignment", "task", "deadline"],
}
_SECTION_ORDER = ["Key Concepts", "Questions", "Tasks", "Detected Signs"]


def _categorize(tokens: List[str]) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {name: [] for name in _SECTION_ORDER}
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


_FALLBACK_DISCLAIMER = (
    "_Notes generated from recognized sign-language glosses using a "
    "constrained-vocabulary recognizer. Verify low-confidence or "
    "ambiguous content before relying on it. This is the deterministic "
    "fallback (no LLM was used) and is not equivalent in quality to the "
    "LLM-generated version._"
)


# ---------------------------------------------------------------------------
# Deterministic fallback -- one rendering function per style, all built on
# the same _categorize() buckets, but genuinely different in structure
# (per RULE 10 / section 30: fallback should be style-aware, not one
# generic template).
# ---------------------------------------------------------------------------

def _template_concise(tokens: List[str], title: str) -> str:
    """CONCISE fallback: the flattest possible representation -- a title
    and, per non-empty bucket, a one-line bullet list. No intros, no
    connective sentences, optimized for fast revision scanning."""
    buckets = _categorize(tokens)
    lines = [f"# {title}", ""]
    for section in _SECTION_ORDER:
        items = buckets[section]
        if not items:
            continue
        lines.append(f"- **{section}:** " + ", ".join(items))
    lines += ["", "---", _FALLBACK_DISCLAIMER]
    return "\n".join(lines)


def _template_detailed(tokens: List[str], title: str) -> str:
    """DETAILED fallback: full section headers, one bullet per gloss
    (not merged into a comma list), and a short factual intro sentence per
    section describing what was found -- more structure than concise, but
    still nothing invented beyond what the buckets contain."""
    buckets = _categorize(tokens)
    intros = {
        "Key Concepts": "Concepts recognized during this session:",
        "Questions": "Questions raised during the session:",
        "Tasks": "Tasks or assignments mentioned:",
        "Detected Signs": "Other signs recognized during this session:",
    }
    lines = [f"# {title}", ""]
    any_section = False
    for section in _SECTION_ORDER:
        items = buckets[section]
        if not items:
            continue
        any_section = True
        lines += [f"## {section}", "", intros[section]]
        lines += [f"- {item}" for item in items]
        lines.append("")
    if not any_section:
        lines += ["## Detected Signs", ""] + [f"- {t}" for t in tokens] + [""]
    lines += ["---", _FALLBACK_DISCLAIMER]
    return "\n".join(lines)


def _template_academic(tokens: List[str], title: str) -> str:
    """ACADEMIC fallback: formal heading hierarchy (Lecture Notes / Core
    Concepts / Questions & Clarifications / Tasks & Follow-up), a session
    metadata line, and formal phrasing throughout -- still zero invented
    content, just a different register and structure from concise/detailed."""
    buckets = _categorize(tokens)
    lines = [f"# {title}", "", f"*Session summary: {len(tokens)} recognized sign(s).*", ""]
    section_map = [
        ("Key Concepts", "## Core Concepts", "The following concepts were identified in this session:"),
        ("Questions", "## Questions & Clarifications", "The following questions were raised:"),
        ("Tasks", "## Tasks & Follow-up", "The following tasks or assignments were noted:"),
        ("Detected Signs", "## Additional Recognized Signs", "The following additional signs were recognized:"),
    ]
    any_section = False
    for bucket_name, header, intro in section_map:
        items = buckets[bucket_name]
        if not items:
            continue
        any_section = True
        lines += [header, "", intro]
        lines += [f"- {item}" for item in items]
        lines.append("")
    if not any_section:
        lines += ["## Additional Recognized Signs", ""] + [f"- {t}" for t in tokens] + [""]
    lines += ["---", _FALLBACK_DISCLAIMER]
    return "\n".join(lines)


_TEMPLATE_RENDERERS = {
    "concise": _template_concise,
    "detailed": _template_detailed,
    "academic": _template_academic,
}


def template_notes_from_tokens(tokens: List[str], title: str = "Lecture Notes", style: str = DEFAULT_STYLE,
                               confidences: Optional[Sequence[float]] = None) -> str:
    """Deterministic, LLM-free notes generator -- always available, no
    external service, no network call. This is the fallback for every LLM
    failure mode (down, timeout, malformed response, network error,
    RULE 17). `style` selects which of the three genuinely different
    renderings above to use; unknown styles fall back to 'concise'.

    Repetition: consecutive repeats are collapsed for every style (one
    held sign recognized several times); concise additionally removes
    non-adjacent duplicates ("no unnecessary repetition"). Uncertain
    glosses (confidence < UNCERTAIN_CONFIDENCE, when confidences are
    given) are marked "(?)" rather than stated as fact."""
    if not tokens:
        return f"# {title}\n\nNo confident signs were detected. Please repeat the gesture."
    marked = _mark_uncertain(tokens, confidences)
    marked = _collapse_consecutive(marked)
    if style == "concise" or style not in _TEMPLATE_RENDERERS:
        seen = set()
        marked = [t for t in marked if not (t in seen or seen.add(t))]
    renderer = _TEMPLATE_RENDERERS.get(style, _template_concise)
    out = renderer(marked, title)
    if any(t.endswith("(?)") for t in marked):
        out = out.replace(
            _FALLBACK_DISCLAIMER,
            "_Signs marked (?) were recognized with low confidence._\n\n" + _FALLBACK_DISCLAIMER,
        )
    return out


# ---------------------------------------------------------------------------
# LLM prompts -- one genuinely different prompt-building function per
# style (RULE 10: not the same prompt with a style word swapped in).
# Every prompt includes the FULL anti-hallucination rule set (section 29),
# not a shortened version, because the risk (an LLM turning one recognized
# gloss into invented lecture content) is the same regardless of style.
# ---------------------------------------------------------------------------

_ANTI_HALLUCINATION_RULES = """\
The gloss sequence is imperfect, constrained-vocabulary recognizer output
-- NOT a transcript of speech or a complete record of what was signed.
Follow these rules without exception:
- Do not invent facts, names, dates, formulas, or examples not present in
  the gloss sequence.
- Do not infer an entire lecture, conversation, or topic from one generic
  or ambiguous gloss.
- Do not claim to have heard audio or to have access to any information
  beyond the gloss sequence given below.
- Preserve the recognized terminology -- do not silently substitute a
  different word for a recognized gloss.
- If the glosses are too sparse or ambiguous to produce meaningful prose,
  produce a short structured list instead of padding with invented detail.
- A gloss marked "(?)" was recognized with LOW confidence: either omit it
  or mention it hedged (e.g. "possibly X") -- never state it as fact.
- Consecutive repeats of the same gloss are usually one sign held for a
  while; treat them as a single mention unless the repetition is
  clearly meaningful.
- Never mention confidence scores, model internals, or that this came
  from an AI recognizer -- write as if these are the user's own notes.
- Return Markdown only -- no commentary before or after the notes.\
"""


def _concise_prompt(tokens: List[str], gloss_line: str, context_block: str) -> str:
    """CONCISE (quick revision): a short title, the essential points as
    flat bullets, and an "Important terms" line -- explicitly asks for
    compression and forbids explanations, as its own instruction set."""
    return f"""You are generating QUICK REVISION NOTES for a student from a sign-language
recognition session, using Sign2Notes.

Glosses (in recognized order): {gloss_line}
{context_block}
Produce the most compact possible notes, using EXACTLY this structure:
# <short, readable title derived only from the glosses>
- <essential point>
- <essential point>
**Important terms:** <comma-separated recognized terms>

Rules for this style:
- Flat bullets only -- no sub-bullets, no paragraphs, no explanations.
- Remove repetition: if a concept recurs, state it once.
- If a question or task gloss is present, give it one bullet, not a section.
- Keep the whole output under roughly 10 bullets.

{_ANTI_HALLUCINATION_RULES}
"""


def _detailed_prompt(tokens: List[str], gloss_line: str, context_block: str) -> str:
    """DETAILED (complete study notes): topic/subtopic organization, an
    explanation of each recognized concept limited to what the glosses
    support, relationships only where evidenced, examples/questions/tasks
    only when the glosses themselves contain them."""
    return f"""You are generating DETAILED STUDY NOTES for a student from a sign-language
recognition session, using Sign2Notes.

Glosses (in recognized order): {gloss_line}
{context_block}
Produce complete, well-organized study material:
- Start with "# <topic>" and organize content under "## <subtopic>" headings
  that group related glosses (one subtopic per coherent group of glosses).
- Use whichever of these Markdown sections are actually supported by the
  glosses (omit any section with nothing to put in it -- do not force all
  of them to appear):
  ## Key Concepts
  ## Explanation
  ## Examples
  ## Important Points
  ## Questions
  ## Tasks / Follow-up
- In "Explanation", describe each recognized concept only as far as the
  gloss itself supports. You may describe a relationship BETWEEN two or
  more glosses only if it is directly supported by their sequence or
  literal meaning (e.g. a concept gloss immediately followed by an example
  gloss can be described as "an example was given for the preceding
  concept") -- never invent a relationship that isn't evidenced.
- "Examples", "Questions" and "Tasks / Follow-up" may appear ONLY if the
  glosses explicitly contain an example, a question or a task.
- It is fine for this output to be longer than a quick-revision summary,
  but every sentence must still be traceable to specific glosses.

{_ANTI_HALLUCINATION_RULES}
"""


def _academic_prompt(tokens: List[str], gloss_line: str, context_block: str) -> str:
    """ACADEMIC (formal university-level notes): formal title and numbered
    section hierarchy, definitions only where a gloss is a term, formal
    paragraphs plus structured lists, explicit distinctions between
    concepts -- and an explicit ban on academic-sounding filler."""
    return f"""You are generating FORMAL, UNIVERSITY-LEVEL LECTURE NOTES from a sign-language
recognition session, using Sign2Notes.

Glosses (in recognized order): {gloss_line}
{context_block}
Produce formally structured academic notes:
- Begin with a formal title ("# ...") followed by a one-sentence formal
  statement of scope that names only the recognized topics.
- Use a hierarchical Markdown structure (### subsections where a section
  has distinct parts), choosing only the sections genuinely supported by
  the glosses (omit any with nothing to put in it):
  ## Core Concepts
  ## Definitions
  ## Conceptual Relationships
  ## Examples
  ## Questions / Clarifications
  ## Tasks / Follow-up
- Write formal, coherent paragraphs where the glosses support connected
  statements, and structured lists elsewhere. Use precise technical
  terminology -- the recognized terms themselves -- with no casual
  phrasing, contractions, or conversational asides.
- Under "Definitions", only include a gloss if it is plausibly a
  term/concept on its own (e.g. a noun-like gloss) -- do not manufacture
  a definition's contents beyond restating that the term was introduced.
- Under "Conceptual Relationships", state a relationship or distinction
  between glosses ONLY if their sequence or literal meaning directly
  supports it -- do not fabricate connections for the sake of sounding
  more rigorous.
- Distinguish clearly between concepts and questions/tasks -- do not mix
  a question into "Core Concepts", for example.
- Do NOT add academic-sounding filler (background, motivation, history,
  significance, implications) to compensate for sparse glosses; sparse
  input must produce short notes.
- Never invent a citation, reference, or source.

{_ANTI_HALLUCINATION_RULES}
"""


_PROMPT_BUILDERS = {
    "concise": _concise_prompt,
    "detailed": _detailed_prompt,
    "academic": _academic_prompt,
}


def build_notes_prompt(tokens: List[str], style: str = DEFAULT_STYLE,
                       confidences: Optional[Sequence[float]] = None,
                       context: Optional[str] = None) -> str:
    """Prompt for the LLM-backed notes generator in infer.py. `style`
    selects one of three genuinely different prompt-building functions
    (see module docstring / RULE 10) -- unknown styles fall back to
    'concise'. Low-confidence glosses are marked "(?)" (see
    UNCERTAIN_CONFIDENCE); `context` is optional user-typed context that
    the prompt explicitly forbids using as a source of facts."""
    builder = _PROMPT_BUILDERS.get(style, _concise_prompt)
    return builder(tokens, _gloss_line(tokens, confidences), _context_block(context))


# ---------------------------------------------------------------------------
# Natural-language TRANSCRIPT -- Live Transcription mode (Mode 3).
#
# This is deliberately a DIFFERENT output layer from notes (see project
# brief section 58's "three output layers"):
#   LAYER 1: recognition        -- QUESTION -> IMPORTANT -> EXAM (glosses)
#   LAYER 2: natural-language transcript -- "A question was raised about
#            an important exam topic." (flowing prose, THIS section)
#   LAYER 3: structured notes   -- "# Lecture Notes\n## Important Topic..."
#            (build_notes_prompt / template_notes_from_tokens, above)
#
# A transcript is NOT a set of notes with the headers stripped out -- it
# reads as continuous sentences, in gloss order, the way a (rough, still
# imperfect) spoken transcript would. Both a template fallback and an LLM
# prompt exist for the same reason as notes: the live session must keep
# working with the LLM off (RULE 17), and Live Transcription mode
# specifically only calls the LLM ONCE, when the user stops the session
# (project brief section 21) -- never per-gloss, never continuously.
# ---------------------------------------------------------------------------

_TRANSCRIPT_FALLBACK_DISCLAIMER = (
    "_Transcript assembled directly from recognized glosses (no LLM was "
    "used) -- a rough, literal rendering of the recognized signs in "
    "order, not natural prose. Verify low-confidence or ambiguous "
    "content before relying on it._"
)


def template_transcript_from_tokens(tokens: List[str]) -> str:
    """Deterministic, LLM-free transcript fallback. Does NOT attempt to
    synthesize grammatical prose from a bag of glosses (that would mean
    inventing sentence structure the recognizer never provided evidence
    for) -- it's an honest, literal rendering: the recognized signs,
    joined in order, clearly labeled as not natural language. Always
    available, no external service, no network call (RULE 17)."""
    if not tokens:
        return "No confident signs were detected during this session."
    literal = " \u2192 ".join(tokens)
    return (
        f"The following signs were recognized, in order: {literal}.\n\n"
        f"{_TRANSCRIPT_FALLBACK_DISCLAIMER}"
    )


def build_transcript_prompt(tokens: List[str], confidences: Optional[Sequence[float]] = None) -> str:
    """LLM prompt for a natural-language TRANSCRIPT (Layer 2) -- flowing
    prose, not structured notes. Genuinely different task from
    build_notes_prompt(): a transcript reads like continuous sentences a
    student could read aloud, in gloss order; it does not use headings,
    bullets, or sections at all."""
    return f"""You are converting a sequence of recognized sign-language glosses into a
SHORT NATURAL-LANGUAGE TRANSCRIPT, using Sign2Notes.

Glosses (in recognized order): {_gloss_line(tokens, confidences)}

Produce a brief, flowing transcript:
- Write continuous prose sentences, in the SAME order as the glosses --
  not headings, not bullet points, not a structured document.
- This should read like a rough transcript of what was signed, not a
  summary or a set of study notes -- keep it close to the literal
  sequence of glosses rather than reorganizing them by topic.
- Only combine two glosses into one sentence if their adjacency and
  literal meaning directly supports it (e.g. a concept gloss immediately
  followed by an example gloss can become "X, for example Y.") -- do not
  invent connective content between unrelated glosses.
- If the glosses don't support coherent prose (too sparse, too
  disconnected), it is fine for the transcript to be a short, plain
  sentence per gloss rather than a forced narrative.
- Keep it brief -- a few sentences, not a full essay.
- Do not use Markdown headings or bullet lists in this output -- plain
  prose paragraphs only.

{_ANTI_HALLUCINATION_RULES}
"""
