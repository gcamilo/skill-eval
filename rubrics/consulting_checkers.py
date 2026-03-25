"""Consulting rubric — programmatic checkers.

Each function takes the output text and returns a dict of checker results.
These are injected into the judge prompt as factual context so the LLM judge
can ground scores in measurable signals rather than pure vibes.

To create checkers for a new rubric, implement a module with a single
`run_checkers(text: str) -> dict` function and a `format_checker_context(checks: dict) -> str`
function. Register the module path in your rubric JSON as `checker_module`.
"""

import re

# ---------------------------------------------------------------------------
# Evidence label patterns — [F], [A], [I] and variants
# ---------------------------------------------------------------------------

EVIDENCE_PATTERNS = [
    r"\[F\]",            # [F]
    r"\[A\]",            # [A]
    r"\[I\]",            # [I]
    r"\[Fact\]",         # [Fact]
    r"\[Assumption\]",   # [Assumption]
    r"\[Inference\]",    # [Inference]
    r"\[F,",             # [F, 70% confidence]
    r"\[A,",             # [A, needs validation]
    r"\[I,",             # [I, based on...]
]

# ---------------------------------------------------------------------------
# Hard gate patterns — should appear in first 25% of text
# ---------------------------------------------------------------------------

HARD_GATE_PATTERNS = [
    r"(?i)provisional\s+assumption",
    r"(?i)before\s+(proceeding|I\s+can|diving|analyzing)",
    r"(?i)I\s+need\s+to\s+(clarif|understand|know)",
    r"(?i)assumptions?\s*(\[A\])?:",
    r"(?i)scope\s*(boundaries|:)",
    r"(?i)what\s+I\s+don't\s+know",
    r"(?i)assuming\s+(the\s+following|that)",
    r"(?i)key\s+unknowns?",
]

# ---------------------------------------------------------------------------
# Devil's advocate / counter-argument patterns
# ---------------------------------------------------------------------------

DEVILS_ADVOCATE_PATTERNS = [
    r"(?i)devil'?s?\s+advocate",
    r"(?i)counter[- ]?argument",
    r"(?i)what\s+if\s+we'?re\s+wrong",
    r"(?i)risk[s]?\s*(and\s+mitigation|:|\n)",
    r"(?i)pre[- ]?mortem",
    r"(?i)strongest\s+(counter|objection|argument\s+against)",
    r"(?i)what\s+would\s+change",
    r"(?i)##\s*risk",
    r"(?i)\*\*risk",
]

# ---------------------------------------------------------------------------
# Framework names — for counting distinct frameworks mentioned
# ---------------------------------------------------------------------------

FRAMEWORK_NAMES = [
    r"(?i)porter'?s?\s+five\s+forces",
    r"(?i)value\s+chain",
    r"(?i)7[- ]?S\s+(framework|model|analysis)",
    r"(?i)VRIO",
    r"(?i)blue\s+ocean",
    r"(?i)ansoff",
    r"(?i)BCG\s+matrix",
    r"(?i)growth[- ]share\s+matrix",
    r"(?i)MECE",
    r"(?i)issue\s+tree",
    r"(?i)hypothesis\s+tree",
    r"(?i)pyramid\s+principle",
    r"(?i)RAPID",
    r"(?i)decision\s+matrix",
    r"(?i)pre[- ]?mortem",
    r"(?i)SWOT",
    r"(?i)PESTEL",
    r"(?i)TAM\s*/?\s*SAM\s*/?\s*SOM",
    r"(?i)unit\s+economics",
    r"(?i)profit\s+tree",
    r"(?i)revenue\s+tree",
    r"(?i)jobs[- ]to[- ]be[- ]done",
    r"(?i)business\s+model\s+canvas",
    r"(?i)ADKAR",
    r"(?i)Kotter",
    r"(?i)Lewin",
    r"(?i)NPS",
    r"(?i)customer\s+journey",
    r"(?i)value[- ]based\s+pricing",
    r"(?i)wardley\s+map",
    r"(?i)scenario\s+planning",
    r"(?i)second[- ]order\s+thinking",
    r"(?i)inversion",
    r"(?i)SCQA",
    r"(?i)3\s+horizons?",
]


# ---------------------------------------------------------------------------
# Individual checker functions
# ---------------------------------------------------------------------------

def count_evidence_labels(text: str) -> int:
    """Count evidence label tags ([F], [A], [I] and variants)."""
    count = 0
    for pattern in EVIDENCE_PATTERNS:
        count += len(re.findall(pattern, text))
    return count


def has_hard_gate(text: str) -> bool:
    """Check if the output has an upfront hard gate / assumptions section.

    Looks in the first 25% of text (minimum 500 chars) for patterns
    indicating the author paused to state assumptions before analyzing.
    """
    cutoff = len(text) // 4
    head = text[: max(cutoff, 500)]
    for pattern in HARD_GATE_PATTERNS:
        if re.search(pattern, head):
            return True
    return False


def has_devils_advocate(text: str) -> bool:
    """Check for a dedicated counter-argument / risk section."""
    for pattern in DEVILS_ADVOCATE_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def framework_count(text: str) -> int:
    """Count distinct frameworks mentioned in the text."""
    found = set()
    for pattern in FRAMEWORK_NAMES:
        if re.search(pattern, text):
            found.add(pattern)
    return len(found)


# ---------------------------------------------------------------------------
# Public API — called by scoring.py
# ---------------------------------------------------------------------------

def run_checkers(text: str) -> dict:
    """Run all programmatic checkers on the output text.

    Returns a dict of checker results that will be passed to
    ``format_checker_context`` and then injected into the judge prompt.
    """
    return {
        "evidence_label_count": count_evidence_labels(text),
        "has_hard_gate": has_hard_gate(text),
        "has_devils_advocate": has_devils_advocate(text),
        "framework_count": framework_count(text),
        "output_length": len(text),
    }


def format_checker_context(checks: dict) -> str:
    """Format checker results as a string for injection into the judge prompt.

    The judge uses these facts to anchor its scores — e.g. if 0 evidence
    labels were found, evidence_labeling cannot score above 4.
    """
    parts = [
        f"Evidence labels found: {checks['evidence_label_count']}",
        f"Hard gate / assumptions section: {'Yes' if checks['has_hard_gate'] else 'No'}",
        f"Devil's advocate / risk section: {'Yes' if checks['has_devils_advocate'] else 'No'}",
        f"Distinct frameworks mentioned: {checks['framework_count']}",
        f"Output length: {checks['output_length']} characters",
    ]
    return "AUTOMATED ANALYSIS:\n" + "\n".join(f"- {p}" for p in parts)
