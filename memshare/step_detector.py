"""
Step boundary detector for reasoning model traces.

Segments a chain-of-thought reasoning trace into discrete "steps" by
detecting re-entry phrases ("Wait, let me verify...", "Another thought:", etc.).

MVP: processes complete traces as strings.
Future: token-by-token streaming for vLLM decode loop integration.
"""

import re

# Boundary patterns derived from manual annotation of AIME 2024 traces.
# Each pattern marks the START of a new reasoning step.
# Order doesn't matter — they're combined into one regex.
BOUNDARY_PATTERNS = [
    r"Wait,?\s",
    r"Hmm,?\s",
    r"Let me verify",
    r"Let me check",
    r"Let me reconsider",
    r"Let me re-?compute",
    r"Let me re-?calculate",
    r"Let me see",
    r"Let me think",
    r"Let me try",
    r"Another thought:",
    r"Another idea:",
    r"Another approach:",
    r"Actually,?\s",
    r"But wait,?\s",
    r"But let me",
    r"Perhaps I should",
    r"Is that correct\??",
    r"Is that true\??",
    r"Is that right\??",
    r"I recall that",
    r"I think I recall",
    r"I remember that",
]

# Compile into a single pattern that matches at line beginnings
# (after a newline) or after sentence-ending punctuation.
_BOUNDARY_RE = re.compile(
    r"(?:^|\n)\s*(" + "|".join(BOUNDARY_PATTERNS) + r")",
    re.IGNORECASE,
)


def detect_steps(trace: str, min_step_chars: int = 40) -> list[str]:
    """Split a reasoning trace into steps at boundary markers.

    Args:
        trace: Full reasoning trace text.
        min_step_chars: Minimum characters for a step. Boundaries that would
            create shorter steps are ignored (avoids over-splitting on
            phrases that appear mid-sentence).

    Returns:
        List of step strings, in order. The full trace is covered —
        concatenating all steps reproduces the original text
        (minus leading/trailing whitespace per step).
    """
    if not trace.strip():
        return []

    splits = list(_BOUNDARY_RE.finditer(trace))

    if not splits:
        return [trace.strip()]

    steps = []
    prev_start = 0

    for match in splits:
        # Split point is the start of the matched boundary phrase
        split_pos = match.start()

        # Don't split if it would create a too-short step
        if split_pos - prev_start < min_step_chars:
            continue

        step_text = trace[prev_start:split_pos].strip()
        if step_text:
            steps.append(step_text)
        prev_start = split_pos

    # Last step: everything after the final split
    last = trace[prev_start:].strip()
    if last:
        steps.append(last)

    return steps
