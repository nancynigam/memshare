"""
Stage 1: Cosine similarity between reasoning steps.

Converts each step into a bag-of-tokens vector and computes pairwise
cosine similarity. Step pairs above a threshold are candidates for
Stage 2 (KV block Euclidean distance).
"""

import math
from collections import Counter


def tokenize(text: str) -> Counter:
    """Convert text to a bag-of-tokens frequency vector (lowercased words)."""
    return Counter(text.lower().split())


def cosine_similarity(a: Counter, b: Counter) -> float:
    """Cosine similarity between two Counter vectors."""
    common = set(a) & set(b)
    if not common:
        return 0.0

    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def find_candidates(
    steps: list[str], threshold: float = 0.8
) -> list[tuple[int, int, float]]:
    """Find step pairs with cosine similarity above threshold.

    Args:
        steps: List of step texts from step_detector.detect_steps().
        threshold: Minimum cosine similarity to be a candidate.

    Returns:
        List of (earlier_step_idx, later_step_idx, similarity) tuples,
        sorted by similarity descending.
    """
    vectors = [tokenize(step) for step in steps]
    candidates = []

    for i in range(1, len(vectors)):
        for j in range(i):
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                candidates.append((j, i, sim))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates
