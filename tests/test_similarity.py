"""Test Stage 1 cosine similarity against real traces."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from memshare.step_detector import detect_steps
from memshare.similarity import find_candidates

TRACES_DIR = Path(__file__).parent.parent / "data-collection" / "traces" / "aime24"


def test_edge_cases():
    """Empty inputs, single step, zero similarity."""
    # No steps
    assert find_candidates([], threshold=0.8) == []
    # Single step — no pairs to compare
    assert find_candidates(["just one step"], threshold=0.8) == []
    # Completely different steps — no candidates
    assert find_candidates(["alpha beta gamma", "99 100 101 202"], threshold=0.5) == []
    print("PASS: edge cases (empty, single, zero similarity)")


def test_identical_steps():
    """Identical steps should all be candidates."""
    steps = ["compute x squared plus three x plus five"] * 3
    candidates = find_candidates(steps, threshold=0.9)
    assert len(candidates) == 3, f"Expected 3 pairs, got {len(candidates)}"
    assert all(sim > 0.999 for _, _, sim in candidates)
    print("PASS: identical steps → all pairs matched (sim=1.0)")


def test_basic_similarity():
    """Near-identical steps should be candidates, different steps should not."""
    steps = [
        "Let me compute -3(4) - 6(2) - 5 = -12 - 12 - 5 = -29",
        "The weather is nice outside today and I like cats",
        "Wait, let me recompute -3(4) - 6(2) - 5 = -12 - 12 - 5 = -29",
    ]
    candidates = find_candidates(steps, threshold=0.7)
    assert len(candidates) == 1, f"Expected 1 candidate, got {len(candidates)}"
    assert candidates[0][0] == 0 and candidates[0][1] == 2
    print(f"PASS: basic similarity → 1 candidate (sim={candidates[0][2]:.3f})")


def test_real_trace():
    """Run full pipeline on AIME #60: detect steps → find candidates."""
    trace_file = TRACES_DIR / "60.json"
    if not trace_file.exists():
        print(f"SKIP: {trace_file} not found")
        return

    with open(trace_file) as f:
        data = json.load(f)

    trace = data.get("reasoning") or data.get("content", "")
    steps = detect_steps(trace)
    candidates = find_candidates(steps, threshold=0.8)

    print(f"\nAIME #60: {len(steps)} steps → {len(candidates)} candidate pairs (threshold=0.8)")
    print("-" * 70)
    for earlier, later, sim in candidates:
        preview_a = steps[earlier][:60].replace("\n", " ")
        preview_b = steps[later][:60].replace("\n", " ")
        print(f"  sim={sim:.3f}  step {earlier+1} ↔ step {later+1}")
        print(f"    A: {preview_a}...")
        print(f"    B: {preview_b}...")
        print()


def test_all_traces():
    """Run pipeline on all AIME traces, report summary."""
    if not TRACES_DIR.exists():
        print(f"SKIP: {TRACES_DIR} not found")
        return

    print("\nAll AIME traces (threshold=0.8):")
    print(f"  {'File':>10s}  {'Steps':>5s}  {'Candidates':>10s}")
    print(f"  {'----':>10s}  {'-----':>5s}  {'----------':>10s}")

    total_steps = 0
    total_candidates = 0

    for f in sorted(TRACES_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        trace = data.get("reasoning") or data.get("content", "")
        steps = detect_steps(trace)
        candidates = find_candidates(steps, threshold=0.8)
        total_steps += len(steps)
        total_candidates += len(candidates)
        print(f"  {f.name:>10s}  {len(steps):5d}  {len(candidates):10d}")

    print(f"\n  Total: {total_steps} steps, {total_candidates} candidate pairs")


if __name__ == "__main__":
    test_basic_similarity()
    test_real_trace()
    print("\n" + "=" * 70)
    test_all_traces()
