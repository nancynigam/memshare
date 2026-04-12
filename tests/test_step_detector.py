"""Test step detector against real AIME 2024 traces."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from memshare.step_detector import detect_steps

TRACES_DIR = Path(__file__).parent.parent / "data-collection" / "traces" / "aime24"


def test_basic_splitting():
    """Verify that a simple synthetic trace splits correctly."""
    trace = (
        "I need to solve this problem. First, compute 2 + 2 = 4.\n\n"
        "Wait, let me verify that. 2 + 2 is indeed 4.\n\n"
        "Another thought: maybe I should check 3 + 3 too.\n\n"
        "Let me see if this makes sense. Yes it does."
    )
    steps = detect_steps(trace)
    assert len(steps) == 4, f"Expected 4 steps, got {len(steps)}"
    assert steps[0].startswith("I need to solve")
    assert steps[1].startswith("Wait")
    assert steps[2].startswith("Another thought")
    assert steps[3].startswith("Let me see")
    print(f"PASS: basic splitting → {len(steps)} steps")


def test_real_trace():
    """Run detector on AIME trace 60 and print results."""
    trace_file = TRACES_DIR / "60.json"
    if not trace_file.exists():
        print(f"SKIP: {trace_file} not found")
        return

    with open(trace_file) as f:
        data = json.load(f)

    trace = data.get("reasoning") or data.get("content", "")
    steps = detect_steps(trace)

    print(f"\nAIME #60: {len(trace)} chars → {len(steps)} steps")
    print("-" * 60)
    for i, step in enumerate(steps):
        preview = step[:80].replace("\n", " ")
        print(f"  Step {i+1:3d} ({len(step):5d} chars): {preview}...")


def test_all_traces():
    """Run detector on all AIME traces, report stats."""
    if not TRACES_DIR.exists():
        print(f"SKIP: {TRACES_DIR} not found")
        return

    total_steps = 0
    total_traces = 0

    for f in sorted(TRACES_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        trace = data.get("reasoning") or data.get("content", "")
        steps = detect_steps(trace)
        total_steps += len(steps)
        total_traces += 1
        print(f"  {f.name}: {len(steps):3d} steps ({len(trace):6d} chars)")

    avg = total_steps / total_traces if total_traces else 0
    print(f"\nTotal: {total_traces} traces, {total_steps} steps, avg {avg:.1f} steps/trace")


if __name__ == "__main__":
    test_basic_splitting()
    test_real_trace()
    print("\n" + "=" * 60)
    test_all_traces()
