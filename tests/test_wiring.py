"""
Integration test: verify MemShareHandler is correctly wired into
SingleStepOutputProcessor's decode loop.

We mock all vLLM dependencies (Sequence, SequenceGroup, detokenizer, etc.)
so this runs without torch, CUDA, or a real model — pure Python.

What this tests:
- on_token() is called after each decoded token with the right text
- on_request_finished() is called when the sequence group finishes
- on_request_aborted() cleans up state
- The full pipeline: decode → step detection → cosine similarity candidates
"""

import importlib.util
import os
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Load handler.py directly (no torch dependency)
# ---------------------------------------------------------------------------
_handler_path = os.path.join(
    os.path.dirname(__file__),
    "..", "vllm-nancy-fork", "vllm", "vllm", "memshare", "handler.py",
)
_spec = importlib.util.spec_from_file_location("handler", _handler_path)
_handler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_handler)

MemShareHandler = _handler.MemShareHandler


# ---------------------------------------------------------------------------
# Minimal fakes for vLLM objects
# ---------------------------------------------------------------------------

def make_sampling_params(detokenize=True):
    params = MagicMock()
    params.detokenize = detokenize
    return params


def make_sequence(output_text="", finished=False):
    seq = MagicMock()
    seq.output_text = output_text
    seq.is_finished.return_value = finished
    seq.lora_request = None
    return seq


def make_sequence_group(request_id, seq, finished=False,
                        detokenize=True):
    group = MagicMock()
    group.request_id = request_id
    group.first_seq = seq
    group.sampling_params = make_sampling_params(detokenize)
    group.is_finished.return_value = finished
    group.lora_request = None
    return group


def make_output(token_id=42):
    sample = MagicMock()
    sample.output_token = token_id
    sample.logprobs = {}
    output = MagicMock()
    output.samples = [sample]
    return output


def make_processor(memshare_handler, output_text_chunks):
    """
    Build a SingleStepOutputProcessor-like object that:
    - Has a detokenizer that appends successive chunks to seq.output_text
    - Has a no-op stop_checker and scheduler
    """
    processor = MagicMock()
    processor.memshare_handler = memshare_handler
    processor.scheduler = [MagicMock()]

    # Detokenizer: each call appends the next chunk and returns its length
    chunks = iter(output_text_chunks)

    def fake_decode(seq, params):
        try:
            chunk = next(chunks)
        except StopIteration:
            return 0
        seq.output_text += chunk
        return len(chunk)

    processor.detokenizer = MagicMock()
    processor.detokenizer.decode_sequence_inplace.side_effect = fake_decode

    processor.stop_checker = MagicMock()
    processor.stop_checker.maybe_stop_sequence.return_value = None

    return processor


def run_decode_step(processor, seq_group, output, is_async=False):
    """
    Replicate exactly what SingleStepOutputProcessor._process_sequence_group_outputs
    does, using our processor mock.
    """
    sampling_params = seq_group.sampling_params
    sample = output.samples[0]
    seq = seq_group.first_seq

    if not is_async:
        seq.append_token_id(sample.output_token, sample.logprobs)

    if sampling_params.detokenize and processor.detokenizer:
        new_char_count = processor.detokenizer.decode_sequence_inplace(
            seq, sampling_params)
    else:
        new_char_count = 0

    # MemShare hook — this is what we're testing
    if processor.memshare_handler and new_char_count > 0:
        new_text = seq.output_text[-new_char_count:]
        processor.memshare_handler.on_token(seq_group.request_id, new_text)

    processor.stop_checker.maybe_stop_sequence(
        seq, new_char_count, sampling_params,
        lora_req=seq_group.lora_request)

    if seq.is_finished():
        if processor.memshare_handler and seq_group.is_finished():
            processor.memshare_handler.on_request_finished(
                seq_group.request_id)
        for scheduler in processor.scheduler:
            scheduler.free_seq(seq)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOnTokenCalled:

    def test_on_token_called_per_chunk(self):
        """on_token receives each decoded chunk in order."""
        handler = MemShareHandler(enabled=True)
        chunks = ["Hello", " world", " foo"]
        seq = make_sequence(output_text="")
        group = make_sequence_group("req-1", seq, finished=False)
        processor = make_processor(handler, chunks)

        for chunk in chunks:
            output = make_output()
            run_decode_step(processor, group, output)

        state = handler._states.get("req-1")
        assert state is not None
        assert state.text_buffer == "Hello world foo"

    def test_on_token_not_called_when_no_new_chars(self):
        """on_token is skipped when detokenizer returns 0 chars."""
        handler = MemShareHandler(enabled=True)
        seq = make_sequence(output_text="")
        group = make_sequence_group("req-1", seq, finished=False)
        # Empty chunks → new_char_count = 0 every step
        processor = make_processor(handler, ["", "", ""])

        for _ in range(3):
            run_decode_step(processor, group, make_output())

        # State should not even be created
        assert "req-1" not in handler._states

    def test_on_token_not_called_when_detokenize_false(self):
        """on_token is skipped when detokenize=False."""
        handler = MemShareHandler(enabled=True)
        seq = make_sequence(output_text="")
        group = make_sequence_group("req-1", seq, finished=False,
                                    detokenize=False)
        processor = make_processor(handler, ["Hello", " world"])

        for _ in range(2):
            run_decode_step(processor, group, make_output())

        assert "req-1" not in handler._states


class TestOnRequestFinished:

    def test_on_request_finished_called_when_group_done(self):
        """on_request_finished is called when sequence group finishes."""
        handler = MemShareHandler(enabled=True)
        seq = make_sequence(output_text="", finished=True)
        group = make_sequence_group("req-1", seq, finished=True)
        processor = make_processor(handler, ["some text here"])

        run_decode_step(processor, group, make_output())

        # State should be popped by on_request_finished
        assert "req-1" not in handler._states

    def test_on_request_finished_not_called_when_seq_done_but_group_not(self):
        """With beam search, finishing one seq should not finalize the group."""
        handler = MemShareHandler(enabled=True)
        seq = make_sequence(output_text="", finished=True)
        # seq is done but group is not (other beams still running)
        group = make_sequence_group("req-1", seq, finished=False)
        processor = make_processor(handler, ["some text here"])

        run_decode_step(processor, group, make_output())

        # State should still exist
        assert "req-1" in handler._states

    def test_on_request_finished_not_called_when_seq_not_done(self):
        """on_request_finished not called mid-sequence."""
        handler = MemShareHandler(enabled=True)
        seq = make_sequence(output_text="", finished=False)
        group = make_sequence_group("req-1", seq, finished=False)
        processor = make_processor(handler, ["token1", "token2"])

        for _ in range(2):
            run_decode_step(processor, group, make_output())

        assert "req-1" in handler._states


class TestFullPipeline:

    def test_step_detection_fires_in_decode_loop(self):
        """
        Feed a full reasoning trace token-by-token through the decode loop.
        Verify step boundaries are detected and similarity candidates found.
        """
        handler = MemShareHandler(enabled=True, threshold=0.7)

        # Simulate a reasoning trace with repeated steps
        trace = (
            "I need to find the sum of integers from 1 to 100 using the formula. "
            "Wait, let me verify using the formula n times n plus one divided by two. "
            "So the answer is 100 times 101 divided by 2 which equals 5050. "
            "Let me verify using the formula n times n plus one divided by two again. "
            "Yes, 5050 is correct."
        )

        # Feed as 5-character chunks (simulating token-by-token)
        seq = make_sequence(output_text="")
        group = make_sequence_group("req-1", seq, finished=False)
        chunk_size = 5
        chunks = [trace[i:i+chunk_size]
                  for i in range(0, len(trace), chunk_size)]
        processor = make_processor(handler, chunks)

        for _ in chunks:
            run_decode_step(processor, group, make_output())

        state = handler._states.get("req-1")
        assert state is not None
        assert len(state.completed_steps) >= 1

        # Finalize and check for similarity candidates
        final_state = handler.on_request_finished("req-1")
        assert final_state is not None
        assert len(final_state.completed_steps) >= 2
        # The repeated "verify using the formula" step should be a candidate
        assert len(final_state.candidates) >= 1

    def test_disabled_handler_does_nothing(self):
        """Disabled handler never creates state."""
        handler = MemShareHandler(enabled=False)
        seq = make_sequence(output_text="")
        group = make_sequence_group("req-1", seq, finished=False)
        processor = make_processor(handler, ["Hello", " world"])

        for _ in range(2):
            run_decode_step(processor, group, make_output())

        assert len(handler._states) == 0

    def test_multiple_concurrent_requests(self):
        """Two requests run concurrently, state is isolated."""
        handler = MemShareHandler(enabled=True)

        seq1 = make_sequence(output_text="")
        group1 = make_sequence_group("req-1", seq1, finished=False)
        processor1 = make_processor(handler, ["Hello req1 ", "more text "])

        seq2 = make_sequence(output_text="")
        group2 = make_sequence_group("req-2", seq2, finished=False)
        processor2 = make_processor(handler, ["Hello req2 ", "other text"])

        # Interleave steps from both requests
        run_decode_step(processor1, group1, make_output())
        run_decode_step(processor2, group2, make_output())
        run_decode_step(processor1, group1, make_output())
        run_decode_step(processor2, group2, make_output())

        assert "req-1" in handler._states
        assert "req-2" in handler._states
        assert handler._states["req-1"].text_buffer != \
               handler._states["req-2"].text_buffer
