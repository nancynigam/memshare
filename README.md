# MemShare

Implementing [MemShare](https://arxiv.org/pdf/2507.21433) — intra-request KV cache block sharing for reasoning models — as a contribution to [vLLM](https://github.com/vllm-project/vllm).

## Problem

Reasoning models (DeepSeek-R1, QwQ-32B) produce long chain-of-thought outputs with significant repetition — re-deriving calculations, re-verifying steps, re-stating known facts. This wastes KV cache memory on duplicate data.

## What MemShare Does

Detects when a reasoning model repeats itself during decoding and reuses the existing KV cache blocks instead of storing duplicates. This is a **metadata operation** on vLLM's block table — no GPU kernel changes needed.

The paper reports up to **85% throughput improvement** by freeing memory → enabling larger batch sizes → better GPU utilization.

## Pipeline

```
Token stream from decoder
    ↓
Step Boundary Detector — segments CoT into reasoning steps
    ↓
Stage 1: Cosine Similarity (CPU, cheap) — finds candidate step pairs
    ↓
Stage 2: KV Block Euclidean Distance (GPU) — confirms block-level matches
    ↓
Block Table Remap — point matching blocks to same physical memory
```

## Current Status

| Component | Status |
|-----------|--------|
| Data collection (reasoning traces) | Done — 30 AIME 2024 + 15 MATH-500 traces |
| Step boundary detector | Done — regex-based, validated on real traces |
| Stage 1: Cosine similarity | Done — 66 candidate pairs from 338 steps at threshold 0.8 |
| Stage 2: KV Euclidean distance | Needs GPU |
| Block table remapping | Needs GPU + vLLM |
| vLLM integration | Needs GPU + vLLM |
| Benchmarking | Needs GPU + vLLM |

## Project Structure

```
memshare/
├── memshare/                # Core modules
│   ├── step_detector.py     # Step boundary detection
│   └── similarity.py        # Stage 1 cosine similarity
├── tests/                   # Tests against real traces
├── data-collection/         # Trace collection pipeline + collected traces
│   ├── collect_traces.py    # DeepSeek-R1 via Together AI
│   └── traces/              # AIME 2024, MATH-500 reasoning traces
```

## Targets

- **vLLM version**: 0.8.2
- **Models**: DeepSeek-R1-Distill-Qwen-32B, QwQ-32B, Phi-4-reasoning-plus
- **Benchmarks**: AIME 24, GPQA Diamond, MATH-500
- **GPU**: NVIDIA A800 80GB (or equivalent)