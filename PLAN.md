# MemShare — Project Plan

## Phase 1: Data Collection & Validation (Done)
Validate the paper's core claim that reasoning models produce redundant steps.

- [x] Set up Together AI API for DeepSeek-R1 inference
- [x] Collect reasoning traces on AIME 2024 (30/30) and MATH-500 (15/500)
- [x] Validate paper's ~38% shareable block rate on independent data

## Phase 2: Standalone Modules (Done)
Build and test MemShare components with zero vLLM dependencies.

- [x] Step boundary detector — regex-based, 23 boundary patterns from manual trace annotation
- [x] Stage 1 cosine similarity — bag-of-tokens pairwise comparison, 66 candidates from 338 steps
- [x] Edge case tests + real trace validation

## Phase 3: GPU Setup & vLLM Internals (Next)
Get a GPU environment running and understand the vLLM code we'll be modifying.

- [x] Get GPU access — RunPod A100-SXM4-80GB
- [x] Set up vLLM 0.8.2 dev environment
- [x] Study vLLM block manager: block tables, physical/logical mapping, CoW ref-counting, block allocation/free
- [x] Identify exact files and functions to modify
  - `vllm/core/block/block_table.py` — BlockTable.fork(), .free(), .physical_block_ids
  - `vllm/core/block/prefix_caching_block.py` — _refcounter.incr(), _free_block_id(), cow_block_if_not_appendable()
  - Remap uses existing primitives: incr ref count on source, free target, reassign block_id

## Phase 4: Core Implementation (Needs GPU)
Build the remaining MemShare components inside vLLM.

- [ ] Stage 2: KV block Euclidean distance — compare actual KV cache tensors for candidate pairs
- [ ] Block table remapping — remap matching blocks to shared physical memory, increment ref counts
- [ ] Adapt step detector to streaming mode (token-by-token in vLLM's decode loop)
- [ ] Wire full pipeline: step detect → Stage 1 → Stage 2 → remap

## Phase 5: Benchmarking & Overhead Analysis
Reproduce the paper's results and verify that MemShare's overhead is acceptable.

### Accuracy & throughput
- [ ] Accuracy: pass@1 on AIME 24, GPQA Diamond, MATH-500
- [ ] Throughput: tokens/sec vs full-cache baseline
- [ ] Memory: KV cache savings, affected block ratio
- [ ] Baselines: full cache, StreamingLLM, Quest, SnapKV

### Overhead analysis (must measure)
MemShare adds per-token and per-step overhead to the decode loop. The key question: does the overhead eat into the throughput gains from freed memory?

**Estimated overhead budget** (DeepSeek-R1-Distill-Qwen-32B on A100 80GB):

| Component | When | Estimated cost | Confidence |
|-----------|------|---------------|------------|
| Buffer append + pattern check | Every token | ~nanoseconds | High |
| Stage 1 cosine similarity | Step boundary (~10-30x/request) | ~0.1ms | High |
| Stage 2 KV Euclidean distance | Stage 1 match (~5-10x/request) | **Unknown** | Need to measure |
| Block remap (metadata op) | Stage 2 confirms (~2-5x/request) | ~microseconds | High |
| Model forward pass (baseline) | Every token | ~30-50ms | Medium |

**What we need to measure in Phase 5:**
- [ ] Stage 2 latency: Euclidean distance computation across layers — does GPU-CPU sync or kernel launch overhead dominate?
- [ ] End-to-end per-token latency: with MemShare enabled vs disabled — is the overhead <1% of forward pass time?
- [ ] Step detection accuracy: are we catching enough boundaries to make meaningful memory savings?
- [ ] Net throughput impact: does freed memory (enabling larger batches) outweigh any per-token overhead?

**Break-even analysis**: If Stage 2 costs 1ms and triggers 10 times per request with 5000 tokens, that's 10ms overhead across the request vs 5000 × 35ms = 175,000ms of forward pass time — 0.006% overhead. Even if Stage 2 is 10x more expensive than estimated, the overhead should be negligible. But we need to confirm this empirically.

## Phase 6: Novel Contributions (Stretch)
Differentiate from the paper.

- [ ] V-only sharing — skip K blocks to avoid RoPE positional aliasing
- [ ] Layer-adaptive thresholds — tighter for shallow layers, looser for deep
- [ ] Ablation: which reasoning step types are safe to share vs dangerous

## Known MVP Divergences from Paper
To be addressed in Phase 4:

| Area | Our MVP | Paper | Fix |
|------|---------|-------|-----|
| Processing | Batch (complete traces) | Streaming (token-by-token) | Adapt to vLLM decode loop |
| Tokenization | Whitespace split | Model tokenizer vectors | Switch when integrated |
| Stage 2 | Not implemented | KV Euclidean distance on GPU | Phase 4 |
| Block remap | Not implemented | Block table metadata op | Phase 4 |
| Step patterns | May vary per model | Not detailed in paper | Tune per model during benchmarking |
| Threshold | 0.8 default | 0.9 recommended for safety | Tune during benchmarking |