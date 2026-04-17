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

- [ ] Get GPU access (cloud or local)
- [ ] Set up vLLM 0.8.2 dev environment
- [ ] Study vLLM block manager: block tables, physical/logical mapping, CoW ref-counting, block allocation/free
- [ ] Identify exact files and functions to modify

## Phase 4: Core Implementation (Needs GPU)
Build the remaining MemShare components inside vLLM.

- [ ] Stage 2: KV block Euclidean distance — compare actual KV cache tensors for candidate pairs
- [ ] Block table remapping — remap matching blocks to shared physical memory, increment ref counts
- [ ] Adapt step detector to streaming mode (token-by-token in vLLM's decode loop)
- [ ] Wire full pipeline: step detect → Stage 1 → Stage 2 → remap

## Phase 5: Benchmarking
Reproduce the paper's results and measure our implementation.

- [ ] Accuracy: pass@1 on AIME 24, GPQA Diamond, MATH-500
- [ ] Throughput: tokens/sec vs full-cache baseline
- [ ] Memory: KV cache savings, affected block ratio
- [ ] Baselines: full cache, StreamingLLM, Quest, SnapKV

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