# Pull Request: [SGLang-Diffusion LLM] Add inference support for d3LLM models (arXiv:2601.07568)

## Summary

This PR adds SGLang serving support for **d3LLM** ([arXiv:2601.07568](https://arxiv.org/abs/2601.07568)), an ultra-fast diffusion language model based on pseudo-trajectory distillation. d3LLM achieves significantly higher tokens-per-forward (TPF) than vanilla diffusion LLMs while maintaining competitive accuracy, enabling up to 5× end-to-end speedup over autoregressive baselines on H100.

Two models are supported:

- **d3LLM-LLaDA** (8B) — an ultra-fast diffusion LLM, distilled from LLaDA, using full-sequence bidirectional attention
- **d3LLM-Dream** (7B) — an ultra-fast diffusion LLM, distilled from Dream, using full-sequence bidirectional attention

Both models require recomputing the full sequence at every decoding step (no causal KV-cache reuse), which demands non-trivial changes to SGLang's scheduling and attention pipeline. In addition, this PR extends support to the original **LLaDA-8B-Instruct** and **Dream-v0-Instruct-7B** models.

### Key Changes

**New Files:**
- `models/d3llm_llada.py`, `models/dream.py`: Model implementations for d3LLM-LLaDA and d3LLM-Dream
- `dllm/algorithm/entropy_threshold.py`: `EntropyThreshold` decoding algorithm
- `dllm/algorithm/full_attn_multi_block.py`: `FullAttnMultiBlock` decoding algorithm for d3LLM multi-block parallel decoding

**Modified Files:**
- `dllm/config.py`: Add `needs_full_prefill` and `pad_full_generation` flags to `DllmConfig`
- `dllm/mixin/req.py`, `dllm/mixin/scheduler.py`: Handle full-prefill mode in request lifecycle
- `flashinfer_backend.py`: Zero out `prefix_lens` when `needs_full_prefill` is enabled
- `schedule_batch.py`: Skip tree-cache matching for full-prefill models; free old KV slots before re-extend
- `schedule_policy.py`: Adjust token budget and truncation for full-prefill mode
- `forward_batch_info.py`: Build positions from full `seq_len` for bidirectional attention
- `cuda_graph_runner.py`: Disable CUDA graph for variable-length full-prefill inputs; work around Blackwell (SM≥10) multi-BS capture instability
- `http_server.py`: Increase `max_new_tokens` for dLLM health checks to ensure proper warmup
- `radix_cache.py`: Add `None` guard for `node` in `inc_lock_ref` / `dec_lock_ref`

**Tests & Docs:**
- `test/registered/dllm/test_dllm_gsm8k.py`: GSM8K-CoT benchmark test for d3LLM models
- `docs/supported_models/text_generation/diffusion_language_models.md`: Updated documentation

## Benchmark Results

**Dataset:** GSM8K-CoT (zero-shot)  
**Decoding:** FullAttnMultiBlock  
**TP Size:** 1

| Model | Threshold | Batch Size | B200 TPS | H800 TPS | A800 TPS | TPF | Accuracy |
|-------|-----------|------------|----------|----------|----------|-----|----------|
| d3LLM-LLaDA (8B dense) | 0.5 | 1 | 1240.99 | 545.31 | 251.61 | 9.91 | 75.36% |
| d3LLM-LLaDA (8B dense) | 0.5 | 4 | 1310.18 | 551.87 | 249.98 | 8.56 | 75.12% |
| d3LLM-Dream (7B dense) | 0.4 | 1 | 586.77 | 280.48 | 125.57 | 4.89 | 80.89% |
| d3LLM-Dream (7B dense) | 0.4 | 4 | 676.81 | 281.82 | 127.85 | 4.22 | 80.76% |

> **TPS** = Tokens Per Second, **TPF** = Tokens Per Forward (average forward passes per token)

## Usage Example

```bash
# Launch server with d3LLM-LLaDA
python -m sglang.launch_server \
    --model d3LLM/d3LLM_LLaDA \
    --trust-remote-code \
    --attention-backend flashinfer \
    --dllm-algorithm FullAttnMultiBlock \
    --mem-fraction-static 0.8 \
    --cuda-graph-max-bs 32

# Launch server with d3LLM-Dream
python -m sglang.launch_server \
    --model d3LLM/d3LLM_Dream \
    --trust-remote-code \
    --attention-backend flashinfer \
    --dllm-algorithm FullAttnMultiBlock \
    --mem-fraction-static 0.8 \
    --cuda-graph-max-bs 32
```

## Test Plan

- [x] Verified accuracy matches HuggingFace reference implementation
- [x] Tested on B200, H800, A800 GPUs
- [x] Added `test_dllm_gsm8k.py` for CI integration
- [x] Confirmed no regression on existing LLaDA 2.x models