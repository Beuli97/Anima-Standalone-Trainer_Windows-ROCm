# Optimization Log

This file tracks the TP/SP and LoRA optimizations we have applied so far.
When we add more, append a new entry here and keep the commit hash up to date.

## Current Branch
- `codex/anima-fused-qkv-tpsp`

## Applied Optimizations

### 1. Fused QKV/KV TP-SP training path
- Commit: `0eac5a5` (`add fused qkv tp-sp lora path`)
- What changed:
  - Added internal fusion for Anima attention projections before TP sharding.
  - Packed self-attention as `qkv_proj` and cross-attention as `kv_proj`.
  - Added LoRA save/load mapping so checkpoints still use standard `q_proj/k_proj/v_proj` names.
  - Added `--no_fuse_qkv` for debugging.
  - Added the TP/SP UI toggle so the option only appears in TP/SP mode.

### 2. Shared TP/SP communication for LoRA wrappers
- Commit: `1eb9246` (`trim tp-sp diagnostics and share lora comms`)
- What changed:
  - Removed per-step TP debug work from the hot path by gating it behind `--tp_debug`.
  - Sampled expensive diagnostics instead of running them every step.
  - Changed TP-aware LoRA wrappers so they share the same transformed input/output tensors as the base TP/SP layer instead of replaying extra gather/scatter work.
  - This reduced duplicated SP communication in the column-parallel and row-parallel LoRA paths.

### 3. Packed QKV LoRA matmul fusion
- Commit: `7aded71` (`fuse packed qkv lora matmuls`)
- What changed:
  - Kept the packed LoRA parameter layout unchanged for checkpoint compatibility.
  - Replaced the three separate packed `lora_down` matmuls with one fused packed down projection.
  - Replaced the three separate packed `lora_up` matmuls with one fused batched up projection.
  - Verified the fused packed path matches the old per-part math on CPU.

### 4. Replicated-context no-input-grad fast path
- Commit: `trainer repo commit; requires local companion changes in wd_parallel`
- What changed:
  - Added a detached replicated-input TP path for column-parallel layers that consume frozen context.
  - Marks only cross-attention K/V replicated-input TP layers, and only when the text encoder is frozen.
  - Updates TP-aware LoRA wrappers to honor the same skip-input-grad flag so the base path and LoRA path stay consistent.
  - Intended to remove the backward `all_reduce` on replicated frozen conditioning branches when gradient checkpointing would otherwise force those tensors to require grad.

### 5. Async cross-attention Q-gather overlap
- Commit: `uncommitted`
- What changed:
  - Added async SP gather support in `wd_parallel` via a pending-handle helper that preserves autograd.
  - Exposed `prepare_input_async()` / `forward_from_prepared_input()` on column-parallel layers so callers can launch the gather, do independent work, then finish the projection later.
  - Taught TP-aware LoRA column wrappers to participate in the same prepared-input path, so overlap does not bypass LoRA math.
  - Added a conservative `--tp_async_overlap` trainer flag that overlaps cross-attention Q gather with local K/V projection work in the main DiT and the LLM adapter.
  - Left the default path unchanged and kept self-attention / row-parallel output reductions on the synchronous reference path for now.

## Notes
- The current implementation is still compatible with existing save/load behavior.
- The log should be updated whenever we add another TP/SP, LoRA, or trainer-side optimization.
- If a change is only in the working tree and not yet committed, note it here as `uncommitted` until it is committed.
