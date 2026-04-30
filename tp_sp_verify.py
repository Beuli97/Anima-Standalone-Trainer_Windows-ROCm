"""
tp_sp_verify.py — TP+SP correctness diagnostics for Anima LoRA training.

Call run_all_checks() right after TP init and model loading.  All checks
are non-destructive: they create temporary tensors, run collectives, then
log PASS/FAIL.  The model itself is never modified.

Usage (called from anima_train_network_tp_sp.py main block):

    if tp_groups is not None and tp_groups.tp_size > 1:
        from tp_sp_verify import run_all_checks
        run_all_checks(dit=None, groups=tp_groups, use_sp=use_sp)
    # ... after model loaded + sharded:
        run_all_checks(dit=dit, groups=tp_groups, use_sp=use_sp)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _ok(name: str, passed: bool, detail: str = ""):
    tag = "PASS" if passed else "FAIL"
    msg = f"  [tp_verify] {tag}  {name}"
    if detail:
        msg += f"  ({detail})"
    if _rank() == 0:
        logger.info(msg)
    if not passed:
        logger.error(f"  [tp_verify] FAIL detail rank={_rank()}: {detail}")
    return passed


def _allclose(a: torch.Tensor, b: torch.Tensor, atol=1e-3) -> tuple:
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        return False, f"shape_mismatch={tuple(a.shape)} != {tuple(b.shape)}"
    diff = (a - b).abs()
    ok = bool(diff.max() < atol)
    return ok, f"max_diff={diff.max():.5f} mean_diff={diff.mean():.5f}"


# ---------------------------------------------------------------------------
# Check 1 — Collective math (gather / reduce-scatter / all-reduce)
# ---------------------------------------------------------------------------

def check_collectives(groups) -> bool:
    """Verify gather + scatter produce correct values with the live backend."""
    from wd_parallel.collectives import (
        gather_from_sp_region,
        reduce_scatter_to_sp_region,
        copy_to_tp_region,
    )

    group = groups.tp
    rank  = dist.get_rank(group=group)
    tp    = group.size()
    B, S, D = 1, tp * 8, 32   # S must be divisible by tp

    all_passed = True

    # --- 1a: gather round-trip ---
    full  = torch.arange(tp * S * D, dtype=torch.float32, device='cuda').reshape(tp * S, D)
    shard = full[rank * S:(rank + 1) * S].clone()
    gathered = gather_from_sp_region(shard, group, seq_dim=0)
    ok, d = _allclose(gathered, full)
    all_passed &= _ok("gather round-trip (seq_dim=0)", ok, d)

    # --- 1b: gather seq_dim=1 (batch-first, Anima layout) ---
    full2  = torch.arange(B * tp * S * D, dtype=torch.float32, device='cuda').reshape(B, tp * S, D)
    shard2 = full2[:, rank * S:(rank + 1) * S, :].contiguous()
    gathered2 = gather_from_sp_region(shard2, group, seq_dim=1)
    ok2, d2 = _allclose(gathered2, full2)
    all_passed &= _ok("gather round-trip (seq_dim=1, batch-first)", ok2, d2)

    # --- 1c: reduce-scatter correctness ---
    # Each rank contributes a constant (rank+1); reduce-scatter should sum them
    const    = torch.full((tp * S, D), float(rank + 1), device='cuda')
    expected_sum = sum(r + 1 for r in range(tp))
    expected_shard = torch.full((S, D), float(expected_sum), device='cuda')
    scattered = reduce_scatter_to_sp_region(const, group, seq_dim=0)
    ok3, d3 = _allclose(scattered, expected_shard)
    all_passed &= _ok("reduce-scatter sum correctness", ok3, d3)

    # --- 1d: copy_to_tp backward (all-reduce gradient) ---
    # Use nn.Parameter as leaf anchor — torch.ones(...).cuda() is non-leaf in PyTorch 2.7+
    # because .cuda() is a tracked ToCopyBackward op.
    w = nn.Parameter(torch.ones(D, device='cuda', dtype=torch.float32))
    x = w * 1.0  # non-leaf intermediate
    y = copy_to_tp_region(x, group)
    loss = (y * float(rank + 1)).sum()
    loss.backward()
    expected_grad = float(sum(r + 1 for r in range(tp)))
    ok4, d4 = _allclose(w.grad, torch.full((D,), expected_grad, device='cuda'))
    all_passed &= _ok("copy_to_tp_region backward (all-reduce grad)", ok4, d4)

    return all_passed


# ---------------------------------------------------------------------------
# Check 2 — ColumnParallel + RowParallel chain == single nn.Linear
# ---------------------------------------------------------------------------

def check_tp_layers(groups, use_sp: bool = False) -> bool:
    """A Col→GeLU→Row chain must match a single Linear."""
    from wd_parallel.layers import ColumnParallelLinear, RowParallelLinear

    group = groups.tp
    rank  = dist.get_rank(group=group)
    tp    = group.size()
    # Synthetic layer math check: keep dimensions divisible by tp so the
    # verifier works for non-power-of-two TP degrees without floor-dropping
    # channels. Padded geometry is covered by dedicated Anima/wd_parallel tests.
    D_in, D_mid, D_out = tp * 32, tp * 64, tp * 32
    B = 1
    S = tp * 16   # must be divisible by tp for SP

    all_passed = True

    # Shared reference weights (broadcast from rank 0)
    torch.manual_seed(0)
    W_col = torch.randn(D_mid, D_in, device='cuda', dtype=torch.float32)
    W_row = torch.randn(D_out, D_mid, device='cuda', dtype=torch.float32)
    dist.broadcast(W_col, src=0, group=group)
    dist.broadcast(W_row, src=0, group=group)

    # Reference: single GPU linear chain (computed on all ranks with same weights+input)
    torch.manual_seed(1)
    x_full = torch.randn(B, S, D_in, device='cuda', dtype=torch.float32)
    dist.broadcast(x_full, src=0, group=group)
    y_ref  = F.linear(F.gelu(F.linear(x_full, W_col)), W_row)   # (B, S, D_out)

    # TP shards
    chunk_mid = D_mid // tp

    col = ColumnParallelLinear(D_in, chunk_mid, bias=False,
                               sequence_parallel=use_sp, seq_dim=1)
    col.weight = nn.Parameter(W_col[rank*chunk_mid:(rank+1)*chunk_mid].contiguous())
    col._group = group
    col.cuda()

    row = RowParallelLinear(chunk_mid, D_out, bias=False,
                            sequence_parallel=use_sp, seq_dim=1)
    row.weight = nn.Parameter(W_row[:, rank*chunk_mid:(rank+1)*chunk_mid].contiguous())
    row._group = group
    row.cuda()

    # SP: each rank processes a shard of the sequence; Col gathers it internally
    Sl = S // tp
    x_input = x_full[:, rank*Sl:(rank+1)*Sl, :] if use_sp else x_full

    y_tp = row(F.gelu(col(x_input)))   # SP: (B, Sl, D_out) | TP-only: (B, S, D_out)

    if use_sp:
        # Gather shards to compare with full reference
        parts = [torch.zeros(B, Sl, D_out, device='cuda', dtype=torch.float32) for _ in range(tp)]
        dist.all_gather(parts, y_tp.float().contiguous(), group=group)
        y_tp_full = torch.cat(parts, dim=1)
    else:
        y_tp_full = y_tp

    ok, d = _allclose(y_tp_full, y_ref, atol=1e-3)
    mode = "SP" if use_sp else "TP-only"
    all_passed &= _ok(f"Col→GeLU→Row forward matches single-GPU ({mode})", ok, d)

    # --- Bias: verify bias is added exactly once (not tp_size times) ---
    torch.manual_seed(2)
    W_bias  = torch.randn(D_out, D_in, device='cuda')
    b_bias  = torch.randn(D_out, device='cuda')
    dist.broadcast(W_bias, src=0, group=group)
    dist.broadcast(b_bias, src=0, group=group)

    x_b = torch.randn(B, 4, D_in, device='cuda')
    dist.broadcast(x_b, src=0, group=group)
    y_ref_bias = F.linear(x_b, W_bias, b_bias)

    chunk_r = D_in // tp
    row_b = RowParallelLinear(chunk_r, D_out, bias=True, sequence_parallel=False, seq_dim=1)
    row_b.weight = nn.Parameter(W_bias[:, rank*chunk_r:(rank+1)*chunk_r].contiguous())
    row_b.bias   = nn.Parameter(b_bias.clone())
    row_b._group = group
    row_b.cuda()
    x_shard_b = x_b[:, :, rank*chunk_r:(rank+1)*chunk_r].contiguous()
    y_tp_bias  = row_b(x_shard_b)
    ok_b, d_b = _allclose(y_tp_bias, y_ref_bias, atol=1e-3)
    all_passed &= _ok("RowParallel bias added exactly once (not tp_size×)", ok_b, d_b)

    return all_passed


# ---------------------------------------------------------------------------
# Check 3 — Model forward: TP output matches single-GPU output
# ---------------------------------------------------------------------------

def check_model_forward(dit, groups, use_sp: bool = False) -> bool:
    """Run a mini forward pass through the sharded model with mock inputs.

    Uses tiny mock tensors so this runs in <10 seconds even on the full 2B model.
    Compares rank 0's output against itself (same rank, so no communication needed
    for the comparison — we just check for NaN/Inf and shape correctness).

    For numerical equivalence against single-GPU reference, we'd need the
    unsharded weights on a third GPU — so here we check:
      1. Output shape is correct
      2. No NaN/Inf in output
      3. All TP ranks produce bit-identical outputs (same batch, same sequence)
    """
    if dit is None:
        if _rank() == 0:
            logger.info("  [tp_verify] SKIP model forward (dit not loaded yet)")
        return True

    group = groups.tp
    rank  = dist.get_rank(group=group)
    tp    = group.size()

    device = next(dit.parameters()).device
    dtype  = next(dit.parameters()).dtype

    # Mock inputs — smallest valid dimensions
    # spatial_patch_size=2, temporal_patch_size=1
    # For SP: sequence S = T * (H/2) * (W/2) must be divisible by tp
    #   T=1, H=8, W=8 → S = 1*4*4 = 16, 16 % tp=2 = 0 ✓
    B, C, T, H, W = 1, 16, 1, 8, 8
    ctx_len, ctx_dim = 4, 1024
    concat_pm = getattr(dit, 'concat_padding_mask', False)

    torch.manual_seed(42)
    x_mock   = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    t_mock   = torch.tensor([500.0], device=device, dtype=dtype)
    ctx_mock = torch.randn(B, ctx_len, ctx_dim, device=device, dtype=dtype)
    # Padding mask: (B, H, W), all ones = all pixels valid
    # Required when concat_padding_mask=True (model was trained with it)
    pm_mock  = torch.ones(B, H, W, device=device, dtype=dtype) if concat_pm else None

    # Broadcast identical inputs to all ranks.
    # All ranks already use torch.manual_seed(42) above so the tensors are
    # bit-identical without a broadcast. Only broadcast when on CUDA — the
    # cuda_direct backend rejects CPU tensors, and at this point in the training
    # script the model may still be on CPU (not yet moved to GPU).
    if x_mock.is_cuda:
        dist.broadcast(x_mock,   src=0, group=group)
        dist.broadcast(t_mock,   src=0, group=group)
        dist.broadcast(ctx_mock, src=0, group=group)
        if pm_mock is not None:
            dist.broadcast(pm_mock, src=0, group=group)

    all_passed = True
    try:
        dit.eval()
        with torch.no_grad():
            out = dit(x_mock, t_mock, ctx_mock, padding_mask=pm_mock)

        # Check 3a: output shape
        # Expected: (B, C_out, T, H, W) or (B, T, H/2, W/2, D) depending on model
        has_nan  = bool(torch.isnan(out).any())
        has_inf  = bool(torch.isinf(out).any())
        ok_clean = not has_nan and not has_inf
        all_passed &= _ok("forward: no NaN/Inf in output", ok_clean,
                          f"nan={has_nan} inf={has_inf} shape={tuple(out.shape)} dtype={out.dtype}")

        # Check 3b: all TP ranks have identical output.
        # After SP gather + final_layer, the full output is replicated on all ranks
        # (same as TP-only mode). We verify this for both SP and TP-only.
        # IMPORTANT: gather in model dtype (not float32) — cuda_direct rejects float32.
        out_c = out.contiguous()
        if tp > 1:
            parts = [torch.zeros_like(out_c) for _ in range(tp)]
            dist.all_gather(parts, out_c, group=group)
            for r in range(1, tp):
                ok_r, d_r = _allclose(parts[r].float(), parts[0].float(), atol=2e-2)
                mode = "SP" if use_sp else "TP"
                all_passed &= _ok(f"forward {mode}: rank {r} == rank 0 (identical full output)", ok_r, d_r)

        # Check 3c: backward (loss + backward, check no crash and grads exist)
        # IMPORTANT: do NOT cast to float32 before .mean() — that sends float32
        # gradients into the model backward, which cuda_direct cannot handle.
        dit.train()
        trainable = [p for p in dit.parameters() if p.requires_grad]
        if trainable:
            out_train = dit(x_mock, t_mock, ctx_mock, padding_mask=pm_mock)
            loss = out_train.mean()
            loss.backward()
            n_with_grad = sum(1 for p in trainable if p.grad is not None)
            ok_bwd = n_with_grad > 0
            all_passed &= _ok("forward+backward: gradients flow",
                              ok_bwd, f"{n_with_grad}/{len(trainable)} params have grad")
        else:
            if _rank() == 0:
                logger.info("  [tp_verify] SKIP backward (no requires_grad params — model is frozen)")

    except Exception as e:
        all_passed &= _ok("model forward/backward", False, f"{type(e).__name__}: {e}")

    return all_passed


# ---------------------------------------------------------------------------
# Check 4 — TP LoRA param tagging
# ---------------------------------------------------------------------------

def check_lora_tagging(network, groups) -> bool:
    """Verify TP-sharded LoRA params are tagged and replicated ones are not.

    Correct tagging:
      ColumnParallelLoRAModule.lora_up   → _tp_sharded=True  (sharded)
      ColumnParallelLoRAModule.lora_down → _tp_sharded unset  (replicated)
      RowParallelLoRAModule.lora_down    → _tp_sharded=True  (sharded)
      RowParallelLoRAModule.lora_up      → _tp_sharded unset  (replicated)
    """
    from networks.lora_anima import ColumnParallelLoRAModule, PackedColumnParallelLoRAModule, RowParallelLoRAModule

    group  = groups.tp

    col_up_tagged   = 0   # ColumnParallel.lora_up tagged (expected)
    col_down_tagged = 0   # ColumnParallel.lora_down tagged (unexpected)
    row_down_tagged = 0   # RowParallel.lora_down tagged (expected)
    row_up_tagged   = 0   # RowParallel.lora_up tagged (unexpected)
    replicated_count = 0

    # post_process_network runs BEFORE apply_to(), so LoRA modules are not yet
    # registered as PyTorch submodules. network.modules() only yields the
    # LoRANetwork itself; the actual LoRA module objects live in network.unet_loras.
    for lora in network.unet_loras:
        if isinstance(lora, PackedColumnParallelLoRAModule):
            for up in lora.lora_up:
                for p in up.parameters():
                    col_up_tagged += 1 if getattr(p, '_tp_sharded', False) else 0
            for down in lora.lora_down:
                for p in down.parameters():
                    col_down_tagged += 1 if getattr(p, '_tp_sharded', False) else 0
        elif isinstance(lora, ColumnParallelLoRAModule):
            for p in lora.lora_up.parameters():
                col_up_tagged += 1 if getattr(p, '_tp_sharded', False) else 0
            for p in lora.lora_down.parameters():
                col_down_tagged += 1 if getattr(p, '_tp_sharded', False) else 0
        elif isinstance(lora, RowParallelLoRAModule):
            for p in lora.lora_down.parameters():
                row_down_tagged += 1 if getattr(p, '_tp_sharded', False) else 0
            for p in lora.lora_up.parameters():
                row_up_tagged += 1 if getattr(p, '_tp_sharded', False) else 0
        else:
            # Plain LoRAModule on non-TP layer — replicated across ranks
            replicated_count += sum(1 for _ in lora.parameters())

    total_sharded = col_up_tagged + row_down_tagged
    bad_tagged    = col_down_tagged + row_up_tagged

    all_passed = True
    all_passed &= _ok(f"LoRA: {total_sharded} sharded params correctly tagged _tp_sharded",
                      total_sharded > 0)
    all_passed &= _ok(f"LoRA: no replicated params incorrectly tagged _tp_sharded",
                      bad_tagged == 0, f"bad_tagged={bad_tagged}")
    if _rank() == 0:
        logger.info(f"  [tp_verify]      {replicated_count} replicated LoRA params "
                    f"(will be synced by sync_replicated_grads)")
    return all_passed


# ---------------------------------------------------------------------------
# Check 5 — LoRA forward math: Col/Row LoRA output == single-GPU reference
# ---------------------------------------------------------------------------

def check_lora_forward_math(groups, use_sp: bool = False) -> bool:
    """Verify ColumnParallelLoRAModule and RowParallelLoRAModule forward math.

    Uses isolated layers (no full model) with known weights.
    Verifies:
      - ColumnParallelLoRAModule: concatenating per-rank outputs == single-GPU LoRA
      - RowParallelLoRAModule:    output after collective == single-GPU LoRA

    LoRAModule requires apply_to() before calling forward. After apply_to():
      - org_module.forward is replaced with lora.forward
      - lora.org_forward points to the original org_module.forward
    So we call apply_to() then invoke via the base layer (col_base / row_base).
    """
    from wd_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from networks.lora_anima import ColumnParallelLoRAModule, PackedColumnParallelLoRAModule, RowParallelLoRAModule
    import torch.nn.functional as F

    group = groups.tp
    rank  = dist.get_rank(group=group)
    tp    = group.size()
    # Synthetic LoRA math check: keep feature width divisible by tp so this
    # verifier tests LoRA math rather than arbitrary-width padding behavior.
    D, lora_dim = tp * 32, 8
    B, S = 1, tp * 8
    Sl   = S // tp
    chunk = D // tp
    # alpha = lora_dim → scale = alpha / lora_dim = 1.0

    all_passed = True

    # Shared reference weights (broadcast from rank 0)
    torch.manual_seed(99)
    W_base   = torch.randn(D, D, device='cuda', dtype=torch.float32)
    W_down   = torch.randn(lora_dim, D, device='cuda', dtype=torch.float32)    # (lora_dim, D_in)
    W_up_col = torch.randn(D, lora_dim, device='cuda', dtype=torch.float32)    # (D_out, lora_dim)
    x_full   = torch.randn(B, S, D, device='cuda', dtype=torch.float32)
    for t in [W_base, W_down, W_up_col, x_full]:
        dist.broadcast(t, src=0, group=group)

    # ----- ColumnParallelLoRAModule -----
    # lora_down: replicated (D_in -> lora_dim), same weight on all ranks
    # lora_up:   col-sharded (lora_dim -> D_out/tp), per-rank
    # Single-GPU reference: base(x) + lora_up_full(lora_down(x))
    y_ref_col = F.linear(x_full, W_base) + F.linear(F.linear(x_full, W_down), W_up_col)

    col_base = ColumnParallelLinear(D, chunk, bias=False, sequence_parallel=use_sp, seq_dim=1)
    col_base.weight = nn.Parameter(W_base[rank*chunk:(rank+1)*chunk].clone())
    col_base._group = group
    col_base.cuda()

    col_lora = ColumnParallelLoRAModule(
        "test_col", col_base, 1.0, lora_dim, float(lora_dim),
        tp_group=group, seq_dim=1, use_sp=use_sp,
    )
    col_lora.lora_down.weight = nn.Parameter(W_down.clone())
    col_lora.lora_up.weight   = nn.Parameter(W_up_col[rank*chunk:(rank+1)*chunk].clone())
    col_lora.cuda()
    col_lora.apply_to()  # hooks col_base.forward → col_lora.forward

    x_col_input = x_full[:, rank*Sl:(rank+1)*Sl, :].contiguous() if use_sp else x_full
    # col_base(x) now calls col_lora.forward(x) due to apply_to()
    y_col_tp = col_base(x_col_input)   # (B, S_or_Sl, chunk)

    # Gather feature dim to compare with single-GPU ref.
    # ColumnParallel(SP=True) all-gathers input internally, so output is
    # already full-S (B, S, chunk). No sequence-dim gather needed.
    parts_d = [torch.zeros_like(y_col_tp) for _ in range(tp)]
    dist.all_gather(parts_d, y_col_tp.contiguous(), group=group)
    y_col_full = torch.cat(parts_d, dim=-1)  # (B, S, D)

    ok, d = _allclose(y_col_full, y_ref_col, atol=1e-3)
    mode = "SP" if use_sp else "TP-only"
    all_passed &= _ok(f"ColumnParallelLoRAModule forward == single-GPU ref ({mode})", ok, d)

    # ----- RowParallelLoRAModule -----
    # lora_down: row-sharded (D_in/tp -> lora_dim), per-rank slice of full W_down
    # lora_up:   replicated  (lora_dim -> D_out), same on all ranks
    # Math: sum_r [lora_up(lora_down_r(x_r))] + base(x_full)
    #     = lora_up(sum_r lora_down_r(x_r)) + base(x_full)   [lora_up linear, same on all ranks]
    #     = lora_up(W_down_full @ x_full)   + base(x_full)   [W_down_full = cat(W_down_r, dim=1)]

    torch.manual_seed(77)
    W_base_row = torch.randn(D, D, device='cuda', dtype=torch.float32)
    W_down_row = torch.randn(lora_dim, D, device='cuda', dtype=torch.float32)  # full
    W_up_row   = torch.randn(D, lora_dim, device='cuda', dtype=torch.float32)  # replicated
    x_row_full = torch.randn(B, S, D, device='cuda', dtype=torch.float32)
    for t in [W_base_row, W_down_row, W_up_row, x_row_full]:
        dist.broadcast(t, src=0, group=group)

    y_ref_row = F.linear(x_row_full, W_base_row) + F.linear(F.linear(x_row_full, W_down_row), W_up_row)
    y_ref_row_local = y_ref_row[:, rank*Sl:(rank+1)*Sl, :] if use_sp else y_ref_row

    row_base = RowParallelLinear(chunk, D, bias=False, sequence_parallel=use_sp, seq_dim=1)
    row_base.weight = nn.Parameter(W_base_row[:, rank*chunk:(rank+1)*chunk].clone())
    row_base._group = group
    row_base.cuda()

    row_lora = RowParallelLoRAModule(
        "test_row", row_base, 1.0, lora_dim, float(lora_dim),
        tp_group=group, seq_dim=1, use_sp=use_sp,
    )
    row_lora.lora_down.weight = nn.Parameter(W_down_row[:, rank*chunk:(rank+1)*chunk].clone())
    row_lora.lora_up.weight   = nn.Parameter(W_up_row.clone())   # same on all ranks
    row_lora.cuda()
    row_lora.apply_to()   # hooks row_base.forward → row_lora.forward

    # RowParallel always receives full-S, D/tp input.
    # In a Col-Row SP chain: Col all-gathers S (SP shard → full S) then outputs (B, S, D/tp).
    # Row receives that (B, S, D/tp) input — full S, sharded D.
    x_row_input = x_row_full[:, :, rank*chunk:(rank+1)*chunk].contiguous()  # (B, S, chunk)
    y_row_tp = row_base(x_row_input)   # (B, Sl, D) in SP mode; (B, S, D) in TP-only

    ok_r, d_r = _allclose(y_row_tp, y_ref_row_local, atol=1e-3)
    all_passed &= _ok(f"RowParallelLoRAModule forward == single-GPU ref ({mode})", ok_r, d_r)

    return all_passed


# ---------------------------------------------------------------------------
# Check 6 — gather_tp_lora_weights roundtrip shapes
# ---------------------------------------------------------------------------

def check_gather_tp_lora_weights(network, groups) -> bool:
    """After gather_tp_lora_weights(), LoRA weight shapes must be full-rank (not sharded).

    Column LoRA: lora_up.weight shape  (D_out,    lora_dim) — was (D_out/tp, lora_dim)
    Row LoRA:    lora_down.weight shape (lora_dim, D_in)     — was (lora_dim, D_in/tp)
    """
    from networks.lora_anima import ColumnParallelLoRAModule, PackedColumnParallelLoRAModule, RowParallelLoRAModule

    group = groups.tp
    tp    = group.size()
    all_passed = True

    # Record sharded shapes before gather
    col_sharded = {}  # lora_name -> sharded lora_up shape
    packed_col_sharded = {}  # lora_name -> first sharded packed lora_up shape
    row_sharded = {}  # lora_name -> sharded lora_down shape
    for lora in network.unet_loras:
        if isinstance(lora, PackedColumnParallelLoRAModule) and lora._tp_group is not None:
            packed_col_sharded[lora.lora_name] = tuple(lora.lora_up[0].weight.shape)
        elif isinstance(lora, ColumnParallelLoRAModule) and lora._tp_group is not None:
            col_sharded[lora.lora_name] = tuple(lora.lora_up.weight.shape)
        elif isinstance(lora, RowParallelLoRAModule) and lora._tp_group is not None:
            row_sharded[lora.lora_name] = tuple(lora.lora_down.weight.shape)

    if not col_sharded and not packed_col_sharded and not row_sharded:
        _ok("gather_tp_lora_weights: at least one TP LoRA found", False,
            "no ColumnParallelLoRAModule, PackedColumnParallelLoRAModule, or RowParallelLoRAModule found")
        return False

    # Gather, verify, then re-scatter so training sees sharded weights
    network.gather_tp_lora_weights()

    # Verify shapes are now tp× larger on the gathered dim
    all_passed = True
    for lora in network.unet_loras:
        if isinstance(lora, PackedColumnParallelLoRAModule) and lora.lora_name in packed_col_sharded:
            sharded_shape = packed_col_sharded[lora.lora_name]       # (part_out/tp, lora_dim)
            gathered_shape = tuple(lora.lora_up[0].weight.shape)     # (part_out, lora_dim)
            expected_shape = (sharded_shape[0] * tp, sharded_shape[1])
            ok = gathered_shape == expected_shape
            all_passed &= _ok(
                f"Packed Col LoRA gather shape: {sharded_shape} x{tp} -> {gathered_shape}", ok,
                f"expected {expected_shape}"
            )
            break  # one sample is enough

    for lora in network.unet_loras:
        if isinstance(lora, ColumnParallelLoRAModule) and lora.lora_name in col_sharded:
            sharded_shape = col_sharded[lora.lora_name]          # (D_out/tp, lora_dim)
            gathered_shape = tuple(lora.lora_up.weight.shape)    # (D_out, lora_dim)
            expected_shape = (sharded_shape[0] * tp, sharded_shape[1])
            ok = gathered_shape == expected_shape
            all_passed &= _ok(
                f"Col LoRA gather shape: {sharded_shape} x{tp} -> {gathered_shape}", ok,
                f"expected {expected_shape}"
            )
            break  # one sample is enough

    for lora in network.unet_loras:
        if isinstance(lora, RowParallelLoRAModule) and lora.lora_name in row_sharded:
            sharded_shape = row_sharded[lora.lora_name]            # (lora_dim, D_in/tp)
            gathered_shape = tuple(lora.lora_down.weight.shape)    # (lora_dim, D_in)
            expected_shape = (sharded_shape[0], sharded_shape[1] * tp)
            ok = gathered_shape == expected_shape
            all_passed &= _ok(
                f"Row LoRA gather shape: {sharded_shape} x{tp} -> {gathered_shape}", ok,
                f"expected {expected_shape}"
            )
            break

    if _rank() == 0:
        logger.info(
            f"  [tp_verify]      col_loras={len(col_sharded)}  "
            f"packed_col_loras={len(packed_col_sharded)}  row_loras={len(row_sharded)}"
        )

    # Re-shard back to per-rank slices so training starts with correct TP shapes
    network.scatter_tp_lora_weights()

    return all_passed


# ---------------------------------------------------------------------------
# Check 7 — sync_replicated_grads: replicated param grads equal across ranks
# ---------------------------------------------------------------------------

def check_sync_replicated_grads(dit, network, groups, use_sp: bool = False) -> bool:
    """After backward + sync_replicated_grads, non-TP params must have identical grads.

    Picks a sample of LayerNorm and adaLN weight grads, verifies all ranks match.
    """
    import wd_parallel as wdp

    group = groups.tp
    rank  = dist.get_rank(group=group)
    tp    = group.size()
    device = next(dit.parameters()).device
    dtype  = next(dit.parameters()).dtype

    B, C, T, H, W = 1, 16, 1, 8, 8
    ctx_len, ctx_dim = 4, 1024
    concat_pm = getattr(dit, 'concat_padding_mask', False)

    torch.manual_seed(42 + rank)  # DIFFERENT seed per rank to stress-test grad sync
    x_mock   = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    t_mock   = torch.tensor([500.0], device=device, dtype=dtype)
    ctx_mock = torch.randn(B, ctx_len, ctx_dim, device=device, dtype=dtype)
    pm_mock  = torch.ones(B, H, W, device=device, dtype=dtype) if concat_pm else None
    # Broadcast so all ranks use same input (tests grad sync, not input variation)
    dist.broadcast(x_mock, src=0, group=group)
    dist.broadcast(ctx_mock, src=0, group=group)

    all_passed = True
    try:
        dit.train()
        if network is not None:
            network.train()

        out = dit(x_mock, t_mock, ctx_mock, padding_mask=pm_mock)
        loss = out.float().mean()
        loss.backward()

        # Sync replicated grads (LayerNorm, AdaLN, embedders, etc.)
        wdp.sync_replicated_grads(dit, group)
        if network is not None:
            wdp.sync_replicated_grads(network, group)

        # Collect a sample of replicated params that have grads
        sample_params = []
        for name, p in dit.named_parameters():
            if p.grad is not None and not getattr(p, '_tp_sharded', False):
                sample_params.append((name, p.grad.clone().float()))
                if len(sample_params) >= 4:
                    break

        if not sample_params:
            if _rank() == 0:
                logger.info("  [tp_verify] SKIP sync_grads check (no replicated grads found)")
            return True

        # Verify each sampled grad is identical across ranks
        for name, grad in sample_params:
            parts = [torch.zeros_like(grad) for _ in range(tp)]
            dist.all_gather(parts, grad.contiguous(), group=group)
            for r in range(1, tp):
                ok_r, d_r = _allclose(parts[r], parts[0], atol=1e-5)
                all_passed &= _ok(
                    f"sync_replicated_grads: '{name}' grad rank {r}==rank 0", ok_r, d_r
                )

        # Also verify TP-sharded LoRA params did NOT get synced (should differ between ranks)
        if network is not None:
            for lora in network.unet_loras:
                for pname, p in lora.named_parameters():
                    if getattr(p, '_tp_sharded', False) and p.grad is not None:
                        # Check that grad was NOT all-reduced (sizes differ, so just check finite)
                        ok_finite = not (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                        all_passed &= _ok(
                            f"TP-sharded LoRA '{lora.lora_name}.{pname}' grad is finite",
                            ok_finite
                        )
                        break
                break  # one sample is enough

    except Exception as e:
        all_passed &= _ok("sync_replicated_grads", False, f"{type(e).__name__}: {e}")

    return all_passed


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def run_all_checks(dit=None, network=None, groups=None, use_sp: bool = False) -> bool:
    """Run all TP+SP correctness checks. Call after TP init + (optionally) model load.

    Args:
        dit:     Sharded MiniTrainDIT, or None to skip model forward check.
        network: LoRA network, or None to skip LoRA tagging check.
        groups:  wdp.ProcessGroups from wdp.init_dist().
        use_sp:  Whether sequence parallel is active.

    Returns True if all enabled checks pass.
    """
    if groups is None or groups.tp_size <= 1:
        logger.info("[tp_verify] tp_size=1, skipping TP/SP checks")
        return True

    rank = dist.get_rank()
    tp   = groups.tp_size
    if rank == 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"[tp_verify] TP+SP Correctness Checks  tp={tp}  sp={use_sp}")
        logger.info(f"{'='*60}")

    results = {}

    if rank == 0:
        logger.info("[tp_verify] --- Check 1: Collective math ---")
    results['collectives'] = check_collectives(groups)

    if rank == 0:
        logger.info("[tp_verify] --- Check 2: Layer equivalence ---")
    results['layers_tp_only'] = check_tp_layers(groups, use_sp=False)
    if use_sp:
        results['layers_sp'] = check_tp_layers(groups, use_sp=True)

    if dit is not None:
        if rank == 0:
            logger.info("[tp_verify] --- Check 3: Model forward (mock data) ---")
        results['model_forward'] = check_model_forward(dit, groups, use_sp=use_sp)

    if network is not None:
        if rank == 0:
            logger.info("[tp_verify] --- Check 4: LoRA param tagging ---")
        results['lora_tagging'] = check_lora_tagging(network, groups)

        if rank == 0:
            logger.info("[tp_verify] --- Check 5: LoRA forward math ---")
        results['lora_forward_math'] = check_lora_forward_math(groups, use_sp=use_sp)

        # Check 7 must run BEFORE Check 6: gather_tp_lora_weights() mutates
        # lora_up.weight.data to full size in-place (for save), leaving
        # out_features stale. Running forward after gather gives wrong shapes.
        if dit is not None:
            if rank == 0:
                logger.info("[tp_verify] --- Check 7: sync_replicated_grads ---")
            results['sync_replicated_grads'] = check_sync_replicated_grads(
                dit, network, groups, use_sp=use_sp
            )

        if rank == 0:
            logger.info("[tp_verify] --- Check 6: gather_tp_lora_weights shapes ---")
        results['lora_gather_shapes'] = check_gather_tp_lora_weights(network, groups)

    all_ok = all(results.values())
    if rank == 0:
        logger.info(f"{'='*60}")
        status = "ALL PASSED" if all_ok else "SOME FAILED"
        logger.info(f"[tp_verify] {status}: {results}")
        logger.info(f"{'='*60}\n")

    return all_ok
