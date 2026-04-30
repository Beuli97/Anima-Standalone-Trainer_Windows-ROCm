"""
FSDP integration for wd_parallel.

wrap_fsdp() applies PyTorch FullyShardedDataParallel using the DP process group.
It is composable with apply_parallelism (TP+SP):

    # Apply TP first (shards weight matrices across tp ranks)
    model = wdp.apply_parallelism(model, spec, config, groups)

    # Then wrap with FSDP (shards already-sharded params + optim states across dp ranks)
    model = wdp.wrap_fsdp(model, config, groups, transformer_layer_cls={MyBlock})

What FSDP adds on top of TP:
  - Optimizer states (Adam m/v) sharded across DP ranks  → largest memory win
  - Params + grads further sharded across DP ranks
  - Each DP rank processes different training samples  → higher throughput

Communication:
  - FSDP uses dist.all_gather_into_tensor + dist.reduce_scatter_tensor
  - These route through whatever backend is registered (cuda_direct / gloo)
  - Same transport as wd_parallel collectives → no backend conflicts

Mixed precision:
  - default: params stored as bfloat16, reduce in float32
  - set mixed_precision=False to disable (e.g. for debugging)

CPU offload:
  - set cpu_offload=True to offload optimizer states to CPU RAM
  - useful when VRAM is the bottleneck for very large models
"""

from typing import Optional, Type, Set
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import functools

from .config import ParallelConfig
from .initialize import ProcessGroups


def wrap_fsdp(
    model:                  nn.Module,
    config:                 ParallelConfig,
    groups:                 ProcessGroups,
    *,
    transformer_layer_cls:  Optional[Set[Type[nn.Module]]] = None,
    min_num_params:         int = 1_000_000,
    mixed_precision:        bool = True,
    cpu_offload:            bool = False,
    sharding_strategy:      ShardingStrategy = ShardingStrategy.FULL_SHARD,
) -> nn.Module:
    """
    Wrap model with FSDP using the DP process group.

    Args:
        model:                  Model to wrap. If TP was already applied, wrap after.
        config:                 ParallelConfig with dp=True.
        groups:                 ProcessGroups from init_dist (must have groups.dp set).
        transformer_layer_cls:  Set of nn.Module subclasses to use as FSDP wrap
                                boundaries (e.g. {DiTBlock}). Preferred over
                                size-based wrapping — gives one FSDP unit per block.
        min_num_params:         Fallback wrap threshold when transformer_layer_cls
                                is not given. Modules with >= this many params are
                                individually wrapped.
        mixed_precision:        If True, use bfloat16 params with float32 reduction.
                                Should match the dtype you train with.
        cpu_offload:            Offload optimizer states (and optionally params) to CPU.
                                Useful for very large models. Adds D2H/H2D overhead.
        sharding_strategy:      FULL_SHARD (default) shards params+grads+optim.
                                SHARD_GRAD_OP shards only grads+optim (lower comm).
                                NO_SHARD disables sharding (useful for debugging).

    Returns:
        FSDP-wrapped model. Gradients are automatically synced — do NOT call
        sync_replicated_grads() for DP-sharded params (FSDP handles it).
        Still call sync_replicated_grads() for TP non-sharded params if tp is active.
    """
    if not config.dp:
        return model

    if groups.dp is None:
        raise RuntimeError("wrap_fsdp called but groups.dp is None — "
                           "init_dist was not called with dp=True in config")

    # --- Mixed precision policy ---
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype  = torch.bfloat16,
            reduce_dtype = torch.float32,
            buffer_dtype = torch.bfloat16,
        )

    # --- CPU offload ---
    cpu_offload_cfg = CPUOffload(offload_params=True) if cpu_offload else None

    # --- Auto-wrap policy ---
    if transformer_layer_cls is not None:
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls,
        )
    else:
        wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params,
        )

    device_id = None
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
    sync_module_states = device_id is not None

    model = FSDP(
        model,
        process_group        = groups.dp,
        sharding_strategy    = sharding_strategy,
        auto_wrap_policy     = wrap_policy,
        mixed_precision      = mp_policy,
        cpu_offload          = cpu_offload_cfg,
        device_id            = device_id,
        sync_module_states   = sync_module_states,
        forward_prefetch     = True,   # overlap next layer all-gather with current compute
        use_orig_params      = True,   # preserve _tp_sharded markers for TP grad sync
    )

    return model
