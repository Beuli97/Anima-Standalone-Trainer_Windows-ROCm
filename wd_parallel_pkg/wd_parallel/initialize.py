"""
Distributed initialization and process group management.

Usage:
    dist.init_process_group()                # caller/training script owns backend
    groups = wdp.init_dist(config)           # build TP/DP sub-groups

    # Or force a backend explicitly:
    # backend = wdp.activate_backend("cuda_direct")
    # dist.init_process_group(backend=backend)
    # ... training ...
    wdp.destroy_dist()

2D mesh layout (dp outer, tp inner):
    rank = dp_rank * tp_size + tp_rank

    4 GPUs, tp=2, dp=2:
        TP groups: [0,1]  [2,3]   (same dp row)
        DP groups: [0,2]  [1,3]   (same tp column)

With tp-only or dp-only, one dimension is 1 and the active mesh dimension is
still represented by a valid process group for that axis.
"""

from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist

from .config import ParallelConfig


@dataclass
class ProcessGroups:
    """Holds one process group per active parallelism dimension."""
    tp: dist.ProcessGroup                    # Tensor + Sequence Parallel group
    dp: Optional[dist.ProcessGroup] = None   # Data Parallel (FSDP) group

    # --- TP helpers ---
    @property
    def tp_rank(self) -> int:
        return dist.get_rank(group=self.tp)

    @property
    def tp_size(self) -> int:
        return dist.get_world_size(group=self.tp)

    # --- DP helpers ---
    @property
    def dp_rank(self) -> int:
        if self.dp is None:
            return 0
        return dist.get_rank(group=self.dp)

    @property
    def dp_size(self) -> int:
        if self.dp is None:
            return 1
        return dist.get_world_size(group=self.dp)

    @property
    def is_dp_main(self) -> bool:
        """True on the rank-0 of each DP group (used for TP-local logging)."""
        return self.dp_rank == 0


def init_dist(config: ParallelConfig) -> ProcessGroups:
    """
    Build TP/DP process sub-groups from an already-initialized dist.

    Caller is responsible for calling dist.init_process_group() before this.
    Omit the backend to let the training script / PyTorch choose, or use
    wdp.activate_backend(<backend>) to force one:

        backend = wdp.activate_backend("gloo")
        dist.init_process_group(backend=backend)
        groups = wdp.init_dist(config)

    2D mesh (tp + dp): sub-groups are built from the global process group.
    Rank layout: rank = dp_rank * tp_size + tp_rank
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "dist.init_process_group() must be called before wdp.init_dist(). "
            "Let the training script / PyTorch choose the backend:\n\n"
            "    dist.init_process_group()\n\n"
            "Or force one explicitly:\n\n"
            "    backend = wdp.activate_backend('gloo')\n"
            "    dist.init_process_group(backend=backend)\n"
            "    groups = wdp.init_dist(config)\n"
        )

    world_size = dist.get_world_size()
    rank       = dist.get_rank()
    backend    = dist.get_backend()
    config.validate(world_size)

    dp_size, tp_size = config.mesh_degrees(world_size)

    if tp_size < 2:
        raise ValueError(
            f"init_dist requires tp_size >= 2 (got {tp_size}). "
            "Use standard DDP for data-parallel-only training."
        )

    if rank == 0:
        print(f"  [wd_parallel] backend={backend}  "
              f"world={world_size}  tp={tp_size}  dp={dp_size}")

    # --- Build TP sub-groups ---
    # Ranks sharing the same dp_rank form a TP group.
    # rank = dp_rank * tp_size + tp_rank  →  tp group for dp_rank d = [d*tp_size .. (d+1)*tp_size)
    if tp_size > 1:
        tp_group = None
        for d in range(dp_size):
            ranks = list(range(d * tp_size, (d + 1) * tp_size))
            g = dist.new_group(ranks)
            if rank in ranks:
                tp_group = g

    # --- Build DP sub-groups ---
    # Ranks sharing the same tp_rank form a DP group.
    # rank = dp_rank * tp_size + tp_rank  →  dp group for tp_rank t = [t, t+tp_size, t+2*tp_size, ...]
    if dp_size > 1:
        dp_group = None
        for t in range(tp_size):
            ranks = [t + d * tp_size for d in range(dp_size)]
            g = dist.new_group(ranks)
            if rank in ranks:
                dp_group = g
    else:
        dp_group = None  # no DP dimension active

    return ProcessGroups(tp=tp_group, dp=dp_group)


def destroy_dist() -> None:
    """Clean up the process group. Call at end of training."""
    if dist.is_initialized():
        dist.destroy_process_group()
