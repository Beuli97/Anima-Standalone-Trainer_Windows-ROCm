"""
ParallelConfig declares which parallelism strategies are active.

Mesh dimension ordering (outer -> inner): dp, cp, tp
TP and SP share the same process group.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParallelConfig:
    tp: bool = False   # Tensor Parallel: shard weights across TP ranks
    sp: bool = False   # Sequence Parallel: SP region between TP ops (requires tp)
    dp: bool = False   # Data Parallel: FSDP across DP ranks (outer mesh dimension)
    tp_degree: Optional[int] = None  # Optional TP degree for tp+dp meshes
    dp_degree: Optional[int] = None  # Optional DP degree for tp+dp meshes

    def _validate_degree(self, name: str, value: Optional[int]) -> None:
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be > 0")

    def _auto_tp_degree(self, world_size: int) -> int:
        tp_size = int(math.isqrt(world_size))
        while tp_size > 1 and world_size % tp_size != 0:
            tp_size -= 1
        return tp_size if tp_size > 1 else world_size

    def validate(self, world_size: int) -> None:
        if self.sp and not self.tp:
            raise ValueError("sp requires tp - they share the same process group")
        self._validate_degree("tp_degree", self.tp_degree)
        self._validate_degree("dp_degree", self.dp_degree)

        n_active = int(self.tp) + int(self.dp)
        if n_active == 0:
            return

        # Validate that active modes can resolve to a valid mesh.
        self._mesh_degrees(world_size)

    def mesh_degrees(self, world_size: int):
        """Return (dp_size, tp_size)."""
        return self._mesh_degrees(world_size)

    def _mesh_degrees(self, world_size: int):
        if not self.tp and not self.dp:
            return 1, 1

        if self.tp and self.dp:
            tp_size = self.tp_degree
            dp_size = self.dp_degree
            if tp_size is None and dp_size is None:
                tp_size = self._auto_tp_degree(world_size)
                dp_size = world_size // tp_size
            elif tp_size is None:
                if world_size % dp_size != 0:
                    raise ValueError(
                        f"world_size={world_size} is not divisible by dp_degree={dp_size}"
                    )
                tp_size = world_size // dp_size
            elif dp_size is None:
                if world_size % tp_size != 0:
                    raise ValueError(
                        f"world_size={world_size} is not divisible by tp_degree={tp_size}"
                    )
                dp_size = world_size // tp_size

            if tp_size * dp_size != world_size:
                raise ValueError(
                    f"tp_degree*dp_degree must equal world_size "
                    f"({tp_size}*{dp_size} != {world_size})"
                )
            if tp_size < 2:
                raise ValueError(
                    f"tp+dp mesh resolved to tp_size={tp_size}; tp requires >= 2 ranks"
                )
            if dp_size < 2:
                raise ValueError(
                    f"tp+dp mesh resolved to dp_size={dp_size}; dp requires >= 2 ranks"
                )
            return dp_size, tp_size

        if self.tp:
            tp_size = self.tp_degree or world_size
            if tp_size != world_size:
                raise ValueError(
                    f"tp-only expects tp_degree==world_size ({world_size}), got {tp_size}"
                )
            return 1, world_size

        dp_size = self.dp_degree or world_size
        if dp_size != world_size:
            raise ValueError(
                f"dp-only expects dp_degree==world_size ({world_size}), got {dp_size}"
            )
        return world_size, 1

    @property
    def is_distributed(self) -> bool:
        return self.tp or self.dp
