"""
torch.distributed compatibility shim for ROCm/Windows.

ROCm PyTorch builds on Windows may ship a stub `torch.distributed` module
that exists but lacks the actual backend functions (is_initialized,
all_reduce, etc.) because no distributed backend (gloo/NCCL) is compiled in.

This module provides a safe wrapper that:
  - Uses the real `torch.distributed` when fully available (CUDA/normal builds).
  - Falls back to stub no-ops when only a stub module is present (ROCm Windows).

Usage (replace `import torch.distributed as dist` with):
    from library.dist_compat import dist
"""

import torch

try:
    import torch.distributed as _dist

    # Check whether this is a real or stub module by probing for a core function.
    _HAS_DIST = hasattr(_dist, "is_initialized")
except Exception:
    _dist = None
    _HAS_DIST = False


if _HAS_DIST:
    # Normal build — pass through to the real torch.distributed
    dist = _dist
else:
    # ROCm Windows stub — provide no-op replacements so single-GPU training works.
    # Multi-GPU training is not supported on this platform anyway.

    class _ReduceOp:
        SUM = None
        MAX = None
        MIN = None
        PRODUCT = None

    class _StubDist:
        """No-op stub that mimics the torch.distributed API surface."""

        ReduceOp = _ReduceOp

        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def is_available():
            return False

        @staticmethod
        def init_process_group(*args, **kwargs):
            raise RuntimeError(
                "torch.distributed is not available in this PyTorch build (ROCm/Windows). "
                "Multi-GPU training is not supported."
            )

        @staticmethod
        def destroy_process_group(*args, **kwargs):
            pass

        @staticmethod
        def get_rank(*args, **kwargs):
            return 0

        @staticmethod
        def get_world_size(*args, **kwargs):
            return 1

        @staticmethod
        def barrier(*args, **kwargs):
            pass

        @staticmethod
        def all_reduce(tensor, *args, **kwargs):
            pass  # no-op for single process

        @staticmethod
        def broadcast(tensor, *args, **kwargs):
            pass

        @staticmethod
        def all_gather(tensor_list, *args, **kwargs):
            pass

        @staticmethod
        def reduce(tensor, *args, **kwargs):
            pass

        @staticmethod
        def get_backend(*args, **kwargs):
            return "undefined"

    dist = _StubDist()
