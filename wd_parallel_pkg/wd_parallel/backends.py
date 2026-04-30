"""
Backend selection for wd_parallel.

activate_backend() is an opt-in helper for activating a specific distributed
backend before dist.init_process_group():

    backend = wdp.activate_backend("cuda_direct")
    dist.init_process_group(backend=backend)
    groups = wdp.init_dist(config)          # build TP/DP sub-groups

If the caller does not pass a backend flag, skip activate_backend() and let the
training script / PyTorch choose its normal default. wd_parallel itself never
calls dist.init_process_group().

Backend selection is explicit:
  * "cuda_direct" imports and registers cuda_direct, then returns "cuda_direct"
  * "gloo" and "nccl" return the requested backend unchanged after validation
  * "auto" chooses a standard PyTorch backend only: NCCL when available on
    non-Windows CUDA builds, otherwise Gloo

In particular, "auto" does not activate cuda_direct. Users who want
cuda_direct must request it directly.
"""

import os
import sys


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _nccl_available() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_nccl_available()
    except Exception:
        return False


def _activate_cuda_direct() -> None:
    """Import and register the cuda_direct backend."""
    cd_path = os.environ.get("CUDA_DIRECT_PATH", r"C:\cuda-direct-backend")
    if cd_path not in sys.path:
        sys.path.insert(0, cd_path)

    from cuda_direct_backend.auto import activate
    activate()


def activate_backend(backend: str) -> str:
    """
    Activate and return an explicitly requested distributed backend.

    Args:
        backend: "auto"        - use NCCL when available on non-Windows CUDA,
                                 otherwise use Gloo
                 "cuda_direct" - force cuda_direct (Windows SHM zero-copy)
                 "gloo"        - force gloo (TCP, cross-platform fallback)
                 "nccl"        - force NCCL (Linux/GPU clusters)

    Returns:
        The resolved backend name to pass to dist.init_process_group().

    Example::

        import torch.distributed as dist
        import wd_parallel as wdp

        backend = wdp.activate_backend("cuda_direct")
        dist.init_process_group(backend=backend)
        groups = wdp.init_dist(config)
    """
    if backend is None:
        raise ValueError(
            "activate_backend() requires an explicit backend. "
            "Skip this call to let the training script / PyTorch choose."
        )
    if backend not in {"auto", "cuda_direct", "gloo", "nccl"}:
        raise ValueError(
            f"Unsupported backend {backend!r}. "
            "Expected one of: auto, cuda_direct, gloo, nccl."
        )

    if os.name == "nt":
        # Disable libuv transport on Windows - use socket-based comm instead.
        # libuv is broken for PyTorch dist on Windows as of PyTorch 2.x.
        os.environ.setdefault("USE_LIBUV", "0")
        if backend == "nccl":
            raise ValueError("NCCL is not supported on Windows. Use gloo.")

    if backend == "cuda_direct":
        if os.name != "nt":
            raise ValueError("cuda_direct is only supported on Windows. Use nccl or gloo.")
        _activate_cuda_direct()
        return "cuda_direct"

    if backend != "auto":
        return backend

    if os.name != "nt" and _cuda_available() and _nccl_available():
        return "nccl"
    return "gloo"
