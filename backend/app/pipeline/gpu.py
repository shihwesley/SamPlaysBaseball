"""GPU/MPS memory management utilities."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return MPS device if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_cache() -> None:
    """Release cached GPU memory."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def auto_batch_size(available_gb: float) -> int:
    """Return safe batch size for given available memory.

    >8 GB → 4, >4 GB → 2, else → 1
    """
    if available_gb > 8:
        return 4
    if available_gb > 4:
        return 2
    return 1


def memory_stats() -> dict:
    """Return basic device memory info.

    MPS does not expose per-allocation stats like CUDA, so we return
    availability only.
    """
    if torch.backends.mps.is_available():
        return {"device": "mps", "available": True}
    if torch.cuda.is_available():
        return {"device": "cuda", "available": True}
    return {"device": "cpu", "available": True}
