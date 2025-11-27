"""
Lightweight helpers for device selection and tensor conversions.
This keeps torch usage centralized so the rest of the code can stay concise.
"""
from __future__ import annotations

from typing import Any, Optional

import torch


def get_default_device(device: Optional[str] = None) -> torch.device:
    """
    Pick a sensible default device.
    - If `device` is provided, honor it.
    - Else, prefer CUDA when available.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(data: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert arbitrary array-like data to a torch tensor on the requested device."""
    return torch.as_tensor(data, device=device, dtype=dtype)


def to_numpy(tensor: torch.Tensor):
    """Detach and move tensor to CPU numpy safely."""
    return tensor.detach().cpu().numpy()


def describe_device(device: torch.device) -> str:
    """Human-readable device description for logging."""
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return str(device)
