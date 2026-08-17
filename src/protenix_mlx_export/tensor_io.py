"""Deterministic tensor serialization helpers for exporter artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
from safetensors.torch import save_file

if TYPE_CHECKING:
    from pathlib import Path


def detached_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a contiguous, serialization-safe tensor without graph ownership."""
    return tensor.detach().cpu().contiguous().clone()


def save_torch_tensors(tensors: Mapping[str, torch.Tensor], path: Path) -> None:
    """Save tensors with deterministic key order and no pickle metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = {name: detached_cpu_tensor(tensors[name]) for name in sorted(tensors)}
    save_file(prepared, path)


def flatten_tensors(value: object, prefix: str) -> dict[str, torch.Tensor]:
    """Flatten tensors nested in tuples, lists, and string-keyed mappings.

    Upstream modules return bare tensors, tuples and dicts depending on the module, so
    a fixture recorder cannot assume any one shape.
    """
    if isinstance(value, torch.Tensor):
        return {prefix: detached_cpu_tensor(value)}
    if isinstance(value, Mapping):
        flattened: dict[str, torch.Tensor] = {}
        for key in sorted(value, key=str):
            flattened.update(flatten_tensors(value[key], f"{prefix}.{key}"))
        return flattened
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(flatten_tensors(item, f"{prefix}.{index}"))
        return flattened
    return {}
