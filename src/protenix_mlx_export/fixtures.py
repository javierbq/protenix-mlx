"""Record PyTorch module-boundary fixtures so the Swift port can be checked against it.

A port of a 464M-parameter network cannot be validated by reading it. Each module is
instantiated here at small dimensions with seeded random weights, run on seeded random
inputs, and its weights/inputs/outputs written to a fixture directory; the Swift tests
replay the same weights and inputs and assert the same outputs. A layer that transposes
the wrong axis or applies a gate in the wrong order fails immediately and by name,
instead of surfacing as a plausible-looking structure at the end of a 200-step diffusion.

Upstream is importable with torch alone provided ``LAYERNORM_TYPE`` is anything other
than ``fast_layernorm`` -- the default tries to build a CUDA extension. This module sets
that automatically, so importing it is enough.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch

from protenix_mlx_export.tensor_io import flatten_tensors, save_torch_tensors

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Must be set before upstream is imported: `protenix.model.triangular.layers` reads it
#: at import time and otherwise tries to compile `fast_layer_norm_cuda_v2`, which needs
#: nvcc and a CUDA device.
os.environ.setdefault("LAYERNORM_TYPE", "torch")

#: Small enough that fixtures stay a few megabytes, but not so small that a broken
#: reshape accidentally passes. Head counts divide the widths exactly, as upstream
#: asserts.
SMALL = {
    "c_z": 32,
    "c_s": 24,
    "c_a": 32,
    "c_hidden": 16,
    "n_heads": 4,
    "n_tokens": 7,
    "n_seq": 5,
}


class UpstreamUnavailableError(RuntimeError):
    """Raised when upstream Protenix is not importable for fixture generation."""


def _require_upstream(source: Path | None) -> None:
    """Put a Protenix checkout on the import path, or explain why we cannot."""
    if source is not None:
        sys.path.insert(0, str(source))
    try:
        import protenix.model.modules.transformer  # noqa: F401, PLC0415
    except ImportError as error:
        message = (
            "fixture generation needs an importable Protenix checkout; pass "
            f"--source /path/to/Protenix (import failed: {error})"
        )
        raise UpstreamUnavailableError(message) from error


@dataclass(frozen=True)
class FixtureCase:
    """One module, its construction, and the inputs it is exercised on."""

    name: str
    #: Returns (module, inputs). Called under a seeded RNG.
    build: Callable[[], tuple[torch.nn.Module, dict[str, torch.Tensor]]]
    #: Extra values the Swift side needs to rebuild the module (dims, flags).
    config: dict[str, Any]
    #: How to call the module. Defaults to keyword-splatting the inputs.
    run: Callable[[torch.nn.Module, dict[str, torch.Tensor]], Any] | None = None


def _randomize(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Replace every parameter with seeded noise.

    Upstream initializes many matrices to exact zeros (`initializer="zeros"` on the
    AdaLN projections and every transition's output). A fixture built on those would
    pass even if the Swift side dropped the term entirely, so all parameters get real
    values here regardless of how they were initialized.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator) * 0.5
            )
    return module.eval()


def _cases() -> list[FixtureCase]:
    """Every module boundary the Swift port is checked against."""
    from protenix.model.modules.primitives import (  # noqa: PLC0415
        AdaptiveLayerNorm,
        Transition,
    )
    from protenix.model.modules.transformer import AttentionPairBias  # noqa: PLC0415
    from protenix.model.triangular.layers import OuterProductMean  # noqa: PLC0415
    from protenix.model.triangular.triangular import (  # noqa: PLC0415
        TriangleAttention,
        TriangleAttentionEndingNode,
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )

    c_z, c_s, c_a = SMALL["c_z"], SMALL["c_s"], SMALL["c_a"]
    c_hidden, n_heads = SMALL["c_hidden"], SMALL["n_heads"]
    n_tokens, n_seq = SMALL["n_tokens"], SMALL["n_seq"]

    def pair() -> torch.Tensor:
        return torch.randn(1, n_tokens, n_tokens, c_z)

    def single(width: int = c_s) -> torch.Tensor:
        return torch.randn(1, n_tokens, width)

    cases: list[FixtureCase] = [
        FixtureCase(
            name="transition",
            build=lambda: (
                Transition(c_in=c_z, n=2),
                {"x": pair()},
            ),
            config={"c_in": c_z, "n": 2},
        ),
        FixtureCase(
            name="adaptive_layernorm",
            build=lambda: (
                AdaptiveLayerNorm(c_a=c_a, c_s=c_s),
                {"a": single(c_a), "s": single(c_s)},
            ),
            config={"c_a": c_a, "c_s": c_s},
            run=lambda module, inputs: module(inputs["a"], inputs["s"]),
        ),
        FixtureCase(
            name="attention_pair_bias_with_s",
            build=lambda: (
                AttentionPairBias(
                    has_s=True, n_heads=n_heads, c_a=c_a, c_s=c_s, c_z=c_z
                ),
                {"a": single(c_a), "s": single(c_s), "z": pair()},
            ),
            config={
                "has_s": True, "n_heads": n_heads, "c_a": c_a, "c_s": c_s, "c_z": c_z,
            },
            run=lambda module, inputs: module(
                a=inputs["a"], s=inputs["s"], z=inputs["z"]
            ),
        ),
        FixtureCase(
            # has_s=False takes the plain-LayerNorm branch and drops the output gate --
            # a different graph, not merely a different argument.
            name="attention_pair_bias_no_s",
            build=lambda: (
                AttentionPairBias(
                    has_s=False, n_heads=n_heads, c_a=c_a, c_s=c_s, c_z=c_z
                ),
                {"a": single(c_a), "z": pair()},
            ),
            config={
                "has_s": False, "n_heads": n_heads, "c_a": c_a, "c_s": c_s, "c_z": c_z,
            },
            run=lambda module, inputs: module(a=inputs["a"], s=None, z=inputs["z"]),
        ),
        FixtureCase(
            name="triangle_multiplication_outgoing",
            build=lambda: (
                TriangleMultiplicationOutgoing(c_z=c_z, c_hidden=c_hidden),
                {"x": pair()},
            ),
            config={"c_z": c_z, "c_hidden": c_hidden, "outgoing": True},
            run=lambda module, inputs: module(inputs["x"], mask=None),
        ),
        FixtureCase(
            name="triangle_multiplication_incoming",
            build=lambda: (
                TriangleMultiplicationIncoming(c_z=c_z, c_hidden=c_hidden),
                {"x": pair()},
            ),
            config={"c_z": c_z, "c_hidden": c_hidden, "outgoing": False},
            run=lambda module, inputs: module(inputs["x"], mask=None),
        ),
        FixtureCase(
            name="triangle_attention_starting",
            build=lambda: (
                TriangleAttention(c_in=c_z, c_hidden=c_hidden, no_heads=n_heads),
                {"x": pair()},
            ),
            config={
                "c_in": c_z, "c_hidden": c_hidden, "n_heads": n_heads,
                "starting": True,
            },
            run=lambda module, inputs: module(inputs["x"], mask=None),
        ),
        FixtureCase(
            name="triangle_attention_ending",
            build=lambda: (
                TriangleAttentionEndingNode(
                    c_in=c_z, c_hidden=c_hidden, no_heads=n_heads
                ),
                {"x": pair()},
            ),
            config={
                "c_in": c_z, "c_hidden": c_hidden, "n_heads": n_heads,
                "starting": False,
            },
            run=lambda module, inputs: module(inputs["x"], mask=None),
        ),
        FixtureCase(
            name="outer_product_mean",
            build=lambda: (
                OuterProductMean(c_m=c_s, c_z=c_z, c_hidden=c_hidden),
                {"m": torch.randn(1, n_seq, n_tokens, c_s)},
            ),
            config={"c_m": c_s, "c_z": c_z, "c_hidden": c_hidden},
            run=lambda module, inputs: module(inputs["m"], mask=None),
        ),
    ]
    return cases


def _as_tensor_map(value: Any, prefix: str) -> dict[str, torch.Tensor]:
    """Flatten whatever a module returned into named tensors."""
    flattened = flatten_tensors(value, prefix)
    if not flattened:
        message = f"module produced no tensors under {prefix}"
        raise ValueError(message)
    return flattened


def write_fixture(case: FixtureCase, *, output: Path, seed: int = 0) -> Path:
    """Build, run and record one case."""
    torch.manual_seed(seed)
    module, inputs = case.build()
    _randomize(module, seed=seed + 1)

    with torch.no_grad():
        result = case.run(module, inputs) if case.run else module(**inputs)

    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in module.state_dict().items():
        tensors[f"weight.{name}"] = parameter
    for name, value in inputs.items():
        tensors[f"input.{name}"] = value
    tensors.update(_as_tensor_map(result, "output"))

    directory = output / case.name
    directory.mkdir(parents=True, exist_ok=True)
    save_torch_tensors(tensors, directory / "tensors.safetensors")
    def _under(section: str) -> dict[str, list[int]]:
        # `section` and `section.*` both count: `flatten_tensors` names a bare tensor
        # return exactly "output", with no trailing dot, so a prefix-only test silently
        # indexes nothing for every module that returns one tensor -- which is most of
        # them.
        return {
            name: list(tensors[name].shape)
            for name in sorted(tensors)
            if name == section or name.startswith(f"{section}.")
        }

    outputs = _under("output")
    if not outputs:
        message = f"{case.name} recorded no outputs"
        raise ValueError(message)
    metadata = {
        "name": case.name,
        "seed": seed,
        "config": case.config,
        "weights": sorted(_under("weight")),
        "inputs": _under("input"),
        "outputs": outputs,
    }
    (directory / "case.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def make_fixtures(
    *, output: Path, source: Path | None = None, seed: int = 0
) -> list[Path]:
    """Record every module fixture into `output`."""
    _require_upstream(source)
    written = [
        write_fixture(case, output=output, seed=seed) for case in _cases()
    ]
    index = {
        "seed": seed,
        "dimensions": SMALL,
        "cases": sorted(path.name for path in written),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return written


def case_names() -> tuple[str, ...]:
    """Names of the fixtures :func:`make_fixtures` writes, without importing upstream."""
    return (
        "adaptive_layernorm",
        "attention_pair_bias_no_s",
        "attention_pair_bias_with_s",
        "outer_product_mean",
        "transition",
        "triangle_attention_ending",
        "triangle_attention_starting",
        "triangle_multiplication_incoming",
        "triangle_multiplication_outgoing",
    )


def _unused(value: Mapping[str, object]) -> None:  # pragma: no cover
    """Placeholder kept so `Mapping` stays referenced under TYPE_CHECKING."""
