"""Tensor-name rules shared by checkpoint export and parity fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Every released Protenix checkpoint was saved from a DDP-wrapped module, so each
#: key carries this prefix. Upstream's loader strips it conditionally; we do the same
#: rather than unconditionally, because a future non-DDP checkpoint would otherwise
#: lose the first component of every name.
DDP_PREFIX = "module."

#: Roots of the inference graph, in the order upstream's `Protenix.forward` runs them.
#: This is the WHOLE released state dict, not a subset: Protenix ships structure and
#: confidence in one checkpoint with no training-only tensors mixed in, so there is no
#: Boltz-style prefix filter here. The tuple exists to VALIDATE -- an unrecognized root
#: means the checkpoint holds something this port does not model, which is a hard error
#: rather than a tensor to quietly carry along.
GRAPH_ROOTS = (
    "input_embedder",
    "relative_position_encoding",
    "template_embedder",
    "msa_module",
    "pairformer_stack",
    "diffusion_module",
    "distogram_head",
    "confidence_head",
    # Top-level projections and norms owned by `Protenix` itself (Algorithm 1).
    "linear_no_bias_sinit",
    "linear_no_bias_zinit1",
    "linear_no_bias_zinit2",
    "linear_no_bias_token_bond",
    "linear_no_bias_z_cycle",
    "linear_no_bias_s",
    "layernorm_s",
    "layernorm_z_cycle",
    # Present only in the constraint variants; harmless when absent.
    "constraint_embedder",
)

Value = TypeVar("Value")


def strip_ddp_prefix(name: str) -> str:
    """Remove the `module.` wrapper DDP adds to every saved parameter name."""
    return name[len(DDP_PREFIX) :] if name.startswith(DDP_PREFIX) else name


def normalize_state_dict(state: Mapping[str, Value]) -> dict[str, Value]:
    """Return the checkpoint's tensors under runtime names, in sorted order."""
    normalized: dict[str, Value] = {}
    for raw_name in sorted(state):
        name = swift_tensor_name(strip_ddp_prefix(raw_name))
        if name in normalized:
            message = f"two checkpoint keys normalize to the same name: {name}"
            raise ValueError(message)
        normalized[name] = state[raw_name]
    return normalized


def swift_tensor_name(name: str) -> str:
    """Normalize wrapper-only path components out of a tensor name.

    `_orig_mod` is inserted by `torch.compile`; `_checkpoint_wrapped_module` by
    activation checkpointing. Neither is a real module, and a runtime that took them
    literally would look for weights under paths that describe the training harness
    rather than the graph.
    """
    dropped = {"_orig_mod", "_checkpoint_wrapped_module", "_fsdp_wrapped_module"}
    return ".".join(
        component for component in name.split(".") if component not in dropped
    )


def graph_root(name: str) -> str:
    """The top-level module a normalized tensor name belongs to."""
    return name.split(".", maxsplit=1)[0]


def unrecognized_roots(names: object) -> tuple[str, ...]:
    """Roots present in `names` that this port does not model, sorted."""
    known = set(GRAPH_ROOTS)
    found = {graph_root(str(name)) for name in names}  # type: ignore[union-attr]
    return tuple(sorted(found - known))
