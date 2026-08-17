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
    "c_noise": 16,
    "c_atompair": 8,
    # Windows small enough for a quick fixture but with the same even/ordering
    # constraints upstream asserts; n_atoms is not a multiple of n_queries on purpose.
    "n_queries": 4,
    "n_keys": 8,
    "n_atoms": 11,
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

    from protenix.model.modules.diffusion import (  # noqa: PLC0415
        DiffusionConditioning,
    )
    from protenix.model.modules.transformer import (  # noqa: PLC0415
        AtomTransformer,
        ConditionedTransitionBlock,
        DiffusionTransformer,
        DiffusionTransformerBlock,
    )
    from protenix.model.modules.pairformer import (  # noqa: PLC0415
        MSAModule,
        MSAPairWeightedAveraging,
        MSAStack,
        PairformerBlock,
        PairformerStack,
    )

    # `c_hidden_pair_att` must divide c_z when hidden_scale_up is on, because upstream
    # derives the head count as c_z // c_hidden_pair_att.
    pair_attention_width = c_z // SMALL["n_heads"]

    def pairformer_kwargs(*, scale_up: bool) -> dict[str, Any]:
        return {
            "n_heads": SMALL["n_heads"],
            "c_z": c_z,
            "c_s": c_s,
            "c_hidden_mul": c_hidden,
            "c_hidden_pair_att": pair_attention_width,
            "no_heads_pair": SMALL["n_heads"],
            "dropout": 0.0,
            "hidden_scale_up": scale_up,
        }

    cases += [
        FixtureCase(
            name="pairformer_block",
            build=lambda: (
                PairformerBlock(**pairformer_kwargs(scale_up=False)),
                {"s": single(c_s), "z": pair()},
            ),
            config={
                **pairformer_kwargs(scale_up=False),
                "pair_head_count": SMALL["n_heads"],
            },
            run=lambda module, inputs: module(
                s=inputs["s"], z=inputs["z"], pair_mask=None
            ),
        ),
        FixtureCase(
            # protenix-v2 turns this on, which changes the hidden widths and the
            # pair-attention head count -- i.e. the weight SHAPES, not just a constant.
            name="pairformer_block_scale_up",
            build=lambda: (
                PairformerBlock(**pairformer_kwargs(scale_up=True)),
                {"s": single(c_s), "z": pair()},
            ),
            config={
                **pairformer_kwargs(scale_up=True),
                "pair_head_count": c_z // pair_attention_width,
            },
            run=lambda module, inputs: module(
                s=inputs["s"], z=inputs["z"], pair_mask=None
            ),
        ),
        FixtureCase(
            name="msa_pair_weighted_averaging",
            build=lambda: (
                MSAPairWeightedAveraging(
                    c_m=c_s, c=c_hidden, c_z=c_z, n_heads=SMALL["n_heads"]
                ),
                {"m": torch.randn(1, n_seq, n_tokens, c_s), "z": pair()},
            ),
            config={
                "c_m": c_s, "c": c_hidden, "c_z": c_z, "n_heads": SMALL["n_heads"],
            },
            run=lambda module, inputs: module(m=inputs["m"], z=inputs["z"]),
        ),
        FixtureCase(
            name="msa_stack",
            build=lambda: (
                MSAStack(c_m=c_s, c_z=c_z, c=c_hidden, dropout=0.0),
                {"m": torch.randn(1, n_seq, n_tokens, c_s), "z": pair()},
            ),
            # MSAStack hard-codes MSAPairWeightedAveraging's default 8 heads.
            config={"c_m": c_s, "c_z": c_z, "c": c_hidden, "n_heads": 8},
            run=lambda module, inputs: module(m=inputs["m"], z=inputs["z"]),
        ),
        FixtureCase(
            # Two blocks so the last-block special case (no MSA stack) is exercised
            # alongside a normal one.
            name="msa_module",
            build=lambda: (
                MSAModule(
                    n_blocks=2,
                    c_m=c_s,
                    c_z=c_z,
                    c_s_inputs=c_s,
                    msa_dropout=0.0,
                    pair_dropout=0.0,
                    msa_configs={},
                ),
                {"m": torch.randn(1, n_seq, n_tokens, c_s), "z": pair()},
            ),
            config={
                "n_blocks": 2, "c_m": c_s, "c_z": c_z, "c_s_inputs": c_s,
                "pair_head_count": 4, "msa_head_count": 8,
            },
            # Driven at the block level rather than through MSAModule.forward, which
            # builds `m` from raw MSA features -- that is featurizer work, not trunk
            # work, and is not ported yet.
            run=lambda module, inputs: _run_msa_blocks(
                module, inputs["m"], inputs["z"]
            ),
        ),
        FixtureCase(
            name="conditioned_transition_block",
            build=lambda: (
                ConditionedTransitionBlock(c_a=c_a, c_s=c_s, n=2),
                {"a": single(c_a), "s": single(c_s)},
            ),
            config={"c_a": c_a, "c_s": c_s, "n": 2},
            run=lambda module, inputs: module(a=inputs["a"], s=inputs["s"]),
        ),
        FixtureCase(
            name="diffusion_transformer_block",
            build=lambda: (
                DiffusionTransformerBlock(
                    c_a=c_a, c_s=c_s, c_z=c_z, n_heads=n_heads
                ),
                {"a": single(c_a), "s": single(c_s), "z": pair()},
            ),
            config={"c_a": c_a, "c_s": c_s, "c_z": c_z, "n_heads": n_heads},
            # Returns (a, s, z); only `a` is updated.
            run=lambda module, inputs: module(
                a=inputs["a"], s=inputs["s"], z=inputs["z"]
            )[0],
        ),
        FixtureCase(
            name="diffusion_transformer",
            build=lambda: (
                DiffusionTransformer(
                    c_a=c_a, c_s=c_s, c_z=c_z, n_blocks=3, n_heads=n_heads
                ),
                {"a": single(c_a), "s": single(c_s), "z": pair()},
            ),
            config={
                "c_a": c_a, "c_s": c_s, "c_z": c_z, "n_blocks": 3,
                "n_heads": n_heads,
            },
            run=lambda module, inputs: module(
                a=inputs["a"], s=inputs["s"], z=inputs["z"]
            ),
        ),
        FixtureCase(
            # Exercised through the two halves the port exposes -- the pair half is
            # noise-independent and cacheable, the single half is not.
            name="diffusion_conditioning",
            build=lambda: (
                DiffusionConditioning(
                    sigma_data=16.0, c_z=c_z, c_s=c_s, c_s_inputs=c_s,
                    c_noise_embedding=SMALL["c_noise"],
                ),
                {
                    "t_hat_noise_level": torch.rand(1, 2) * 10 + 0.1,
                    "s_inputs": single(c_s),
                    "s_trunk": single(c_s),
                    "z_trunk": pair(),
                    # The relp one-hot block, width 4*r_max + 2*s_max + 7 with
                    # DiffusionConditioning's default r_max=32, s_max=2.
                    "relpe": torch.randn(1, n_tokens, n_tokens, 4 * 32 + 2 * 2 + 7),
                },
            ),
            config={
                "sigma_data": 16.0, "c_z": c_z, "c_s": c_s, "c_s_inputs": c_s,
                "c_noise": SMALL["c_noise"], "r_max": 32, "s_max": 2,
            },
            run=lambda module, inputs: _run_conditioning(module, inputs),
        ),
        FixtureCase(
            # n_atoms deliberately NOT a multiple of n_queries, so the query tail is
            # padded and the window mask has to be right at both ends.
            name="atom_transformer",
            build=lambda: (
                AtomTransformer(
                    c_atom=c_a,
                    c_atompair=SMALL["c_atompair"],
                    n_blocks=2,
                    n_heads=n_heads,
                    n_queries=SMALL["n_queries"],
                    n_keys=SMALL["n_keys"],
                ),
                {
                    "q": torch.randn(1, SMALL["n_atoms"], c_a),
                    "c": torch.randn(1, SMALL["n_atoms"], c_a),
                    "p": torch.randn(
                        1,
                        -(-SMALL["n_atoms"] // SMALL["n_queries"]),
                        SMALL["n_queries"],
                        SMALL["n_keys"],
                        SMALL["c_atompair"],
                    ),
                },
            ),
            config={
                "c_atom": c_a, "c_atompair": SMALL["c_atompair"], "n_blocks": 2,
                "n_heads": n_heads, "n_queries": SMALL["n_queries"],
                "n_keys": SMALL["n_keys"], "n_atoms": SMALL["n_atoms"],
            },
            run=lambda module, inputs: module(
                q=inputs["q"], c=inputs["c"], p=inputs["p"]
            ),
        ),
        FixtureCase(
            name="atom_attention_encoder",
            build=_build_atom_encoder,
            config={
                "c_token": c_s, "c_atom": c_a, "c_atompair": SMALL["c_atompair"],
                "c_s": c_s, "c_z": c_z, "n_blocks": 2, "n_heads": n_heads,
                "n_queries": 32, "n_keys": 128,
                "n_atoms": SMALL["n_atoms"], "n_tokens": n_tokens,
            },
            run=_run_atom_encoder,
        ),
        FixtureCase(
            name="diffusion_module",
            build=_build_diffusion_module,
            config={
                "sigma_data": 16.0, "c_atom": c_a, "c_atompair": SMALL["c_atompair"],
                "c_token": c_a, "c_s": c_s, "c_z": c_z,
                "atom_encoder_blocks": 2, "atom_encoder_heads": n_heads,
                "transformer_blocks": 2, "transformer_heads": n_heads,
                "atom_decoder_blocks": 2, "atom_decoder_heads": n_heads,
                "n_atoms": SMALL["n_atoms"], "n_tokens": n_tokens,
                "r_max": 32, "s_max": 2,
            },
            run=_run_diffusion_module,
        ),
        FixtureCase(
            name="atom_attention_decoder",
            build=_build_atom_decoder,
            config={
                "c_token": c_s, "c_atom": c_a, "c_atompair": SMALL["c_atompair"],
                "n_blocks": 2, "n_heads": n_heads, "n_queries": SMALL["n_queries"],
                "n_keys": SMALL["n_keys"], "n_atoms": SMALL["n_atoms"],
                "n_tokens": n_tokens,
            },
            run=lambda module, inputs: module(
                atom_to_token_idx=inputs["atom_to_token_idx"],
                a=inputs["a"], q_skip=inputs["q_skip"], c_skip=inputs["c_skip"],
                p_skip=inputs["p_skip"],
            ),
        ),
        FixtureCase(
            # Three blocks, so an error in how blocks are chained (feeding the wrong
            # tensor forward, or reusing block 0's weights) cannot hide.
            name="pairformer_stack",
            build=lambda: (
                PairformerStack(
                    n_blocks=3,
                    n_heads=SMALL["n_heads"],
                    c_z=c_z,
                    c_s=c_s,
                    dropout=0.0,
                ),
                {"s": single(c_s), "z": pair()},
            ),
            config={
                "n_blocks": 3,
                "n_heads": SMALL["n_heads"],
                "c_z": c_z,
                "c_s": c_s,
                # PairformerStack does not forward c_hidden_pair_att, so its blocks use
                # the upstream default of 32 with 4 heads.
                "pair_head_count": 4,
            },
            run=lambda module, inputs: module(
                s=inputs["s"], z=inputs["z"], pair_mask=None
            ),
        ),
    ]
    return cases


def _run_conditioning(module: Any, inputs: dict) -> dict:
    """Drive DiffusionConditioning through both halves the Swift port exposes."""
    pair = module.prepare_cache(inputs["relpe"], inputs["z_trunk"])
    single, pair_out = module(
        t_hat_noise_level=inputs["t_hat_noise_level"],
        relp_feature=inputs["relpe"],
        s_inputs=inputs["s_inputs"],
        s_trunk=inputs["s_trunk"],
        z_trunk=inputs["z_trunk"],
        pair_z=pair,
    )
    return {"pair": pair_out, "single": single}


def _atom_features(n_atoms: int, n_tokens: int, c_z: int, c_s: int, c_atom_in: int):
    """Build a toy atom-attention input, with d_lm/v_lm/pad_info from upstream."""
    from protenix.model.protenix import update_input_feature_dict

    generator = torch.Generator().manual_seed(7)
    # A contiguous atom->token map, roughly n_atoms/n_tokens atoms per token.
    per_token = max(1, n_atoms // n_tokens)
    atom_to_token = torch.clamp(
        torch.arange(n_atoms) // per_token, max=n_tokens - 1
    ).long()
    ref_space_uid = atom_to_token.clone()
    features = {
        "ref_pos": torch.randn(1, n_atoms, 3, generator=generator),
        "ref_space_uid": ref_space_uid.reshape(1, n_atoms),  # batched for feature prep
    }
    update_input_feature_dict(features)
    return atom_to_token, features


def _build_atom_encoder():
    """Construct an AtomAttentionEncoder (has_coords) and its inputs."""
    from protenix.model.modules.transformer import AtomAttentionEncoder

    c_z, c_s, c_a = SMALL["c_z"], SMALL["c_s"], SMALL["c_a"]
    n_atoms, n_tokens = SMALL["n_atoms"], SMALL["n_tokens"]
    # update_input_feature_dict hardcodes these; the encoder must match.
    n_queries, n_keys = 32, 128
    generator = torch.Generator().manual_seed(11)

    module = AtomAttentionEncoder(
        has_coords=True, c_token=c_s, c_atom=c_a, c_atompair=SMALL["c_atompair"],
        c_s=c_s, c_z=c_z, n_blocks=2, n_heads=SMALL["n_heads"],
        n_queries=n_queries, n_keys=n_keys,
    )
    atom_to_token, feats = _atom_features(n_atoms, n_tokens, c_z, c_s, c_a)
    inputs = {
        "atom_to_token_idx": atom_to_token,  # 1-D [N_atom]
        "ref_pos": feats["ref_pos"],
        "ref_charge": torch.randn(1, n_atoms, generator=generator),
        "ref_mask": torch.ones(1, n_atoms),
        "ref_element": torch.randn(1, n_atoms, 128, generator=generator),
        "ref_atom_name_chars": torch.randn(1, n_atoms, 4 * 64, generator=generator),
        # v_lm / mask_trunked are boolean geometry features; the network casts them to
        # float, so they are stored float here to match what the bundle would carry.
        "d_lm": feats["d_lm"],
        "v_lm": feats["v_lm"].float(),
        "mask_trunked": feats["pad_info"]["mask_trunked"].float(),
        "r_l": torch.randn(1, 1, n_atoms, 3, generator=generator),
        "s": torch.randn(1, 1, n_tokens, c_s, generator=generator),
        "z": torch.randn(1, 1, n_tokens, n_tokens, c_z, generator=generator),
    }
    return module, inputs


def _build_atom_decoder():
    """Construct an AtomAttentionDecoder and its inputs."""
    from protenix.model.modules.transformer import AtomAttentionDecoder

    c_s, c_a = SMALL["c_s"], SMALL["c_a"]
    n_atoms, n_tokens = SMALL["n_atoms"], SMALL["n_tokens"]
    n_queries, n_keys = SMALL["n_queries"], SMALL["n_keys"]
    n_blocks = -(-n_atoms // n_queries)
    generator = torch.Generator().manual_seed(13)

    module = AtomAttentionDecoder(
        n_blocks=2, n_heads=SMALL["n_heads"], c_token=c_s, c_atom=c_a,
        c_atompair=SMALL["c_atompair"], n_queries=n_queries, n_keys=n_keys,
    )
    atom_to_token = torch.clamp(
        torch.arange(n_atoms) // max(1, n_atoms // n_tokens), max=n_tokens - 1
    ).long()
    inputs = {
        "atom_to_token_idx": atom_to_token,  # 1-D [N_atom]
        "a": torch.randn(1, n_tokens, c_s, generator=generator),
        "q_skip": torch.randn(1, n_atoms, c_a, generator=generator),
        "c_skip": torch.randn(1, n_atoms, c_a, generator=generator),
        "p_skip": torch.randn(
            1, n_blocks, n_queries, n_keys, SMALL["c_atompair"], generator=generator
        ),
    }
    return module, inputs


def _build_diffusion_module():
    """A whole DiffusionModule at small dims, plus one denoise step's inputs."""
    from protenix.model.modules.diffusion import DiffusionModule
    from protenix.model.modules.embedders import RelativePositionEncoding

    c_z, c_s, c_a = SMALL["c_z"], SMALL["c_s"], SMALL["c_a"]
    c_token = SMALL["c_a"]  # divisible by n_heads
    n_atoms, n_tokens = SMALL["n_atoms"], SMALL["n_tokens"]
    heads = SMALL["n_heads"]
    generator = torch.Generator().manual_seed(17)

    module = DiffusionModule(
        sigma_data=16.0, c_atom=c_a, c_atompair=SMALL["c_atompair"], c_token=c_token,
        c_s=c_s, c_z=c_z, c_s_inputs=c_s,
        atom_encoder={"n_blocks": 2, "n_heads": heads},
        transformer={"n_blocks": 2, "n_heads": heads},
        atom_decoder={"n_blocks": 2, "n_heads": heads},
    )

    atom_to_token, feats = _atom_features(n_atoms, n_tokens, c_z, c_s, c_a)

    # Build the relp one-hot the conditioning consumes, from toy token indices.
    relpe = RelativePositionEncoding(c_z=c_z)
    token_ids = torch.arange(n_tokens)
    feature_dict = {
        "asym_id": torch.zeros(1, n_tokens, dtype=torch.long),
        "residue_index": token_ids.reshape(1, n_tokens).long(),
        "entity_id": torch.zeros(1, n_tokens, dtype=torch.long),
        "token_index": token_ids.reshape(1, n_tokens).long(),
        "sym_id": torch.zeros(1, n_tokens, dtype=torch.long),
    }
    relpe.generate_relp(feature_dict)

    input_feature_dict = {
        "atom_to_token_idx": atom_to_token,
        "ref_pos": feats["ref_pos"],
        "ref_charge": torch.randn(1, n_atoms, generator=generator),
        "ref_mask": torch.ones(1, n_atoms),
        "ref_element": torch.randn(1, n_atoms, 128, generator=generator),
        "ref_atom_name_chars": torch.randn(1, n_atoms, 4 * 64, generator=generator),
        "d_lm": feats["d_lm"],
        "v_lm": feats["v_lm"],
        "pad_info": feats["pad_info"],
        "relp": feature_dict["relp"],
    }
    inputs = {
        "x_noisy": torch.randn(1, 1, n_atoms, 3, generator=generator) * 8,
        "t_hat": torch.rand(1, 1, generator=generator) * 20 + 1,
        "s_inputs": torch.randn(1, n_tokens, c_s, generator=generator),
        "s_trunk": torch.randn(1, n_tokens, c_s, generator=generator),
        "z_trunk": torch.randn(1, n_tokens, n_tokens, c_z, generator=generator),
        # The relp and atom features travel with the module rather than as plain
        # tensors, so they are recorded separately for the Swift side to rebuild.
        "relp": feature_dict["relp"],
        "atom_to_token_idx": atom_to_token,
        "ref_pos": input_feature_dict["ref_pos"],
        "ref_charge": input_feature_dict["ref_charge"],
        "ref_mask": input_feature_dict["ref_mask"],
        "ref_element": input_feature_dict["ref_element"],
        "ref_atom_name_chars": input_feature_dict["ref_atom_name_chars"],
        "d_lm": input_feature_dict["d_lm"],
        "v_lm": input_feature_dict["v_lm"].float(),
        "mask_trunked": feats["pad_info"]["mask_trunked"].float(),
    }
    module._feature_dict = input_feature_dict  # stashed for the run callback
    return module, inputs


def _run_diffusion_module(module: Any, inputs: dict) -> Any:
    """One denoise step through the assembled DiffusionModule."""
    return module(
        x_noisy=inputs["x_noisy"],
        t_hat_noise_level=inputs["t_hat"],
        input_feature_dict=module._feature_dict,
        s_inputs=inputs["s_inputs"],
        s_trunk=inputs["s_trunk"],
        z_trunk=inputs["z_trunk"],
        pair_z=None,
        p_lm=None,
        c_l=None,
    )


def _run_atom_encoder(module: Any, inputs: dict) -> dict:
    """Run AtomAttentionEncoder.forward (has_coords) and name its four outputs."""
    a, q_l, c_l, p_lm = module(
        atom_to_token_idx=inputs["atom_to_token_idx"],
        ref_pos=inputs["ref_pos"],
        ref_charge=inputs["ref_charge"],
        ref_mask=inputs["ref_mask"],
        ref_atom_name_chars=inputs["ref_atom_name_chars"],
        ref_element=inputs["ref_element"],
        d_lm=inputs["d_lm"],
        v_lm=inputs["v_lm"],
        pad_info={"mask_trunked": inputs["mask_trunked"]},
        r_l=inputs["r_l"],
        s=inputs["s"],
        z=inputs["z"],
    )
    return {"a": a, "q_skip": q_l, "c_skip": c_l, "p_skip": p_lm}


def _run_msa_blocks(module: Any, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Run an MSAModule's blocks directly, bypassing raw-feature embedding."""
    pair = z
    msa = m
    for block in module.blocks:
        updated, pair = block(msa, pair, pair_mask=None)
        if updated is not None:
            msa = updated
    return pair


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

    # Snapshot the inputs BEFORE running. Several upstream modules mutate their input
    # in place at inference -- `MSAStack.inference_forward` does `m[start:end] += ...`
    # to bound memory -- so recording `inputs` afterwards would save the OUTPUT under
    # the input's name. The fixture would then ask the port to reproduce f(x) from
    # f(x), which no correct implementation can do.
    recorded = {name: value.detach().clone() for name, value in inputs.items()}

    with torch.no_grad():
        result = case.run(module, inputs) if case.run else module(**inputs)

    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in module.state_dict().items():
        tensors[f"weight.{name}"] = parameter
    for name, value in recorded.items():
        tensors[f"input.{name}"] = value
    tensors.update(_as_tensor_map(result, "output"))

    # An in-place module returns its own input tensor, so the recorded output would
    # alias the (already overwritten) input. Detach a copy so both are independent.
    for name in list(tensors):
        tensors[name] = tensors[name].detach().clone()

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
        "atom_attention_decoder",
        "atom_attention_encoder",
        "atom_transformer",
        "attention_pair_bias_with_s",
        "conditioned_transition_block",
        "diffusion_conditioning",
        "diffusion_module",
        "diffusion_transformer",
        "diffusion_transformer_block",
        "msa_module",
        "msa_pair_weighted_averaging",
        "msa_stack",
        "outer_product_mean",
        "pairformer_block",
        "pairformer_block_scale_up",
        "pairformer_stack",
        "transition",
        "triangle_attention_ending",
        "triangle_attention_starting",
        "triangle_multiplication_incoming",
        "triangle_multiplication_outgoing",
    )


def _unused(value: Mapping[str, object]) -> None:  # pragma: no cover
    """Placeholder kept so `Mapping` stays referenced under TYPE_CHECKING."""
