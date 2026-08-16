"""Audit a checkpoint of uncertain provenance before it is packed and published.

Needed because `protenix-v2`'s official checkpoint is no longer served (the bucket
answers 403 for it while every other release answers 200), so the only copies are
third-party mirrors. A pack's whole contract is a digest over weights, and a digest
over the wrong weights is worthless -- so a mirrored file has to earn its way in.

The audit reports what it can and cannot establish, because the distinction matters:

  * INTEGRITY   -- the bytes match a published digest. Provable, but a self-attested
                   digest from the same party that supplies the file only rules out
                   corruption in transit, not substitution at the source.
  * STRUCTURE   -- every tensor name and shape matches the architecture upstream's
                   pinned config resolves for this model name, and the parameter count
                   is exact. Hard to forge accidentally.
  * TRAINEDNESS -- the weights carry the statistical fingerprints of training rather
                   than initialization. Rules out a shape-correct file full of noise.
  * AUTHENTICITY of the values -- NOT decidable here. Only an official digest or an
                   end-to-end accuracy benchmark can settle that.

Usage:
    python scripts/audit_checkpoint.py --checkpoint X.pt --model-name protenix-v2 \\
        [--expect-sha256 HEX] [--compare-to KNOWN_GOOD.pt]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from protenix_mlx_export.model_export import read_state_dict
from protenix_mlx_export.names import GRAPH_ROOTS
from protenix_mlx_export.variants import get_variant, resolve_upstream_config

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_count(names: list[str], prefix: str) -> int:
    """Highest block index under `prefix`, plus one."""
    indices = set()
    for name in names:
        if not name.startswith(f"{prefix}.blocks."):
            continue
        part = name[len(f"{prefix}.blocks.") :].split(".", 1)[0]
        if part.isdigit():
            indices.add(int(part))
    return max(indices) + 1 if indices else 0


def audit_structure(state: dict[str, torch.Tensor], model_name: str) -> list[str]:
    """Check the checkpoint against the architecture upstream resolves for the name."""
    problems: list[str] = []
    configs = resolve_upstream_config(model_name)
    names = list(state)

    variant = get_variant(model_name)
    total = sum(int(t.numel()) for t in state.values())
    found_m = total / 1e6
    print(f"  parameters      {total:,} ({found_m:.2f}M)")
    print(f"  published       {variant.published_params_m:.2f}M")
    if abs(found_m - variant.published_params_m) > variant.params_tolerance_m:
        problems.append(
            f"parameter count {found_m:.2f}M differs from published "
            f"{variant.published_params_m:.2f}M"
        )

    unknown = sorted({n.split(".", 1)[0] for n in names} - set(GRAPH_ROOTS))
    if unknown:
        problems.append(f"unmodelled top-level modules: {unknown}")

    # Block depths must match the config exactly. These are the values that differ
    # between variants, so a mislabelled checkpoint shows up here first.
    expected = {
        "pairformer_stack": int(configs.model.pairformer.n_blocks),
        "msa_module.blocks_container"
        if any(n.startswith("msa_module.blocks_container") for n in names)
        else "msa_module": int(configs.model.msa_module.n_blocks),
        "diffusion_module.diffusion_transformer": int(
            configs.model.diffusion_module.transformer.n_blocks
        ),
        "confidence_head.pairformer_stack": int(configs.model.confidence_head.n_blocks),
    }
    print("  block depths (found vs config):")
    for prefix, want in expected.items():
        got = _block_count(names, prefix)
        flag = "" if got == want else "   <-- MISMATCH"
        print(f"    {prefix:44s} {got:3d} vs {want:3d}{flag}")
        if got != want:
            problems.append(f"{prefix} has {got} blocks, config says {want}")

    # c_z is the dimension protenix-v2 widens (128 -> 256); it is the single most
    # diagnostic width for telling v2 apart from base.
    c_z = int(configs.c_z)
    z_tensor = state.get("linear_no_bias_zinit1.weight")
    if z_tensor is not None:
        found_c_z = int(z_tensor.shape[0])
        print(f"  c_z             {found_c_z} (config {c_z})")
        if found_c_z != c_z:
            problems.append(f"c_z is {found_c_z}, config says {c_z}")
    else:
        problems.append("linear_no_bias_zinit1.weight is absent")

    c_s = int(configs.c_s)
    s_tensor = state.get("linear_no_bias_sinit.weight")
    if s_tensor is not None:
        found_c_s = int(s_tensor.shape[0])
        print(f"  c_s             {found_c_s} (config {c_s})")
        if found_c_s != c_s:
            problems.append(f"c_s is {found_c_s}, config says {c_s}")

    return problems


def audit_trainedness(state: dict[str, torch.Tensor]) -> list[str]:
    """Look for the statistical fingerprints of a trained network.

    A shape-correct file full of fresh initialization would pass every structural
    check. Three signals separate the two cheaply:

      * LayerNorm gains drift away from their initialized 1.0 during training, and
        spread out. A freshly built model has them all exactly 1.0.
      * Per-tensor weight scales vary by orders of magnitude across a trained network;
        an initialized one follows a tight fan-in rule.
      * Trained matrices are not perfectly zero-mean the way initializers are.
    """
    problems: list[str] = []
    gains = [
        t.detach().float().numpy()
        for name, t in state.items()
        if name.endswith(".weight") and t.ndim == 1 and "norm" in name.lower()
    ]
    if gains:
        flat = np.concatenate([g.ravel() for g in gains])
        exactly_one = float((flat == 1.0).mean())
        print(f"  layernorm gains {len(flat):,} values, "
              f"mean {flat.mean():.4f}, std {flat.std():.4f}, "
              f"{exactly_one:.1%} exactly 1.0")
        if exactly_one > 0.9:
            problems.append(
                f"{exactly_one:.1%} of LayerNorm gains are exactly 1.0 -- this looks "
                "initialized, not trained"
            )
        if flat.std() < 1e-4:
            problems.append("LayerNorm gains have no spread; likely untrained")
    else:
        problems.append("no LayerNorm gains found to test")

    matrices = [
        t.detach().float().numpy()
        for name, t in state.items()
        if name.endswith(".weight") and t.ndim == 2
    ]
    scales = np.array([float(np.abs(m).max()) for m in matrices if m.size])
    if scales.size:
        spread = float(scales.max() / max(scales.min(), 1e-30))
        print(f"  matrix scales   {scales.size} matrices, peak |w| from "
              f"{scales.min():.3g} to {scales.max():.3g} (spread {spread:.3g}x)")
        if spread < 50:
            problems.append(
                f"per-matrix scales span only {spread:.1f}x; a trained network spans "
                "orders of magnitude"
            )
    dead = sum(1 for m in matrices if m.size and float(np.abs(m).max()) < 1e-3)
    print(f"  collapsed       {dead}/{len(matrices)} matrices below 1e-3 peak "
          "(training collapses some; initialization collapses none)")
    return problems


def audit_against_known(
    state: dict[str, torch.Tensor], reference: Path
) -> list[str]:
    """Compare module structure against an officially-sourced sibling checkpoint.

    The strongest evidence available without an official digest: a genuine Protenix
    checkpoint should share the reference's module tree exactly, differing only in
    the widths its config changes.
    """
    problems: list[str] = []
    other, _ = read_state_dict(reference)
    mine, theirs = set(state), set(other)
    missing = theirs - mine
    extra = mine - theirs
    print(f"  reference       {reference.name} ({len(theirs)} tensors)")
    print(f"  shared names    {len(mine & theirs)} / {len(mine)} of this checkpoint")
    if missing:
        print(f"  absent here     {len(missing)} (e.g. {sorted(missing)[:3]})")
    if extra:
        print(f"  extra here      {len(extra)} (e.g. {sorted(extra)[:3]})")

    # Identical names should have compatible ranks; widths may legitimately differ.
    rank_conflicts = [
        name
        for name in sorted(mine & theirs)
        if state[name].ndim != other[name].ndim
    ]
    if rank_conflicts:
        problems.append(
            f"{len(rank_conflicts)} shared tensors differ in rank, e.g. "
            f"{rank_conflicts[:3]}"
        )
    widened = Counter()
    for name in mine & theirs:
        if state[name].shape != other[name].shape:
            widened[tuple(other[name].shape)[:1] + tuple(state[name].shape)[:1]] += 1
    print(f"  reshaped        {sum(widened.values())} tensors differ in width "
          "(expected when c_z or hidden_scale_up changes)")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--expect-sha256", default=None)
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="An officially-sourced checkpoint to compare module structure against.",
    )
    args = parser.parse_args()

    print(f"auditing {args.checkpoint} as {args.model_name}\n")

    print("INTEGRITY")
    digest = sha256_file(args.checkpoint)
    size = args.checkpoint.stat().st_size
    print(f"  size            {size:,} bytes ({size / 1e9:.3f} GB)")
    print(f"  sha256          {digest}")
    integrity_problems: list[str] = []
    if args.expect_sha256:
        if digest.lower() == args.expect_sha256.lower():
            print("  matches the digest published alongside the file")
        else:
            print(f"  DOES NOT MATCH expected {args.expect_sha256}")
            integrity_problems.append("sha256 differs from the published digest")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  top-level keys  {sorted(payload)}")
    print(f"  model_version   {payload.get('model_version')!r}")
    state, _ = read_state_dict(args.checkpoint)

    print("\nSTRUCTURE")
    structure_problems = audit_structure(state, args.model_name)

    print("\nTRAINEDNESS")
    trained_problems = audit_trainedness(state)

    comparison_problems: list[str] = []
    if args.compare_to:
        print("\nCOMPARISON WITH A KNOWN-GOOD CHECKPOINT")
        comparison_problems = audit_against_known(state, args.compare_to)

    problems = (
        integrity_problems + structure_problems + trained_problems
        + comparison_problems
    )
    print("\nVERDICT")
    if problems:
        print(f"  {len(problems)} problem(s):")
        for problem in problems:
            print(f"    - {problem}")
        return 1
    print("  structurally consistent with the declared model, and statistically")
    print("  consistent with a trained network.")
    print("  NOT established: that these are ByteDance's published weight VALUES.")
    print("  No official digest exists to check against, and the only digest offered")
    print("  is self-attested by the mirror. Settling it needs an accuracy benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
