"""Compare an exported pack against the checkpoint it came from.

Answers the only question that matters about an export: if the Swift runtime loads
these arrays and undoes the packing, does it get the model back? Reports per-tensor
Pearson r and relative error against the original float32 weights, worst cases first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from protenix_mlx_export.model_export import ModelConfiguration, read_state_dict
from protenix_mlx_export.schema import ArtifactManifest


def _dequantize(
    packed: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int,
    bits: int,
    logical_width: int,
) -> np.ndarray:
    """Undo the exporter's affine quantization exactly as the runtime must."""
    restored = mx.dequantize(
        packed, scales=scales, biases=biases, group_size=group_size, bits=bits,
        mode="affine",
    )
    mx.eval(restored)
    return np.asarray(restored, dtype=np.float32)[:, :logical_width]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _relative_error(reference: np.ndarray, value: np.ndarray) -> float:
    scale = np.abs(reference).max()
    if scale == 0:
        return float(np.abs(value).max())
    return float(np.abs(value - reference).max() / scale)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--worst", type=int, default=10)
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.05,
        help="Allowed error as a fraction of a tensor's own peak weight. 0.05 is a "
             "loose bound on affine int8 at group 64; dense packs sit far below it.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Error floor, as a fraction of the model's characteristic weight scale, "
             "for tensors too small for a relative bound to mean anything.",
    )
    args = parser.parse_args()

    manifest = ArtifactManifest.read(args.pack / "manifest.json")
    configuration = ModelConfiguration.read(args.pack / "config.json")
    arrays = mx.load(str(args.pack / "model.safetensors"))
    reference, _ = read_state_dict(args.checkpoint)

    quantization = manifest.quantization
    specs = {spec.name: spec for spec in manifest.tensors}
    results: list[tuple[float, float, str, str]] = []
    covered: set[str] = set()

    for name, tensor in reference.items():
        original = tensor.detach().cpu().float().numpy()
        module = name.removesuffix(".weight")
        is_quantized = (
            quantization is not None
            and name.endswith(".weight")
            and f"{module}.scales" in arrays
        )
        if is_quantized:
            spec = specs[name]
            if spec.logical_shape is None:
                message = f"quantized tensor {name} has no logical_shape"
                raise ValueError(message)
            restored = _dequantize(
                arrays[name],
                arrays[f"{module}.scales"],
                arrays[f"{module}.biases"],
                group_size=int(quantization["group_size"]),
                bits=int(quantization["bits"]),
                logical_width=int(spec.logical_shape[1]),
            )
            covered.update({name, f"{module}.scales", f"{module}.biases"})
            kind = "int8"
        else:
            if name not in arrays:
                message = f"pack is missing tensor {name}"
                raise ValueError(message)
            restored = np.asarray(arrays[name].astype(mx.float32), dtype=np.float32)
            covered.add(name)
            kind = specs[name].dtype

        if restored.shape != original.shape:
            message = (
                f"{name}: pack shape {restored.shape} != checkpoint {original.shape}"
            )
            raise ValueError(message)
        if not np.isfinite(restored).all():
            message = f"{name}: pack contains non-finite values"
            raise ValueError(message)
        results.append((
            _pearson(original, restored),
            _relative_error(original, restored),
            name,
            kind,
            float(np.abs(original).max()),
        ))

    unexpected = sorted(set(arrays) - covered)
    if unexpected:
        message = f"pack holds tensors absent from the checkpoint: {unexpected}"
        raise ValueError(message)

    results.sort()
    print(f"model     {configuration.model_name}")
    print(f"pack      {args.pack.name}")
    print(f"tensors   {len(results)} compared, {len(arrays)} arrays in pack")
    print(f"params    {configuration.parameter_count / 1e6:.2f}M "
          f"({configuration.quantized_matrix_count} matrices quantized)")

    # Correlation alone is the wrong gate, in both directions.
    #
    # Some released variants ship tensors that training collapsed to the noise floor:
    # protenix_mini's `pairformer_stack.blocks.0.attention_pair_bias.linear_nobias_z`
    # has std 1e-5 against tiny's 0.33, with 64-wide groups spanning 1e-36. Quantizing
    # near-zero weights reconstructs them with near-zero ABSOLUTE error but poor
    # correlation -- there is no signal to correlate with -- so r alone condemns a pack
    # that is numerically fine.
    #
    # The gate is therefore the familiar atol + rtol pair. `rtol` covers tensors that
    # carry real magnitude; `atol`, scaled by the model's characteristic weight, covers
    # the dead ones without excusing genuine corruption anywhere.
    #
    # The characteristic scale is a MEDIAN, not a max: `confidence_head.upper_bins`
    # ends in a 1e6 sentinel, and normalizing by that makes every real weight in the
    # model look negligible.
    magnitudes = np.array([row[4] for row in results])
    scale = float(np.median(magnitudes[magnitudes > 0]))
    correlations = np.array([row[0] for row in results])

    checked = []
    for pearson, relative, name, kind, magnitude in results:
        error = relative * magnitude
        tolerance = args.rtol * magnitude + args.atol * scale
        checked.append((error > tolerance, error, tolerance, pearson, relative,
                        magnitude, name, kind))

    failures = [row for row in checked if row[0]]
    print(f"scale     {scale:.4g} (median peak weight, robust to bin-edge sentinels)")
    print(f"pearson   min {correlations.min():.6f}  mean {correlations.mean():.6f}")
    print(f"tolerance {args.rtol:g}*max|w| + {args.atol:g}*scale = "
          f"{args.atol * scale:.3e} floor")

    print(f"\nworst {min(args.worst, len(results))} by correlation:")
    for _, error, tolerance, pearson, relative, magnitude, name, kind in (
        sorted(checked, key=lambda row: row[3])[: args.worst]
    ):
        headroom = tolerance / error if error > 0 else float("inf")
        print(f"  r={pearson:.6f}  relerr={relative:.3e}  max|w|={magnitude:.3e}  "
              f"{headroom:6.1f}x within tolerance  [{kind}] {name}")

    if failures:
        print(f"\nFAIL: {len(failures)} tensors exceed tolerance")
        for _, error, tolerance, _, _, _, name, _ in failures[: args.worst]:
            print(f"  {name}: error {error:.3e} > tolerance {tolerance:.3e}")
        return 1
    print(f"\nOK: all {len(results)} tensors within tolerance "
          f"(closest {min(row[2] / row[1] for row in checked if row[1] > 0):.1f}x margin)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
