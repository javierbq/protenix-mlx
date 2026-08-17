"""Convert a released Protenix checkpoint into an MLX Swift artifact directory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from safetensors.numpy import save_file
from safetensors.torch import save_file as save_torch_file

from protenix_mlx_export.names import (
    normalize_state_dict,
    unrecognized_roots,
)
from protenix_mlx_export.quantization import (
    INT8_BITS,
    INT8_GROUP_SIZE,
    is_quantizable,
    quantize_affine_int8,
)
from protenix_mlx_export.schema import (
    PROTENIX_SOURCE_COMMIT,
    PROTENIX_SOURCE_REVISION,
    SCHEMA_VERSION,
    ArtifactManifest,
    TensorSpec,
)
from protenix_mlx_export.variants import get_variant, resolve_upstream_config

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


#: Manifest dtype spellings. These must match the Swift loader's `dtypeName` exactly:
#: it compares the string per tensor and throws on any mismatch.
_TORCH_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

_MILLION = 1e6


class Precision(StrEnum):
    """Weight representation for an exported model artifact.

    ``INT8`` packs every eligible matrix with MLX affine int8 (group 64) and stores
    everything else as float16. ``FLOAT16`` and ``BFLOAT16`` store every float tensor
    dense at that width, matrices included, and emit no quantization block.

    A pack is single-dtype on purpose: MLX promotes a mixed float16/bfloat16
    operation to float32, so a pack that mixed widths would silently change both
    numerics and speed relative to either pure pack.
    """

    INT8 = "int8"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    @property
    def torch_dtype(self) -> torch.dtype:
        """The torch dtype float tensors are stored at under this precision."""
        return torch.bfloat16 if self is Precision.BFLOAT16 else torch.float16


@dataclass(frozen=True)
class ModelConfiguration:
    """The architecture contract a runtime needs to rebuild the graph.

    Protenix checkpoints carry no hyperparameters, so every value here is resolved
    from the pinned upstream config tree by model name and frozen at export time.
    ``model`` is upstream's ``configs.model`` subtree verbatim -- copied rather than
    filtered so it stays diffable against upstream. It therefore also contains
    training-only knobs (``dropout``, ``blocks_per_ckpt``, ``use_fine_grained_checkpoint``)
    which an inference runtime must ignore.
    """

    schema_version: int
    model_name: str
    source_revision: str
    source_commit: str

    # Global widths every submodule is sized against.
    c_s: int
    c_z: int
    c_s_inputs: int
    c_atom: int
    c_atompair: int
    c_token: int

    #: Recycling iterations. Upstream's default for the variant, not a hard limit.
    n_cycle: int
    #: Default diffusion steps for the variant (5 for mini/tiny, 200 for base/v2).
    n_diffusion_steps: int

    model: dict[str, Any]
    sample_diffusion: dict[str, Any]
    inference_noise_scheduler: dict[str, Any]

    #: Filled in by the exporter from the checkpoint actually read.
    parameter_count: int = 0
    quantized_matrix_count: int = 0
    graph_roots: tuple[str, ...] = field(default_factory=tuple)

    #: ``"official"`` when the checkpoint came from ByteDance's own bucket, otherwise
    #: the mirror it came from. Carried into the artifact so a pack built from a
    #: mirrored checkpoint stays identifiable as one after it leaves this machine --
    #: the manifest's digest attests to the bytes, not to where they came from.
    checkpoint_provenance: str = "official"
    checkpoint_source_url: str = ""

    @classmethod
    def from_model_name(cls, model_name: str) -> ModelConfiguration:
        """Resolve one variant's architecture from the pinned config tree."""
        variant = get_variant(model_name)
        configs = resolve_upstream_config(model_name)
        model = _json_mapping(configs.model)
        sample_diffusion = _json_mapping(configs.sample_diffusion)
        noise_scheduler = _json_mapping(configs.inference_noise_scheduler)
        return cls(
            schema_version=SCHEMA_VERSION,
            model_name=model_name,
            source_revision=PROTENIX_SOURCE_REVISION,
            source_commit=PROTENIX_SOURCE_COMMIT,
            c_s=int(configs.c_s),
            c_z=int(configs.c_z),
            c_s_inputs=int(configs.c_s_inputs),
            c_atom=int(configs.c_atom),
            c_atompair=int(configs.c_atompair),
            c_token=int(configs.c_token),
            n_cycle=int(configs.model.N_cycle),
            n_diffusion_steps=int(configs.sample_diffusion.N_step),
            model=model,
            sample_diffusion=sample_diffusion,
            inference_noise_scheduler=noise_scheduler,
            checkpoint_provenance=variant.provenance,
            checkpoint_source_url=variant.url,
        )

    def with_checkpoint_facts(
        self,
        *,
        parameter_count: int,
        quantized_matrix_count: int,
        graph_roots: tuple[str, ...],
    ) -> ModelConfiguration:
        """Return a copy carrying what the checkpoint itself turned out to hold."""
        values = asdict(self)
        values["parameter_count"] = parameter_count
        values["quantized_matrix_count"] = quantized_matrix_count
        values["graph_roots"] = graph_roots
        return ModelConfiguration(**values)

    def write(self, path: Path) -> None:
        """Write deterministic configuration JSON."""
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> ModelConfiguration:
        """Read a configuration written by :meth:`write`."""
        value = json.loads(path.read_text(encoding="utf-8"))
        value["graph_roots"] = tuple(value.get("graph_roots", ()))
        return cls(**value)


def _is_mapping(value: object) -> bool:
    """Accept plain mappings and ml_collections' mapping-compatible ConfigDict."""
    return isinstance(value, Mapping) or callable(getattr(value, "items", None))


def _json_mapping(value: object) -> dict[str, Any]:
    """Copy a nested config subtree into deterministic JSON-compatible data."""
    if not _is_mapping(value):
        message = f"config value is not a mapping: {type(value)}"
        raise TypeError(message)
    return {
        str(key): _json_value(str(key), item)
        for key, item in value.items()  # type: ignore[union-attr]
    }


def _json_value(name: str, value: object) -> object:
    """Normalize one config value for stable JSON output."""
    if _is_mapping(value):
        return _json_mapping(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Iterable):
        return [
            _json_value(f"{name}[{index}]", item) for index, item in enumerate(value)
        ]
    message = f"config value {name} is not JSON serializable: {type(value)}"
    raise TypeError(message)


def _narrows_without_overflow(value: Tensor, dtype: torch.dtype) -> bool:
    """Whether every finite element of `value` survives a cast to `dtype`.

    Protenix stores non-learned constants as parameters alongside real weights --
    `confidence_head.upper_bins` ends in a 1e6 sentinel standing in for "no upper
    bound". Narrowing that to float16 yields `inf`, which happens to still compare
    correctly in the bucketize it feeds but poisons any arithmetic downstream. Rather
    than rewrite a released model's constants or depend on inf behaving, the exporter
    keeps such tensors at float32 and declares that per-tensor in the manifest.
    """
    if not value.dtype.is_floating_point:
        return True
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return True
    return bool(torch.isfinite(finite.to(dtype)).all())


def _numpy_parameter(tensor: Tensor) -> np.ndarray:
    """Return one contiguous numpy array, floats narrowed to float16 when safe."""
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.to(torch.float16)
    if not value.dtype.is_floating_point:
        return value.numpy()
    if not _narrows_without_overflow(value, torch.float16):
        return value.to(torch.float32).numpy()
    return value.numpy().astype(np.float16, copy=False)


def _dense_parameter(tensor: Tensor, dtype: torch.dtype) -> Tensor:
    """Return one contiguous tensor, floats narrowed to the pack's dtype when safe."""
    value = tensor.detach().cpu().contiguous()
    if not value.dtype.is_floating_point:
        return value
    if not _narrows_without_overflow(value, dtype):
        return value.to(torch.float32)
    return value.to(dtype)


def read_state_dict(checkpoint: Path) -> tuple[dict[str, Tensor], str | None]:
    """Load a Protenix checkpoint's parameters under normalized runtime names."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        message = (
            f"{checkpoint.name} is not a Protenix checkpoint: expected a mapping with "
            "a 'model' key"
        )
        raise ValueError(message)
    state = normalize_state_dict(payload["model"])
    unknown = unrecognized_roots(state)
    if unknown:
        message = (
            "checkpoint contains modules this port does not model: "
            f"{', '.join(unknown)}"
        )
        raise ValueError(message)
    version = payload.get("model_version")
    return state, str(version) if version is not None else None


def _check_parameter_count(model_name: str, state: Mapping[str, Tensor]) -> int:
    """Compare the checkpoint's size against upstream's published figure."""
    total = sum(int(tensor.numel()) for tensor in state.values())
    variant = get_variant(model_name)
    published = variant.published_params_m
    found = total / _MILLION
    if abs(found - published) > variant.params_tolerance_m:
        message = (
            f"{model_name} holds {found:.2f}M parameters but upstream publishes "
            f"{published:.2f}M; the checkpoint and the model name disagree"
        )
        raise ValueError(message)
    return total


def _export_dense(
    state: Mapping[str, Tensor],
    *,
    output: Path,
    model_name: str,
    source_checkpoint_sha256: str,
    precision: Precision,
) -> tuple[ArtifactManifest, int]:
    """Write an unquantized artifact directory at one uniform float width."""
    dtype = precision.torch_dtype
    tensors: dict[str, Tensor] = {}
    specs: list[TensorSpec] = []
    for name, tensor in state.items():
        value = _dense_parameter(tensor, dtype)
        tensors[name] = value
        specs.append(
            TensorSpec(
                name=name,
                shape=tuple(int(dimension) for dimension in value.shape),
                dtype=_TORCH_DTYPE_NAMES[value.dtype],
            ),
        )

    output.mkdir(parents=True, exist_ok=True)
    # safetensors.numpy cannot round-trip bfloat16 -- numpy has no such dtype -- so
    # dense packs go through the torch backend regardless of width.
    save_torch_file(dict(sorted(tensors.items())), output / "model.safetensors")
    manifest = ArtifactManifest.model_v1(
        model_name=model_name,
        source_checkpoint_sha256=source_checkpoint_sha256,
        tensors=tuple(sorted(specs, key=lambda spec: spec.name)),
        quantization=None,
    )
    manifest.write(output / "manifest.json")
    return manifest, 0


def _export_int8(
    state: Mapping[str, Tensor],
    *,
    output: Path,
    model_name: str,
    source_checkpoint_sha256: str,
) -> tuple[ArtifactManifest, int]:
    """Write an affine-int8 artifact directory."""
    arrays: dict[str, np.ndarray] = {}
    specs: list[TensorSpec] = []
    quantized_count = 0
    for name, tensor in state.items():
        shape = tuple(int(dimension) for dimension in tensor.shape)
        if is_quantizable(name, shape):
            quantized = quantize_affine_int8(_numpy_parameter(tensor))
            quantized_count += 1
            module_name = name.removesuffix(".weight")
            packed = {
                f"{module_name}.weight": quantized.weight,
                f"{module_name}.scales": quantized.scales,
                f"{module_name}.biases": quantized.biases,
            }
            for packed_name, array in packed.items():
                arrays[packed_name] = array
                is_weight = packed_name.endswith(".weight")
                specs.append(
                    TensorSpec(
                        name=packed_name,
                        shape=tuple(int(value) for value in array.shape),
                        dtype=str(array.dtype),
                        logical_shape=quantized.logical_shape if is_weight else None,
                        physical_shape=quantized.physical_shape if is_weight else None,
                    ),
                )
        else:
            array = _numpy_parameter(tensor)
            arrays[name] = array
            specs.append(
                TensorSpec(
                    name=name,
                    shape=tuple(int(value) for value in array.shape),
                    dtype=str(array.dtype),
                ),
            )

    output.mkdir(parents=True, exist_ok=True)
    save_file(dict(sorted(arrays.items())), output / "model.safetensors")
    manifest = ArtifactManifest.model_v1(
        model_name=model_name,
        source_checkpoint_sha256=source_checkpoint_sha256,
        tensors=tuple(sorted(specs, key=lambda spec: spec.name)),
        quantization={
            "bits": INT8_BITS,
            "group_size": INT8_GROUP_SIZE,
            "mode": "affine",
        },
    )
    manifest.write(output / "manifest.json")
    return manifest, quantized_count


def _assert_matrix_invariant(state: Mapping[str, Tensor]) -> None:
    """Fail if the rank-2 / `.weight` correspondence this port assumes breaks.

    :func:`~protenix_mlx_export.quantization.is_quantizable` decides what to pack by
    shape and suffix alone. That is exact for every released checkpoint, but it is an
    assumption about the graph, not a guarantee -- so check it here, where a violation
    is one clear error rather than a pack whose matmuls are silently mis-typed.
    """
    rank2_not_weight = sorted(
        name
        for name, tensor in state.items()
        if tensor.ndim == 2 and not name.endswith(".weight")  # noqa: PLR2004
    )
    if rank2_not_weight:
        message = (
            "checkpoint has rank-2 parameters that are not named '.weight', so the "
            f"quantization rule cannot classify them: {rank2_not_weight}"
        )
        raise ValueError(message)
    high_rank = sorted(
        f"{name}{tuple(tensor.shape)}"
        for name, tensor in state.items()
        if tensor.ndim > 2  # noqa: PLR2004
        and not name.startswith("confidence_head.")
    )
    if high_rank:
        message = (
            "checkpoint has rank-3+ parameters outside the confidence head, which "
            f"this port stores dense without having modelled them: {high_rank}"
        )
        raise ValueError(message)


def export_checkpoint(
    *,
    checkpoint: Path,
    output: Path,
    model_name: str,
    precision: Precision = Precision.INT8,
) -> ArtifactManifest:
    """Export one Protenix checkpoint into an MLX artifact directory."""
    if not checkpoint.is_file():
        message = f"checkpoint does not exist: {checkpoint}"
        raise FileNotFoundError(message)
    # Resolve the architecture BEFORE the slow tensor work, so a bad model name fails
    # in a second rather than after a multi-gigabyte load.
    configuration = ModelConfiguration.from_model_name(model_name)

    # A mirrored checkpoint has a digest known from outside the file, so pin it before
    # doing anything else: hashing is far cheaper than loading two gigabytes, and for
    # a mirrored file "these are not the bytes that were audited" is the most useful
    # thing that can be said, whatever else is also wrong with it.
    #
    # This cannot establish that the weights are ByteDance's -- no official checksum
    # exists for any Protenix checkpoint -- only that the pack was built from the same
    # bytes the audit ran against.
    digest = _sha256(checkpoint)
    variant = get_variant(model_name)
    if variant.expected_sha256 and digest != variant.expected_sha256:
        message = (
            f"{checkpoint.name} has sha256 {digest} but {model_name} is pinned to "
            f"{variant.expected_sha256}; this is not the checkpoint that was audited"
        )
        raise ValueError(message)

    state, _version = read_state_dict(checkpoint)
    _assert_matrix_invariant(state)
    parameter_count = _check_parameter_count(model_name, state)

    if precision is Precision.INT8:
        manifest, quantized_count = _export_int8(
            state,
            output=output,
            model_name=model_name,
            source_checkpoint_sha256=digest,
        )
    else:
        manifest, quantized_count = _export_dense(
            state,
            output=output,
            model_name=model_name,
            source_checkpoint_sha256=digest,
            precision=precision,
        )

    roots = tuple(sorted({name.split(".", maxsplit=1)[0] for name in state}))
    configuration.with_checkpoint_facts(
        parameter_count=parameter_count,
        quantized_matrix_count=quantized_count,
        graph_roots=roots,
    ).write(output / "config.json")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
