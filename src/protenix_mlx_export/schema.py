"""Stable JSON schemas shared by the offline exporter and the Swift runtime.

Any change to these shapes is a change to the artifact contract: packs already
published carry ``schema_version`` and the Swift loader refuses one it does not
recognize, so bump :data:`SCHEMA_VERSION` rather than widening a field in place.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = 1
SHA256_HEX_LENGTH = 64

#: Upstream revision these artifacts are exported from. Unlike Boltz, a Protenix
#: checkpoint carries NO architecture metadata -- only ``{"model": state_dict,
#: "model_version": str}`` -- so the graph shape comes from the pinned config tree
#: vendored under ``src/_protenix_upstream``. That makes the commit part of the
#: contract rather than a provenance note: a different commit can resolve the same
#: model name to different dimensions.
PROTENIX_SOURCE_REVISION = "main"
PROTENIX_SOURCE_COMMIT = "4c355be4553512f72453ecbfb65e69f4c35d1413"


class ManifestValidationError(ValueError):
    """Raised when an artifact manifest violates the supported schema."""


class ArtifactKind(StrEnum):
    """Kinds of directories understood by Protenix MLX."""

    MODEL = "model"
    FEATURES = "features"
    FIXTURE = "fixture"


@dataclass(frozen=True)
class TensorSpec:
    """Serializable declaration for one tensor in an artifact.

    ``logical_shape``/``physical_shape`` are set only on affine-int8 matrices, whose
    input width is zero-padded up to the quantization group size. A dense tensor
    needs no padding, so its ``shape`` is its own single source of truth and both
    fields stay ``None``.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str = "model.safetensors"
    logical_shape: tuple[int, ...] | None = None
    physical_shape: tuple[int, ...] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TensorSpec:
        """Decode a tensor declaration from JSON-compatible values."""
        return cls(
            name=str(value["name"]),
            shape=tuple(int(dimension) for dimension in value["shape"]),
            dtype=str(value["dtype"]),
            shard=str(value.get("shard", "model.safetensors")),
            logical_shape=(
                tuple(int(dimension) for dimension in value["logical_shape"])
                if value.get("logical_shape") is not None
                else None
            ),
            physical_shape=(
                tuple(int(dimension) for dimension in value["physical_shape"])
                if value.get("physical_shape") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ArtifactManifest:
    """Top-level manifest for a model, feature bundle, or parity fixture."""

    supported_schema_version: ClassVar[int] = SCHEMA_VERSION

    schema_version: int
    artifact_kind: ArtifactKind
    model_name: str
    source_revision: str
    source_commit: str
    source_checkpoint_sha256: str | None
    tensors: tuple[TensorSpec, ...]
    quantization: dict[str, int | str] | None = None

    def __post_init__(self) -> None:
        """Validate invariants that both exporters and runtimes rely on."""
        if self.schema_version != self.supported_schema_version:
            message = (
                f"unsupported schema version {self.schema_version}, "
                f"expected {self.supported_schema_version}"
            )
            raise ManifestValidationError(message)
        if not self.model_name:
            message = "manifest must name the model variant it was exported from"
            raise ManifestValidationError(message)
        digest = self.source_checkpoint_sha256
        if digest is not None and len(digest) != SHA256_HEX_LENGTH:
            message = f"source checkpoint digest is not a sha256: {digest!r}"
            raise ManifestValidationError(message)
        names = [spec.name for spec in self.tensors]
        if len(names) != len(set(names)):
            message = "manifest declares the same tensor name twice"
            raise ManifestValidationError(message)
        if names != sorted(names):
            message = "manifest tensors must be sorted by name"
            raise ManifestValidationError(message)

    @classmethod
    def model_v1(
        cls,
        *,
        model_name: str,
        source_checkpoint_sha256: str,
        tensors: tuple[TensorSpec, ...],
        quantization: dict[str, int | str] | None,
    ) -> ArtifactManifest:
        """Build a model manifest at the current schema version."""
        return cls(
            schema_version=SCHEMA_VERSION,
            artifact_kind=ArtifactKind.MODEL,
            model_name=model_name,
            source_revision=PROTENIX_SOURCE_REVISION,
            source_commit=PROTENIX_SOURCE_COMMIT,
            source_checkpoint_sha256=source_checkpoint_sha256,
            tensors=tensors,
            quantization=quantization,
        )

    def write(self, path: Path) -> None:
        """Write deterministic manifest JSON."""
        payload = asdict(self)
        payload["artifact_kind"] = str(self.artifact_kind)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> ArtifactManifest:
        """Read and validate a manifest written by :meth:`write`."""
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            schema_version=int(value["schema_version"]),
            artifact_kind=ArtifactKind(value["artifact_kind"]),
            model_name=str(value["model_name"]),
            source_revision=str(value["source_revision"]),
            source_commit=str(value["source_commit"]),
            source_checkpoint_sha256=value["source_checkpoint_sha256"],
            tensors=tuple(
                TensorSpec.from_dict(spec) for spec in value["tensors"]
            ),
            quantization=value.get("quantization"),
        )
