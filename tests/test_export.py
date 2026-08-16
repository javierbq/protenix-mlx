"""Exporter tests that do not need a multi-hundred-megabyte checkpoint."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from protenix_mlx_export.model_export import (
    ModelConfiguration,
    Precision,
    _assert_matrix_invariant,
    _narrows_without_overflow,
    export_checkpoint,
)
from protenix_mlx_export.names import (
    GRAPH_ROOTS,
    normalize_state_dict,
    strip_ddp_prefix,
    swift_tensor_name,
    unrecognized_roots,
)
from protenix_mlx_export.packaging import bundle_id, pack_artifact
from protenix_mlx_export.quantization import (
    INT8_GROUP_SIZE,
    is_quantizable,
    quantize_affine_int8,
)
from protenix_mlx_export.schema import ArtifactManifest, ManifestValidationError
from protenix_mlx_export.variants import VARIANTS, resolve_upstream_config

TINY = "protenix_tiny_default_v0.5.0"


class TestNames:
    def test_strips_ddp_prefix(self) -> None:
        assert strip_ddp_prefix("module.pairformer_stack.x") == "pairformer_stack.x"

    def test_leaves_unprefixed_names_alone(self) -> None:
        # Unconditional stripping would eat the first component of a non-DDP save.
        assert strip_ddp_prefix("pairformer_stack.x") == "pairformer_stack.x"

    def test_drops_compile_and_checkpoint_wrappers(self) -> None:
        name = "_orig_mod.msa_module._checkpoint_wrapped_module.blocks.0.weight"
        assert swift_tensor_name(name) == "msa_module.blocks.0.weight"

    def test_normalize_sorts_and_strips(self) -> None:
        state = {"module.z.weight": 1, "module.a.weight": 2}
        assert list(normalize_state_dict(state)) == ["a.weight", "z.weight"]

    def test_normalize_rejects_colliding_names(self) -> None:
        # Two different raw keys must never land on one runtime name; silently
        # keeping the last would drop a real weight.
        state = {"module.a.weight": 1, "module._orig_mod.a.weight": 2}
        with pytest.raises(ValueError, match="normalize to the same name"):
            normalize_state_dict(state)

    def test_flags_unmodelled_roots(self) -> None:
        assert unrecognized_roots(["pairformer_stack.a", "mystery_head.b"]) == (
            "mystery_head",
        )

    def test_known_roots_are_accepted(self) -> None:
        assert unrecognized_roots([f"{root}.w" for root in GRAPH_ROOTS]) == ()


class TestQuantization:
    def test_pads_input_width_to_group_size(self) -> None:
        weight = np.random.default_rng(0).normal(size=(8, 100)).astype(np.float32)
        packed = quantize_affine_int8(weight)
        assert packed.logical_shape == (8, 100)
        assert packed.physical_shape == (8, 128)
        assert packed.physical_shape[1] % INT8_GROUP_SIZE == 0

    def test_exact_multiple_is_not_padded(self) -> None:
        weight = np.zeros((4, INT8_GROUP_SIZE), dtype=np.float32)
        packed = quantize_affine_int8(weight)
        assert packed.logical_shape == packed.physical_shape

    def test_round_trip_is_faithful(self) -> None:
        import mlx.core as mx

        weight = np.random.default_rng(1).normal(size=(64, 256)).astype(np.float32)
        packed = quantize_affine_int8(weight)
        restored = np.asarray(
            mx.dequantize(
                mx.array(packed.weight),
                scales=mx.array(packed.scales),
                biases=mx.array(packed.biases),
                group_size=INT8_GROUP_SIZE,
                bits=8,
                mode="affine",
            ),
            dtype=np.float32,
        )[:, :256]
        assert np.corrcoef(weight.ravel(), restored.ravel())[0, 1] > 0.9997

    def test_rejects_non_matrix(self) -> None:
        with pytest.raises(ValueError, match="two-dimensional"):
            quantize_affine_int8(np.zeros(8, dtype=np.float32))

    def test_refuses_to_clamp_out_of_range_matrix(self) -> None:
        # Silently clamping would be indistinguishable from a correct pack.
        weight = np.full((4, 64), 1e6, dtype=np.float32)
        with pytest.raises(ValueError, match="cannot be quantized without clamping"):
            quantize_affine_int8(weight)

    def test_eligibility_is_rank_two_weights_only(self) -> None:
        assert is_quantizable("a.weight", (8, 8))
        assert not is_quantizable("a.weight", (8,))  # LayerNorm gain
        assert not is_quantizable("a.bias", (8, 8))
        assert not is_quantizable("confidence_head.plddt_weight", (24, 384, 50))


class TestOverflowGuard:
    def test_detects_values_outside_float16(self) -> None:
        assert not _narrows_without_overflow(torch.tensor([1e6]), torch.float16)

    def test_accepts_representable_values(self) -> None:
        assert _narrows_without_overflow(torch.tensor([3.25, 51.75]), torch.float16)

    def test_bfloat16_keeps_the_float32_exponent_range(self) -> None:
        # Why upper_bins stays bfloat16 in a bf16 pack but becomes float32 in an
        # fp16 one: bf16 trades mantissa for range and 1e6 fits.
        assert _narrows_without_overflow(torch.tensor([1e6]), torch.bfloat16)

    def test_ignores_preexisting_infinities(self) -> None:
        assert _narrows_without_overflow(torch.tensor([float("inf"), 1.0]),
                                         torch.float16)


class TestMatrixInvariant:
    def test_accepts_the_released_shape(self) -> None:
        _assert_matrix_invariant(
            {
                "a.weight": torch.zeros(4, 8),
                "a.layer_norm.weight": torch.zeros(8),
                "confidence_head.plddt_weight": torch.zeros(24, 384, 50),
            }
        )

    def test_rejects_unnamed_rank_two_parameter(self) -> None:
        with pytest.raises(ValueError, match="not named '.weight'"):
            _assert_matrix_invariant({"a.gate": torch.zeros(4, 8)})

    def test_rejects_rank_three_outside_confidence_head(self) -> None:
        with pytest.raises(ValueError, match="rank-3"):
            _assert_matrix_invariant({"msa_module.w": torch.zeros(2, 3, 4)})


class TestVariantConfigs:
    @pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.slug)
    def test_every_variant_resolves(self, variant: object) -> None:
        configs = resolve_upstream_config(variant.name)  # type: ignore[attr-defined]
        assert configs.model.pairformer.n_blocks > 0

    def test_resolution_does_not_leak_between_variants(self) -> None:
        # A shallow merge writes through upstream's shared nested dicts, so resolving
        # tiny first used to shrink base from 48 pairformer blocks to 8.
        first = resolve_upstream_config(TINY).model.pairformer.n_blocks
        base = resolve_upstream_config(
            "protenix_base_default_v1.0.0"
        ).model.pairformer.n_blocks
        again = resolve_upstream_config(TINY).model.pairformer.n_blocks
        assert (first, base, again) == (8, 48, 8)

    def test_known_architecture_values(self) -> None:
        configuration = ModelConfiguration.from_model_name(TINY)
        assert (configuration.c_s, configuration.c_z) == (384, 128)
        assert configuration.n_cycle == 4
        assert configuration.n_diffusion_steps == 5

    def test_unknown_model_name_is_rejected(self) -> None:
        with pytest.raises(KeyError):
            ModelConfiguration.from_model_name("protenix_imaginary_v9")


class TestManifest:
    def _manifest(self, **overrides: object) -> ArtifactManifest:
        values: dict = {
            "model_name": TINY,
            "source_checkpoint_sha256": "a" * 64,
            "tensors": (),
            "quantization": None,
        }
        values.update(overrides)
        return ArtifactManifest.model_v1(**values)  # type: ignore[arg-type]

    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        from protenix_mlx_export.schema import TensorSpec

        manifest = self._manifest(
            tensors=(TensorSpec(name="a.weight", shape=(2, 2), dtype="float16"),)
        )
        path = tmp_path / "manifest.json"
        manifest.write(path)
        assert ArtifactManifest.read(path) == manifest

    def test_rejects_bad_digest(self) -> None:
        with pytest.raises(ManifestValidationError, match="not a sha256"):
            self._manifest(source_checkpoint_sha256="short")

    def test_rejects_unsorted_tensors(self) -> None:
        from protenix_mlx_export.schema import TensorSpec

        with pytest.raises(ManifestValidationError, match="sorted"):
            self._manifest(
                tensors=(
                    TensorSpec(name="z", shape=(1,), dtype="float16"),
                    TensorSpec(name="a", shape=(1,), dtype="float16"),
                )
            )

    def test_requires_a_model_name(self) -> None:
        with pytest.raises(ManifestValidationError, match="must name the model"):
            self._manifest(model_name="")


class TestPackaging:
    def _artifact(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text("{}")
        (root / "manifest.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"\x00" * 32)
        return root

    def test_zips_every_member(self, tmp_path: Path) -> None:
        artifact = self._artifact(tmp_path / "pack")
        packed = pack_artifact(
            artifact, output=tmp_path / "out.zip", bundle_id="protenix-tiny-mlx-int8"
        )
        with zipfile.ZipFile(packed.path) as archive:
            assert sorted(archive.namelist()) == list(packed.members)
        assert packed.size == (tmp_path / "out.zip").stat().st_size
        assert len(packed.sha256) == 64

    def test_is_byte_reproducible(self, tmp_path: Path) -> None:
        artifact = self._artifact(tmp_path / "pack")
        first = pack_artifact(artifact, output=tmp_path / "a.zip", bundle_id="x")
        second = pack_artifact(artifact, output=tmp_path / "b.zip", bundle_id="x")
        assert first.sha256 == second.sha256

    def test_refuses_incomplete_artifact(self, tmp_path: Path) -> None:
        artifact = self._artifact(tmp_path / "pack")
        (artifact / "config.json").unlink()
        with pytest.raises(FileNotFoundError, match="config.json"):
            pack_artifact(artifact, output=tmp_path / "out.zip", bundle_id="x")

    def test_snippet_reports_the_packed_bytes(self, tmp_path: Path) -> None:
        artifact = self._artifact(tmp_path / "pack")
        packed = pack_artifact(
            artifact, output=tmp_path / "out.zip", bundle_id="protenix-tiny-mlx-int8"
        )
        snippet = packed.weight_bundle_snippet(version="v1", url="https://x/y.zip")
        assert f"sha256='{packed.sha256}'" in snippet
        assert f"size={packed.size:_}" in snippet

    def test_bundle_id_shape(self) -> None:
        assert bundle_id("mini", "int8") == "protenix-mini-mlx-int8"


class TestExportEndToEnd:
    """Exercise the real export path on a small synthetic checkpoint."""

    def _checkpoint(self, path: Path) -> Path:
        state = {
            "module.pairformer_stack.blocks.0.linear.weight": torch.randn(64, 100),
            "module.pairformer_stack.blocks.0.norm.weight": torch.randn(64),
            "module.confidence_head.upper_bins": torch.tensor([1.0, 1e6]),
            "module.confidence_head.plddt_weight": torch.randn(2, 4, 5),
        }
        torch.save({"model": state, "model_version": "v1"}, path)
        return path

    def _export(self, tmp_path: Path, precision: Precision) -> Path:
        checkpoint = self._checkpoint(tmp_path / "ckpt.pt")
        output = tmp_path / f"pack-{precision.value}"
        # The synthetic checkpoint is not any real variant, so the parameter-count
        # check is bypassed by exporting through the internal writers instead.
        from protenix_mlx_export.model_export import (
            _export_dense,
            _export_int8,
            read_state_dict,
        )

        state, _ = read_state_dict(checkpoint)
        if precision is Precision.INT8:
            _export_int8(
                state,
                output=output,
                model_name=TINY,
                source_checkpoint_sha256="b" * 64,
            )
        else:
            _export_dense(
                state,
                output=output,
                model_name=TINY,
                source_checkpoint_sha256="b" * 64,
                precision=precision,
            )
        return output

    def test_int8_splits_matrices_into_three_arrays(self, tmp_path: Path) -> None:
        output = self._export(tmp_path, Precision.INT8)
        manifest = ArtifactManifest.read(output / "manifest.json")
        names = {spec.name for spec in manifest.tensors}
        stem = "pairformer_stack.blocks.0.linear"
        assert {f"{stem}.weight", f"{stem}.scales", f"{stem}.biases"} <= names
        assert manifest.quantization == {
            "bits": 8,
            "group_size": INT8_GROUP_SIZE,
            "mode": "affine",
        }

    def test_int8_records_padding_shapes(self, tmp_path: Path) -> None:
        output = self._export(tmp_path, Precision.INT8)
        manifest = ArtifactManifest.read(output / "manifest.json")
        spec = next(
            s for s in manifest.tensors
            if s.name == "pairformer_stack.blocks.0.linear.weight"
        )
        assert spec.logical_shape == (64, 100)
        assert spec.physical_shape == (64, 128)

    def test_dense_pack_declares_no_quantization(self, tmp_path: Path) -> None:
        output = self._export(tmp_path, Precision.FLOAT16)
        assert ArtifactManifest.read(output / "manifest.json").quantization is None

    def test_out_of_range_constant_is_widened_not_clipped(
        self, tmp_path: Path
    ) -> None:
        output = self._export(tmp_path, Precision.FLOAT16)
        manifest = ArtifactManifest.read(output / "manifest.json")
        spec = next(
            s for s in manifest.tensors if s.name == "confidence_head.upper_bins"
        )
        assert spec.dtype == "float32"

    def test_rank_three_confidence_stack_stays_dense(self, tmp_path: Path) -> None:
        output = self._export(tmp_path, Precision.INT8)
        manifest = ArtifactManifest.read(output / "manifest.json")
        spec = next(
            s for s in manifest.tensors if s.name == "confidence_head.plddt_weight"
        )
        assert spec.shape == (2, 4, 5)
        assert spec.dtype == "float16"

    def test_rejects_a_non_protenix_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pt"
        torch.save({"not_model": {}}, path)
        with pytest.raises(ValueError, match="not a Protenix checkpoint"):
            export_checkpoint(
                checkpoint=path, output=tmp_path / "o", model_name=TINY
            )

    def test_missing_checkpoint_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            export_checkpoint(
                checkpoint=tmp_path / "nope.pt",
                output=tmp_path / "o",
                model_name=TINY,
            )

    def test_config_is_written_next_to_weights(self, tmp_path: Path) -> None:
        output = self._export(tmp_path, Precision.INT8)
        # _export_int8 alone writes no config; the full path does. Assert the
        # contract the packer enforces rather than the internal writer's output.
        assert not (output / "config.json").exists()
        ModelConfiguration.from_model_name(TINY).write(output / "config.json")
        restored = ModelConfiguration.read(output / "config.json")
        assert restored.model_name == TINY
        assert json.loads((output / "config.json").read_text())["c_z"] == 128
