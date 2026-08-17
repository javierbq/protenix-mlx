"""Turn a sequence into a Swift-loadable feature bundle.

This is the offline half of the boltz-mlx pattern applied to Protenix: the featurizer
(CCD tokenization, reference conformers, geometry, dummy MSA) runs here in Python, and
the Swift runtime consumes a flat bundle of precomputed tensors. It deliberately does not
reimplement any of upstream's data pipeline — it drives it and repackages the result.

Requires upstream Protenix importable and the CCD cache present (``PROTENIX_ROOT_DIR``
pointing at a tree with ``common/components.cif`` and its rdkit pickle). MSA and templates
are not run: a single-sequence dummy MSA is used, matching upstream's no-MSA inference.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file

if TYPE_CHECKING:
    from collections.abc import Mapping

# 32/128 is what update_input_feature_dict hardcodes and every atom encoder uses.
ATOM_QUERY_WINDOW = 32
ATOM_KEY_WINDOW = 128


def _install_stubs(source: Path | None) -> None:
    """Make upstream's featurizer importable without its optional heavy deps."""
    if source is not None:
        sys.path.insert(0, str(source))
    if "esm" not in sys.modules:
        esm = types.ModuleType("esm")
        esm.FastaBatchedDataset = object
        esm.pretrained = object
        sys.modules["esm"] = esm
    # The inference package __init__ pulls in the ESM/template dataloaders; stub it so
    # only json_to_feature (which we actually need) is imported.
    if "protenix.data.inference" not in sys.modules and source is not None:
        pkg = types.ModuleType("protenix.data.inference")
        pkg.__path__ = [str(source / "protenix" / "data" / "inference")]
        sys.modules["protenix.data.inference"] = pkg


@dataclass(frozen=True)
class SequenceFeatures:
    """Everything the Swift trunk and diffusion module need, plus write-back metadata."""

    tensors: dict[str, torch.Tensor]
    token_count: int
    atom_count: int
    #: Per-atom (element symbol, atom name, residue index) for the structure writer.
    atom_metadata: list[dict[str, Any]]


def build_features(
    sequence: str, *, name: str = "prediction", source: Path | None = None
) -> SequenceFeatures:
    """Featurize one protein sequence into bundle tensors."""
    _install_stubs(source)
    from protenix.data.inference.json_to_feature import (  # noqa: PLC0415
        SampleDictToFeatures,
    )
    from protenix.data.utils import make_dummy_feature  # noqa: PLC0415
    from protenix.model.protenix import (  # noqa: PLC0415
        update_input_feature_dict,
    )
    from protenix.model.modules.embedders import (  # noqa: PLC0415
        RelativePositionEncoding,
    )

    sample = {
        "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
        "name": name,
    }
    converter = SampleDictToFeatures(sample)
    feature_dict, atom_array, token_array = converter.get_feature_dict()

    # Dummy single-row MSA, exactly as upstream does when no alignment is supplied.
    feature_dict = make_dummy_feature(feature_dict, dummy_feats=["msa"])

    # Relative-position one-hot and the atom-pair geometry, both pure functions of the
    # already-computed features.
    RelativePositionEncoding(c_z=1).generate_relp(feature_dict)
    with torch.no_grad():
        update_input_feature_dict(feature_dict)

    token_count = int(feature_dict["restype"].shape[0])
    atom_count = int(feature_dict["ref_pos"].shape[0])

    tensors = _bundle_tensors(feature_dict, token_count=token_count)
    metadata = _atom_metadata(atom_array, token_array)
    return SequenceFeatures(
        tensors=tensors, token_count=token_count, atom_count=atom_count,
        atom_metadata=metadata,
    )


def _bundle_tensors(
    feature_dict: Mapping[str, Any], *, token_count: int
) -> dict[str, torch.Tensor]:
    """Select and shape the tensors the Swift runtime reads, all float32."""

    def f(name: str) -> torch.Tensor:
        return feature_dict[name].float()

    # MSA feature block [S, N, 34] = one-hot(msa, 32) ++ has_deletion ++ deletion_value.
    msa = feature_dict["msa"].long()
    msa_onehot = torch.nn.functional.one_hot(msa, num_classes=32).float()
    has_deletion = feature_dict["has_deletion"].float().unsqueeze(-1)
    deletion_value = feature_dict["deletion_value"].float().unsqueeze(-1)
    msa_features = torch.cat([msa_onehot, has_deletion, deletion_value], dim=-1)

    tensors = {
        # Atom features.
        "ref_pos": f("ref_pos"),
        "ref_charge": f("ref_charge"),
        "ref_mask": f("ref_mask"),
        "ref_element": f("ref_element"),
        # [N_atom, 4, 64] -> [N_atom, 256], the flat width the linear expects.
        "ref_atom_name_chars": f("ref_atom_name_chars").reshape(
            feature_dict["ref_atom_name_chars"].shape[0], -1
        ),
        "atom_to_token_idx": feature_dict["atom_to_token_idx"].float(),
        "d_lm": f("d_lm"),
        "v_lm": f("v_lm"),
        "mask_trunked": feature_dict["pad_info"]["mask_trunked"].float(),
        # Token features.
        "restype": f("restype"),
        "profile": f("profile"),
        "deletion_mean": f("deletion_mean").reshape(token_count, 1),
        "relp": f("relp"),
        "token_bonds": f("token_bonds"),
        # MSA.
        "msa_features": msa_features,
    }
    return {name: value.contiguous() for name, value in tensors.items()}


def _atom_metadata(atom_array: Any, token_array: Any) -> list[dict[str, Any]]:
    """Per-atom identity for the structure writer."""
    metadata: list[dict[str, Any]] = []
    for index in range(len(atom_array)):
        atom = atom_array[index]
        metadata.append(
            {
                "element": str(getattr(atom, "element", "C")).strip() or "C",
                "atom_name": str(getattr(atom, "atom_name", "CA")).strip() or "CA",
                "res_name": str(getattr(atom, "res_name", "UNK")).strip() or "UNK",
                "res_id": int(getattr(atom, "res_id", 0)),
                "chain_id": str(getattr(atom, "chain_id", "A")).strip() or "A",
            }
        )
    return metadata


def export_features(
    *, sequence: str, output: Path, name: str = "prediction",
    source: Path | None = None,
) -> SequenceFeatures:
    """Featurize `sequence` and write a feature bundle directory."""
    features = build_features(sequence, name=name, source=source)
    output.mkdir(parents=True, exist_ok=True)
    save_file(features.tensors, str(output / "features.safetensors"))
    metadata = {
        "schema_version": 1,
        "artifact_kind": "features",
        "name": name,
        "sequence": sequence,
        "token_count": features.token_count,
        "atom_count": features.atom_count,
        "atom_query_window": ATOM_QUERY_WINDOW,
        "atom_key_window": ATOM_KEY_WINDOW,
        "tensors": {
            key: list(value.shape) for key, value in features.tensors.items()
        },
        "atoms": features.atom_metadata,
    }
    (output / "features.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return features
