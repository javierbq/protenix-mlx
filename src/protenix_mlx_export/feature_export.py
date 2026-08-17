"""Turn one or more sequences into a Swift-loadable feature bundle.

This is the offline half of the boltz-mlx pattern applied to Protenix: the featurizer
(CCD tokenization, reference conformers, geometry, dummy MSA) runs here in Python, and
the Swift runtime consumes a flat bundle of precomputed tensors. It deliberately does not
reimplement any of upstream's data pipeline — it drives it and repackages the result.

Requires upstream Protenix importable and the CCD cache present (``PROTENIX_ROOT_DIR``
pointing at a tree with ``common/components.cif`` and its rdkit pickle). MSA and templates
are not run: a single-sequence dummy MSA is used, matching upstream's no-MSA inference.

Two departures from what upstream's inference path does, both deliberate:

**Reference conformers are not augmented by default.** ``Featurizer`` defaults
``ref_pos_augment=True``, which draws a fresh ``Rotation.random()`` and a
``U(-1, 1)`` translation per reference conformer from the *global* numpy RNG. Upstream
trains that way on purpose, and at inference it makes every bundle — and therefore every
fold — irreproducible even at a fixed diffusion seed. Off, ``ref_pos`` is the centred
conformer and nothing else: a pure function of the residue, which is what lets the Swift
featurizer produce the same tensors from a shipped table. Pass ``augment=True`` (with
``seed``) to get upstream's behaviour back.

**Identical sequences are one entity.** Upstream keys ``entity_id`` off the index in
``sequences``, so passing the same sequence twice as two entries yields two entities and
a ``relp`` block that says two identical chains are unrelated. Chains are grouped by
sequence here and expanded with ``count``, which is the AF3 semantics.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Set before any upstream import: the default LAYERNORM_TYPE builds a CUDA extension,
# which on a Mac fails at "Ninja is required to load C++ extensions" halfway down an
# import chain that has nothing to do with layer norm. fixtures.py does the same.
os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

# 32/128 is what update_input_feature_dict hardcodes and every atom encoder uses.
ATOM_QUERY_WINDOW = 32
ATOM_KEY_WINDOW = 128

#: Chain ids upstream hands out, in order, as it expands entities. Used only to map its
#: labels back onto the caller's, never to decide feature content.
_UPSTREAM_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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
    #: (chain id, sequence) in the order the atoms appear, after entity grouping.
    chains: tuple[tuple[str, str], ...] = ()
    #: Whether ref_pos carries a random rotation/translation per reference conformer.
    augmented: bool = False


def parse_chains(sequences: str | Iterable[str | tuple[str, str]]) -> list[tuple[str, str]]:
    """Normalise the caller's chains to [(chain id, sequence)].

    Accepts what RayMol's own spec string looks like — ``"AAA/BBB"`` — as well as a list
    of sequences or of explicit (id, sequence) pairs. Ids are labels only: they reach the
    PDB the structure writer emits and nothing in the features.
    """
    if isinstance(sequences, str):
        items: list[str | tuple[str, str]] = [
            part for part in sequences.split("/") if part
        ]
    else:
        items = list(sequences)
    if not items:
        message = "no chains given"
        raise ValueError(message)
    chains: list[tuple[str, str]] = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            chain_id, sequence = _UPSTREAM_CHAIN_IDS[index % 26], item
        else:
            chain_id, sequence = item
        sequence = sequence.strip().upper()
        if not sequence:
            message = f"chain {chain_id} is empty"
            raise ValueError(message)
        chains.append((chain_id, sequence))
    if len(chains) > len(_UPSTREAM_CHAIN_IDS):
        message = f"{len(chains)} chains exceeds the {len(_UPSTREAM_CHAIN_IDS)} upstream labels"
        raise ValueError(message)
    return chains


def _group_entities(chains: Sequence[tuple[str, str]]) -> tuple[list[dict], list[str]]:
    """Group identical sequences into one entity, preserving first-appearance order.

    Returns upstream's ``sequences`` list and the caller's chain ids reordered to match
    how upstream will lay the copies out: entity by entity, copies consecutive.
    """
    order: list[str] = []
    members: dict[str, list[str]] = {}
    for chain_id, sequence in chains:
        if sequence not in members:
            order.append(sequence)
            members[sequence] = []
        members[sequence].append(chain_id)
    entities = [
        {"proteinChain": {"sequence": sequence, "count": len(members[sequence])}}
        for sequence in order
    ]
    labels = [chain_id for sequence in order for chain_id in members[sequence]]
    return entities, labels


def build_features(
    sequences: str | Iterable[str | tuple[str, str]],
    *,
    name: str = "prediction",
    source: Path | None = None,
    augment: bool = False,
    seed: int = 0,
) -> SequenceFeatures:
    """Featurize one or more protein chains into bundle tensors.

    `augment` reinstates upstream's random per-conformer rotation and translation; it is
    off by default so a bundle is a function of its sequences alone. `seed` seeds numpy
    so that even the augmented path is reproducible, which upstream's is not.
    """
    _install_stubs(source)
    from protenix.data.core.featurizer import Featurizer  # noqa: PLC0415
    from protenix.data.inference.json_to_feature import (  # noqa: PLC0415
        SampleDictToFeatures,
    )
    from protenix.data.utils import make_dummy_feature  # noqa: PLC0415
    from protenix.model.modules.embedders import (  # noqa: PLC0415
        RelativePositionEncoding,
    )
    from protenix.model.protenix import (  # noqa: PLC0415
        update_input_feature_dict,
    )

    chains = parse_chains(sequences)
    entities, labels = _group_entities(chains)
    sample = {"sequences": entities, "name": name}

    if augment:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)  # noqa: NPY002
    with _augmentation(Featurizer, enabled=augment):
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
    metadata = _atom_metadata(atom_array, token_array, labels)
    by_id = dict(chains)
    ordered = tuple((chain_id, by_id[chain_id]) for chain_id in labels)
    return SequenceFeatures(
        tensors=tensors, token_count=token_count, atom_count=atom_count,
        atom_metadata=metadata, chains=ordered, augmented=augment,
    )


@contextmanager
def _augmentation(featurizer_class: type, *, enabled: bool) -> Iterator[None]:
    """Force ``Featurizer(ref_pos_augment=...)`` for the duration of the block.

    Patched rather than passed because ``SampleDictToFeatures.get_feature_dict``
    constructs the ``Featurizer`` itself and exposes no way through. The alternative —
    transforming ``ref_pos`` after the fact — would be wrong: ``get_token_frame`` reads
    ``ref_pos`` to pick each token's frame atoms, so the flag has to be right at the
    source, not corrected downstream.
    """
    original = featurizer_class.__init__

    @functools.wraps(original)
    def patched(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["ref_pos_augment"] = enabled
        original(self, *args, **kwargs)

    featurizer_class.__init__ = patched
    try:
        yield
    finally:
        featurizer_class.__init__ = original


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
        # The two the CONFIDENCE head needs and the structure path does not:
        # which atom stands for its token when token-token distances are measured
        # (CB, or CA for glycine), and each atom's index within its own token, which
        # is what selects its row of plddt_weight / resolved_weight.
        "distogram_rep_atom_mask": feature_dict["distogram_rep_atom_mask"].float(),
        "atom_to_tokatom_idx": feature_dict["atom_to_tokatom_idx"].float(),
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


def _atom_metadata(
    atom_array: Any, token_array: Any, labels: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """Per-atom identity for the structure writer.

    `labels` renames upstream's generated chain ids (A, B, C...) back to the caller's, by
    order of first appearance. A relabel, never a reorder: the atom order is the
    featurizer's and every tensor is indexed by it.
    """
    seen: list[str] = []
    metadata: list[dict[str, Any]] = []
    for index in range(len(atom_array)):
        atom = atom_array[index]
        chain_id = str(getattr(atom, "chain_id", "A")).strip() or "A"
        if chain_id not in seen:
            seen.append(chain_id)
        position = seen.index(chain_id)
        metadata.append(
            {
                "element": str(getattr(atom, "element", "C")).strip() or "C",
                "atom_name": str(getattr(atom, "atom_name", "CA")).strip() or "CA",
                "res_name": str(getattr(atom, "res_name", "UNK")).strip() or "UNK",
                "res_id": int(getattr(atom, "res_id", 0)),
                "chain_id": labels[position] if position < len(labels) else chain_id,
            }
        )
    return metadata


def export_features(
    *, sequence: str | Iterable[str | tuple[str, str]], output: Path,
    name: str = "prediction", source: Path | None = None,
    augment: bool = False, seed: int = 0,
) -> SequenceFeatures:
    """Featurize `sequence` and write a feature bundle directory."""
    features = build_features(
        sequence, name=name, source=source, augment=augment, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    save_file(features.tensors, str(output / "features.safetensors"))
    metadata = {
        "schema_version": 1,
        "artifact_kind": "features",
        "name": name,
        # The chains as folded, in atom order. `sequence` stays a single string so a
        # reader that predates multi-chain still finds what it expects; for a complex it
        # is the "/"-joined form, which is also what RayMol's spec string looks like.
        "sequence": "/".join(sequence for _, sequence in features.chains),
        "chains": [
            {"chain": chain_id, "sequence": chain_sequence}
            for chain_id, chain_sequence in features.chains
        ],
        # Whether ref_pos carries upstream's random per-conformer transform. Recorded
        # because it decides whether this bundle is reproducible, and a reader cannot
        # tell from the tensors.
        "ref_pos_augmented": features.augmented,
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
