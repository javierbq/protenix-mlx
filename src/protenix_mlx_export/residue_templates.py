"""Freeze the CCD-derived per-residue data the Swift featurizer needs into a table.

Everything upstream's featurizer looks up in the CCD for a *standard* amino acid is a
constant: the atom set, their order, elements, formal charges, and the reference
conformer's coordinates. Only ligands and modified residues genuinely need a chemical
component dictionary at fold time. So the canonical 20 are exported once, here, and ship
with the Swift package — which is what lets a fold run with no torch, no rdkit and no
624 MB ``components.cif`` on the machine doing it.

The table is *derived by running upstream's own featurizer* and slicing per residue,
never by reimplementing the CCD lookup. Two properties make that sound, both asserted by
``verify_templates`` and by the tests:

* **A residue's reference conformer does not depend on its neighbours.** Checked by
  building each residue in several sequence contexts and requiring the coordinates to
  agree bitwise.
* **Position matters in exactly one way.** The C-terminal residue carries an extra
  ``OXT``; the N-terminal residue is not special at all. So each residue has two forms
  and no more.

Coordinates are stored **already centred**, in float32, exactly as the featurizer emits
them. Storing them uncentred and centring in Swift would be prettier and would put a
float32 sum between the table and ``ref_pos`` — this way the Swift featurizer copies
bits and parity on ``ref_pos`` is exact rather than within a tolerance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from protenix_mlx_export.feature_export import build_features

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The canonical 20, in one-letter form. Anything else is a fold this table cannot serve.
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"

#: restype is 32-wide: 20 amino acids + unknown, 4 RNA + unknown, 4 DNA + unknown, gap.
RESTYPE_CLASSES = 32

#: ref_element is one-hot over atomic number, up to 128 (AF3 SI Table 5).
ELEMENT_CLASSES = 128

#: Atom-name characters are encoded ord(c) - 32, clipped to [0, 63], names padded to 4.
ATOM_NAME_LENGTH = 4
ATOM_NAME_CLASSES = 64

SCHEMA_VERSION = 1

#: Contexts each residue is built in. Flanking glycines put it mid-chain; the pair of
#: probes is what proves the conformer is context-free rather than assuming it.
_STANDARD_PROBES = ("G{aa}G", "W{aa}P")

#: The C-terminal form. Two flanks so the residue under test is never also N-terminal.
_TERMINAL_PROBE = "GG{aa}"


@dataclass(frozen=True)
class TemplateAtom:
    """One atom of a reference conformer."""

    name: str
    element: str
    #: Index into the 128-wide ref_element one-hot: atomic number - 1.
    element_index: int
    charge: float
    #: Centred reference-conformer coordinates, float32 as the featurizer emitted them.
    pos: tuple[float, float, float]


@dataclass(frozen=True)
class ResidueTemplate:
    """A canonical residue in both the forms a chain can contain."""

    one_letter: str
    code: str
    #: Index into the 32-wide restype one-hot.
    restype_index: int
    #: The ordinary form.
    atoms: tuple[TemplateAtom, ...]
    #: The C-terminal form: the same conformer plus OXT, re-centred over all of it.
    terminal_atoms: tuple[TemplateAtom, ...]


def _slice_residues(sequence: str, source: Path | None) -> list[dict[str, Any]]:
    """Featurize `sequence` and split it into per-token atom records."""
    features = build_features(sequence, source=source, augment=False)
    token_of_atom = features.tensors["atom_to_token_idx"].long()
    positions = features.tensors["ref_pos"]
    charges = features.tensors["ref_charge"]
    elements = features.tensors["ref_element"].argmax(dim=-1)
    restypes = features.tensors["restype"].argmax(dim=-1)

    residues: list[dict[str, Any]] = []
    for token in range(features.token_count):
        indices = (token_of_atom == token).nonzero().flatten().tolist()
        atoms = tuple(
            TemplateAtom(
                name=features.atom_metadata[index]["atom_name"],
                element=features.atom_metadata[index]["element"],
                element_index=int(elements[index]),
                charge=float(charges[index]),
                pos=tuple(float(value) for value in positions[index]),
            )
            for index in indices
        )
        residues.append(
            {
                "code": features.atom_metadata[indices[0]]["res_name"],
                "restype_index": int(restypes[token]),
                "atoms": atoms,
            }
        )
    return residues


def build_templates(source: Path | None = None) -> list[ResidueTemplate]:
    """Derive the table by running the real featurizer over probe sequences."""
    standard: dict[str, dict[str, Any]] = {}
    for probe in _STANDARD_PROBES:
        for one_letter in CANONICAL:
            residues = _slice_residues(probe.format(aa=one_letter), source)
            # Index 1: the residue under test, flanked on both sides.
            found = residues[1]
            previous = standard.get(one_letter)
            if previous is None:
                standard[one_letter] = found
                continue
            _assert_same(one_letter, previous, found, probe)

    templates: list[ResidueTemplate] = []
    for one_letter in CANONICAL:
        ordinary = standard[one_letter]
        terminal = _slice_residues(_TERMINAL_PROBE.format(aa=one_letter), source)[-1]
        extra = [
            atom.name
            for atom in terminal["atoms"]
            if atom.name not in {a.name for a in ordinary["atoms"]}
        ]
        if extra != ["OXT"]:
            message = (
                f"{ordinary['code']}: expected the C-terminal form to add exactly OXT, "
                f"got {extra or 'nothing'}"
            )
            raise ValueError(message)
        templates.append(
            ResidueTemplate(
                one_letter=one_letter,
                code=ordinary["code"],
                restype_index=ordinary["restype_index"],
                atoms=ordinary["atoms"],
                terminal_atoms=terminal["atoms"],
            )
        )
    return templates


def _assert_same(
    one_letter: str, first: dict[str, Any], second: dict[str, Any], probe: str
) -> None:
    """Refuse a residue whose conformer depends on what sits next to it."""
    if first["code"] != second["code"] or first["restype_index"] != second["restype_index"]:
        message = f"{one_letter}: identity differs between probes"
        raise ValueError(message)
    if [atom.name for atom in first["atoms"]] != [atom.name for atom in second["atoms"]]:
        message = f"{one_letter}: atom set differs between probes"
        raise ValueError(message)
    for left, right in zip(first["atoms"], second["atoms"], strict=True):
        if left.pos != right.pos or left.element_index != right.element_index:
            message = (
                f"{one_letter}: reference conformer is not context-free -- {left.name} "
                f"moved in probe {probe!r}. The table cannot be a lookup if this holds."
            )
            raise ValueError(message)


def write_templates(
    output: Path, *, source: Path | None = None, upstream_commit: str = ""
) -> list[ResidueTemplate]:
    """Build the table and write it as JSON."""
    templates = build_templates(source=source)
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": "residue_templates",
        "restype_classes": RESTYPE_CLASSES,
        "element_classes": ELEMENT_CLASSES,
        "atom_name_length": ATOM_NAME_LENGTH,
        "atom_name_classes": ATOM_NAME_CLASSES,
        # Recorded so a table can be traced to what produced it: these coordinates are
        # a released checkpoint's chemistry, not this repo's opinion.
        "upstream_commit": upstream_commit,
        "residues": [asdict(template) for template in templates],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return templates


def load_templates(path: Path) -> dict[str, ResidueTemplate]:
    """Read a written table back, keyed by one-letter code."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        message = (
            f"residue template schema {document.get('schema_version')!r} is not "
            f"{SCHEMA_VERSION}"
        )
        raise ValueError(message)
    table: dict[str, ResidueTemplate] = {}
    for entry in document["residues"]:
        table[entry["one_letter"]] = ResidueTemplate(
            one_letter=entry["one_letter"],
            code=entry["code"],
            restype_index=entry["restype_index"],
            atoms=tuple(TemplateAtom(**atom) for atom in entry["atoms"]),
            terminal_atoms=tuple(TemplateAtom(**atom) for atom in entry["terminal_atoms"]),
        )
    return table


def verify_templates(
    table: dict[str, ResidueTemplate], *, sequences: Sequence[str], source: Path | None
) -> None:
    """Rebuild each sequence from the table and require the real featurizer to agree.

    The table is only worth anything if what it reconstructs is what upstream produces,
    so this compares the assembled tensors rather than the table's own numbers.
    """
    from protenix_mlx_export.template_features import features_from_templates

    for sequence in sequences:
        expected = build_features(sequence, source=source, augment=False)
        actual = features_from_templates(sequence, table)
        for name, tensor in expected.tensors.items():
            other = actual.tensors[name]
            if tensor.shape != other.shape:
                message = (
                    f"{sequence}: {name} shape {tuple(other.shape)} != "
                    f"{tuple(tensor.shape)}"
                )
                raise ValueError(message)
            if not (tensor == other).all():
                worst = (tensor - other).abs().max().item()
                message = f"{sequence}: {name} differs from upstream (max {worst:.3e})"
                raise ValueError(message)
