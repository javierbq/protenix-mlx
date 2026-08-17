"""Feature-export tests.

Split from ``test_export.py`` because half of these need what the model exporter does
not: upstream Protenix importable and the CCD cache reachable. Those skip **loudly**
with the reason printed, for the same reason the Swift parity suite does — a suite that
silently passes when its reference data is absent is worse than no suite at all. The
chain-grouping tests below need neither and always run.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from protenix_mlx_export.feature_export import (
    _group_entities,
    build_features,
    export_features,
    parse_chains,
)
from protenix_mlx_export.residue_templates import (
    CANONICAL,
    SCHEMA_VERSION,
    load_templates,
    verify_templates,
)
from protenix_mlx_export.template_features import features_from_templates

#: The table the Swift package ships and reads at fold time.
SHIPPED_TABLE = (
    Path(__file__).resolve().parent.parent
    / "Sources" / "ProtenixMLX" / "Resources" / "residue_templates.json"
)


def _upstream() -> Path | None:
    """A Protenix checkout, from PROTENIX_UPSTREAM or the conventional artifact dir."""
    declared = os.environ.get("PROTENIX_UPSTREAM")
    candidates = [Path(declared)] if declared else []
    candidates.append(Path(__file__).resolve().parent.parent / ".artifacts" / "upstream")
    for candidate in candidates:
        if (candidate / "protenix" / "data" / "inference").is_dir():
            return candidate
    return None


def _ccd_present() -> bool:
    root = os.environ.get("PROTENIX_ROOT_DIR")
    return bool(root) and (Path(root) / "common" / "components.cif").is_file()


UPSTREAM = _upstream()
needs_upstream = pytest.mark.skipif(
    UPSTREAM is None or not _ccd_present(),
    reason=(
        "needs a Protenix checkout (PROTENIX_UPSTREAM or .artifacts/upstream) and the "
        "CCD cache (PROTENIX_ROOT_DIR pointing at a tree with common/components.cif)"
    ),
)


class TestParseChains:
    def test_splits_a_complex_on_slash(self) -> None:
        assert parse_chains("GSHM/AWKD") == [("A", "GSHM"), ("B", "AWKD")]

    def test_labels_default_in_order(self) -> None:
        assert parse_chains(["GG", "AA", "WW"]) == [("A", "GG"), ("B", "AA"), ("C", "WW")]

    def test_explicit_labels_are_kept(self) -> None:
        assert parse_chains([("H", "GG"), ("L", "AA")]) == [("H", "GG"), ("L", "AA")]

    def test_case_and_whitespace_normalised(self) -> None:
        assert parse_chains(" gshm ") == [("A", "GSHM")]

    def test_empty_input_refused(self) -> None:
        with pytest.raises(ValueError, match="no chains"):
            parse_chains("")

    def test_empty_chain_refused(self) -> None:
        # "A//B" is a slip a user makes; folding it as two chains would silently drop one.
        with pytest.raises(ValueError, match="no chains|empty"):
            parse_chains([("A", "  ")])


class TestEntityGrouping:
    """Identical sequences are copies of ONE entity, not two unrelated entities.

    This is the whole reason chains are grouped before upstream sees them: upstream keys
    entity_id off the index in `sequences`, so two entries with the same sequence produce
    a relp block saying two identical chains are unrelated.
    """

    def test_identical_sequences_become_one_entity_with_count(self) -> None:
        entities, labels = _group_entities([("A", "GSHM"), ("B", "GSHM")])
        assert entities == [{"proteinChain": {"sequence": "GSHM", "count": 2}}]
        assert labels == ["A", "B"]

    def test_distinct_sequences_stay_separate(self) -> None:
        entities, labels = _group_entities([("A", "GSHM"), ("B", "AWKD")])
        assert [e["proteinChain"]["count"] for e in entities] == [1, 1]
        assert labels == ["A", "B"]

    def test_labels_follow_upstream_expansion_order(self) -> None:
        # Upstream emits entity by entity with copies consecutive, so an interleaved
        # request has to be relabelled in that order or every atom lands on the wrong
        # chain in the written PDB.
        entities, labels = _group_entities([("A", "GG"), ("B", "AA"), ("C", "GG")])
        assert [e["proteinChain"]["count"] for e in entities] == [2, 1]
        assert labels == ["A", "C", "B"]


@needs_upstream
class TestBuildFeatures:
    def test_bundle_is_reproducible(self) -> None:
        """Two runs agree bitwise. Upstream's own inference path does not."""
        first = build_features("GSHM", source=UPSTREAM)
        second = build_features("GSHM", source=UPSTREAM)
        for name, tensor in first.tensors.items():
            assert (tensor == second.tensors[name]).all(), name

    def test_augmentation_is_off_by_default_and_centres_conformers(self) -> None:
        features = build_features("GSHM", source=UPSTREAM)
        assert features.augmented is False
        # Centred, per reference conformer: each residue's atoms average to the origin.
        positions = features.tensors["ref_pos"]
        token_of_atom = features.tensors["atom_to_token_idx"].long()
        for token in token_of_atom.unique():
            centroid = positions[token_of_atom == token].mean(dim=0)
            assert centroid.abs().max() < 1e-4

    def test_augmentation_is_seeded_when_asked_for(self) -> None:
        first = build_features("GSHM", source=UPSTREAM, augment=True, seed=7)
        second = build_features("GSHM", source=UPSTREAM, augment=True, seed=7)
        plain = build_features("GSHM", source=UPSTREAM)
        assert first.augmented is True
        assert (first.tensors["ref_pos"] == second.tensors["ref_pos"]).all()
        assert (first.tensors["ref_pos"] - plain.tensors["ref_pos"]).abs().max() > 0.1

    def test_augmentation_is_rigid(self) -> None:
        """It rotates and translates a conformer; it must not deform one."""
        import torch

        plain = build_features("GSHM", source=UPSTREAM)
        augmented = build_features("GSHM", source=UPSTREAM, augment=True, seed=3)
        token_of_atom = plain.tensors["atom_to_token_idx"].long()
        for token in token_of_atom.unique():
            mask = token_of_atom == token
            before = torch.cdist(*(plain.tensors["ref_pos"][mask],) * 2)
            after = torch.cdist(*(augmented.tensors["ref_pos"][mask],) * 2)
            assert (before - after).abs().max() < 1e-4

    def test_multi_chain_concatenates_tokens_and_atoms(self) -> None:
        single = build_features("GSHM", source=UPSTREAM)
        dimer = build_features("GSHM/GSHM", source=UPSTREAM)
        assert dimer.token_count == 2 * single.token_count
        assert dimer.atom_count == 2 * single.atom_count
        assert [chain for chain, _ in dimer.chains] == ["A", "B"]

    def test_homodimer_chains_share_an_entity(self) -> None:
        # relp is [rel_pos(66) | rel_token(66) | same_entity(1) | rel_chain(6)], so 132
        # is the same-entity bit. A homodimer that reads 0 there is being told its two
        # identical chains are unrelated.
        dimer = build_features("GSHM/GSHM", source=UPSTREAM)
        same_entity = dimer.tensors["relp"][..., 132]
        assert same_entity[0, -1] == 1

    def test_heterodimer_chains_do_not(self) -> None:
        dimer = build_features("GSHM/AWKD", source=UPSTREAM)
        same_entity = dimer.tensors["relp"][..., 132]
        assert same_entity[0, -1] == 0

    def test_chain_labels_reach_the_atom_metadata(self) -> None:
        dimer = build_features([("H", "GSHM"), ("L", "AWKD")], source=UPSTREAM)
        labels = {atom["chain_id"] for atom in dimer.atom_metadata}
        assert labels == {"H", "L"}

    def test_written_bundle_records_its_chains_and_augmentation(
        self, tmp_path: Path
    ) -> None:
        import json

        export_features(sequence="GSHM/AWKD", output=tmp_path, source=UPSTREAM)
        metadata = json.loads((tmp_path / "features.json").read_text())
        assert metadata["sequence"] == "GSHM/AWKD"
        assert metadata["chains"] == [
            {"chain": "A", "sequence": "GSHM"},
            {"chain": "B", "sequence": "AWKD"},
        ]
        assert metadata["ref_pos_augmented"] is False


class TestShippedTemplateTable:
    """The table itself, which needs neither upstream nor the CCD to inspect."""

    def test_carries_the_canonical_twenty(self) -> None:
        table = load_templates(SHIPPED_TABLE)
        assert set(table) == set(CANONICAL)

    def test_restype_indices_are_distinct_and_in_range(self) -> None:
        table = load_templates(SHIPPED_TABLE)
        indices = sorted(template.restype_index for template in table.values())
        assert indices == list(range(20))

    def test_terminal_form_adds_exactly_oxt(self) -> None:
        # The C-terminal carboxyl oxygen is the ONLY way position changes a residue --
        # the N-terminus is not special. If that ever stops being true, the table needs
        # a third form and this is where it should be noticed.
        table = load_templates(SHIPPED_TABLE)
        for one_letter, template in sorted(table.items()):
            ordinary = [atom.name for atom in template.atoms]
            terminal = [atom.name for atom in template.terminal_atoms]
            assert terminal == [*ordinary, "OXT"], one_letter

    def test_conformers_are_centred(self) -> None:
        table = load_templates(SHIPPED_TABLE)
        for one_letter, template in sorted(table.items()):
            for atoms in (template.atoms, template.terminal_atoms):
                for axis in range(3):
                    centre = sum(atom.pos[axis] for atom in atoms) / len(atoms)
                    assert abs(centre) < 1e-6, (one_letter, axis)

    def test_backbone_geometry_is_chemically_sane(self) -> None:
        """A table of the right shape carrying nonsense coordinates would still load."""
        table = load_templates(SHIPPED_TABLE)
        for one_letter, template in sorted(table.items()):
            by_name = {atom.name: atom.pos for atom in template.atoms}
            n_ca = math.dist(by_name["N"], by_name["CA"])
            ca_c = math.dist(by_name["CA"], by_name["C"])
            assert 1.3 < n_ca < 1.6, (one_letter, n_ca)
            assert 1.4 < ca_c < 1.7, (one_letter, ca_c)

    def test_schema_version_matches_the_writer(self) -> None:
        document = json.loads(SHIPPED_TABLE.read_text())
        assert document["schema_version"] == SCHEMA_VERSION
        assert document["kind"] == "residue_templates"

    def test_records_the_upstream_commit_it_came_from(self) -> None:
        # These coordinates are a released model's chemistry. A table that cannot say
        # which tree produced it cannot be re-derived or audited.
        document = json.loads(SHIPPED_TABLE.read_text())
        assert len(document["upstream_commit"]) == 40


class TestFeaturesFromTemplates:
    """The reference assembly the Swift featurizer transliterates.

    These need no upstream: they check the table-driven path against itself and against
    what the shapes have to be. The bitwise comparison with upstream is below, and skips.
    """

    def test_shapes_follow_the_sequence(self) -> None:
        table = load_templates(SHIPPED_TABLE)
        features = features_from_templates("GSHM", table)
        assert features.token_count == 4
        assert features.tensors["restype"].shape == (4, 32)
        assert features.tensors["relp"].shape == (4, 4, 139)
        assert features.tensors["msa_features"].shape == (1, 4, 34)
        assert features.tensors["ref_pos"].shape == (features.atom_count, 3)

    def test_windows_are_blocks_of_thirty_two(self) -> None:
        table = load_templates(SHIPPED_TABLE)
        features = features_from_templates("ACDEFGHIKLMNPQRSTVWY", table)
        blocks = -(-features.atom_count // 32)
        assert features.tensors["d_lm"].shape == (blocks, 32, 128, 3)
        assert features.tensors["mask_trunked"].shape == (blocks, 32, 128)

    def test_non_canonical_residues_are_refused_by_name(self) -> None:
        table = load_templates(SHIPPED_TABLE)
        with pytest.raises(ValueError, match="X, Z"):
            features_from_templates("GSHXMZ", table)

    def test_last_residue_of_each_chain_gets_its_oxt(self) -> None:
        table = load_templates(SHIPPED_TABLE)
        dimer = features_from_templates("GG/GG", table)
        oxt = [atom for atom in dimer.atom_metadata if atom["atom_name"] == "OXT"]
        assert len(oxt) == 2
        assert {atom["chain_id"] for atom in oxt} == {"A", "B"}


@needs_upstream
class TestTemplateParity:
    """The table is only worth anything if it rebuilds what upstream produces."""

    #: Chosen for their edge cases: one residue (a single token), every canonical
    #: residue, a chain crossing the 32-atom window boundary, a heterodimer, a homodimer
    #: (shared entity), a trimer, and prolines/cysteines whose conformers are unusual.
    SEQUENCES = (
        "M",
        "GSHM",
        "ACDEFGHIKLMNPQRSTVWY",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        "GSHM/AWKD",
        "GSHM/GSHM",
        "GG/GG/GG",
        "PPPP",
        "CCCC",
    )

    @pytest.mark.parametrize("sequence", SEQUENCES)
    def test_rebuilds_upstream_bitwise(self, sequence: str) -> None:
        verify_templates(
            load_templates(SHIPPED_TABLE), sequences=[sequence], source=UPSTREAM
        )

    def test_atom_identity_matches_too(self) -> None:
        """Tensors agreeing is not enough: the PDB is written from this metadata."""
        table = load_templates(SHIPPED_TABLE)
        expected = build_features("GSHM/AWKD", source=UPSTREAM)
        actual = features_from_templates("GSHM/AWKD", table)
        keys = ("atom_name", "res_name", "res_id", "chain_id")
        assert [
            tuple(atom[key] for key in keys) for atom in actual.atom_metadata
        ] == [tuple(atom[key] for key in keys) for atom in expected.atom_metadata]

    @pytest.mark.parametrize("remainder", [0, 1, 31])
    def test_chains_at_the_window_boundary(self, remainder: int) -> None:
        """Trunking is where an off-by-one hides.

        The cases that matter are an atom count that exactly fills the last block of 32,
        one that leaves a single atom in it, and one that leaves it one short. Poly-A is
        5 atoms per residue plus OXT, so all three are reachable; the length is searched
        for rather than hardcoded because it moves if the reference conformers ever do.
        """
        table = load_templates(SHIPPED_TABLE)
        for length in range(1, 80):
            sequence = "A" * length
            atoms = features_from_templates(sequence, table).atom_count
            if atoms % 32 == remainder and atoms > 32:
                verify_templates(table, sequences=[sequence], source=UPSTREAM)
                return
        pytest.fail(f"no poly-alanine under 80 residues has {remainder} atoms mod 32")
