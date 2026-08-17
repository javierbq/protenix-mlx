"""Assemble a feature bundle from the residue table, with no upstream imports.

This is the reference the Swift featurizer is a transliteration of. Keeping it in Python
first buys two things: it can be diffed **bitwise** against what upstream's own featurizer
produces (``verify_templates``), and the Swift port then has a known-correct algorithm to
follow rather than a paper to interpret.

Nothing here reads the CCD, imports torch's autograd, or touches Protenix. Everything is
either a lookup in the table or index arithmetic, which is precisely the claim that makes
an on-device fold possible.

Scope: canonical-20 protein chains. Ligands, nucleic acids, modified residues, covalent
modifications and templates are all out, and every one of them is refused rather than
approximated -- ``token_bonds`` is all zeros here, which is true for polypeptide chains
and false the moment a ligand or a disulfide enters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from protenix_mlx_export.residue_templates import (
    ATOM_NAME_CLASSES,
    ATOM_NAME_LENGTH,
    ELEMENT_CLASSES,
    RESTYPE_CLASSES,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from protenix_mlx_export.residue_templates import ResidueTemplate

#: The atom-attention window, as update_input_feature_dict hardcodes it.
N_QUERIES = 32
N_KEYS = 128

#: RelativePositionEncoding's clamps. Both come from the model config and are the same
#: in every released Protenix variant.
R_MAX = 32
S_MAX = 2


@dataclass(frozen=True)
class TemplateFeatures:
    """The same payload build_features returns, assembled from the table."""

    tensors: dict[str, torch.Tensor]
    token_count: int
    atom_count: int
    atom_metadata: list[dict[str, object]]
    chains: tuple[tuple[str, str], ...]


def features_from_templates(
    sequences: str | Iterable[str | tuple[str, str]],
    table: dict[str, ResidueTemplate],
) -> TemplateFeatures:
    """Featurize chains from the table alone."""
    from protenix_mlx_export.feature_export import parse_chains

    chains = parse_chains(sequences)
    _reject_unsupported(chains, table)

    # Chains are grouped into entities exactly as the exporter groups them, because relp
    # reads entity and symmetry ids and a homodimer whose copies land in two entities is
    # told its identical chains are unrelated.
    entity_of_sequence: dict[str, int] = {}
    copies_seen: dict[str, int] = {}
    ordered: list[tuple[str, str]] = []
    for sequence in dict.fromkeys(sequence for _, sequence in chains):
        entity_of_sequence[sequence] = len(entity_of_sequence)
    for chain_id, sequence in _expansion_order(chains):
        ordered.append((chain_id, sequence))

    restype: list[int] = []
    asym_id: list[int] = []
    entity_id: list[int] = []
    sym_id: list[int] = []
    residue_index: list[int] = []

    positions: list[tuple[float, float, float]] = []
    charges: list[float] = []
    element_indices: list[int] = []
    atom_names: list[str] = []
    token_of_atom: list[int] = []
    index_in_token: list[int] = []
    representative: list[float] = []
    atom_metadata: list[dict[str, object]] = []

    for chain_number, (chain_id, sequence) in enumerate(ordered):
        entity = entity_of_sequence[sequence]
        copy_index = copies_seen.get(sequence, 0)
        copies_seen[sequence] = copy_index + 1
        for offset, one_letter in enumerate(sequence):
            template = table[one_letter]
            terminal = offset == len(sequence) - 1
            atoms = template.terminal_atoms if terminal else template.atoms
            token = len(restype)
            restype.append(template.restype_index)
            asym_id.append(chain_number)
            entity_id.append(entity)
            sym_id.append(copy_index)
            residue_index.append(offset)
            # The token's representative atom for distance purposes: CB, or CA for the
            # one residue with no side chain. Glycine is not a special case bolted on --
            # it is what "the first side-chain atom" degenerates to.
            names = [atom.name for atom in atoms]
            rep_name = "CB" if "CB" in names else "CA"
            for position_in_token, atom in enumerate(atoms):
                positions.append(atom.pos)
                charges.append(atom.charge)
                element_indices.append(atom.element_index)
                atom_names.append(atom.name)
                token_of_atom.append(token)
                index_in_token.append(position_in_token)
                representative.append(1.0 if atom.name == rep_name else 0.0)
                atom_metadata.append(
                    {
                        "element": atom.element,
                        "atom_name": atom.name,
                        "res_name": template.code,
                        "res_id": offset + 1,
                        "chain_id": chain_id,
                    }
                )

    token_count = len(restype)
    atom_count = len(positions)

    ref_pos = torch.tensor(positions, dtype=torch.float32).reshape(atom_count, 3)
    ref_charge = torch.tensor(charges, dtype=torch.float32)
    ref_mask = torch.ones(atom_count, dtype=torch.float32)
    ref_element = _one_hot(element_indices, ELEMENT_CLASSES)
    ref_atom_name_chars = _atom_name_chars(atom_names)
    atom_to_token_idx = torch.tensor(token_of_atom, dtype=torch.float32)

    # ref_space_uid numbers each (chain, residue) pair on first appearance, which for a
    # polymer laid out in order is the token index -- so this is atom_to_token_idx by
    # another name, and it is what makes v_lm "are these two atoms in one residue".
    ref_space_uid = torch.tensor(token_of_atom, dtype=torch.float32)
    d_lm, v_lm, mask_trunked = _atom_pair_geometry(ref_pos, ref_space_uid)

    restype_onehot = _one_hot(restype, RESTYPE_CLASSES)
    relp = _relative_position(
        asym_id=asym_id, entity_id=entity_id, sym_id=sym_id,
        residue_index=residue_index,
    )

    # A single dummy alignment row: one-hot(restype) ++ has_deletion ++ deletion_value,
    # which is what make_dummy_feature builds when no a3m is supplied. Not an MSA and
    # not claimed to be one.
    msa_features = torch.cat(
        [restype_onehot.unsqueeze(0), torch.zeros(1, token_count, 2)], dim=-1
    )

    tensors = {
        "ref_pos": ref_pos,
        "ref_charge": ref_charge,
        "ref_mask": ref_mask,
        "ref_element": ref_element,
        "ref_atom_name_chars": ref_atom_name_chars,
        "atom_to_token_idx": atom_to_token_idx,
        "distogram_rep_atom_mask": torch.tensor(representative, dtype=torch.float32),
        "atom_to_tokatom_idx": torch.tensor(index_in_token, dtype=torch.float32),
        "d_lm": d_lm,
        "v_lm": v_lm,
        "mask_trunked": mask_trunked,
        "restype": restype_onehot,
        # With a dummy alignment the profile IS the query's one-hot, and the mean
        # deletion count is zero. Both become real the day an a3m is plumbed through.
        "profile": restype_onehot.clone(),
        "deletion_mean": torch.zeros(token_count, 1),
        "relp": relp,
        # True for a polypeptide chain and false for anything with a ligand, a disulfide
        # or a covalent modification -- all of which are refused above.
        "token_bonds": torch.zeros(token_count, token_count),
        "msa_features": msa_features,
    }
    return TemplateFeatures(
        tensors={name: tensor.contiguous() for name, tensor in tensors.items()},
        token_count=token_count,
        atom_count=atom_count,
        atom_metadata=atom_metadata,
        chains=tuple(ordered),
    )


def _expansion_order(chains: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Entity by entity, copies consecutive -- the order upstream lays chains out in."""
    order: list[str] = []
    members: dict[str, list[str]] = {}
    for chain_id, sequence in chains:
        if sequence not in members:
            order.append(sequence)
            members[sequence] = []
        members[sequence].append(chain_id)
    return [
        (chain_id, sequence) for sequence in order for chain_id in members[sequence]
    ]


def _reject_unsupported(
    chains: Sequence[tuple[str, str]], table: dict[str, ResidueTemplate]
) -> None:
    """Refuse anything the table cannot serve, by name.

    Substituting X for an unknown letter is what upstream's tokenizers do; here it would
    return a structure for a sequence the caller never asked to fold.
    """
    for chain_id, sequence in chains:
        unknown = sorted({letter for letter in sequence if letter not in table})
        if unknown:
            message = (
                f"chain {chain_id} contains residues the template table does not "
                f"carry: {', '.join(unknown)} (canonical 20 only)"
            )
            raise ValueError(message)


def _one_hot(indices: Sequence[int], classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(
        torch.tensor(indices, dtype=torch.long), num_classes=classes
    ).float()


def _atom_name_chars(names: Sequence[str]) -> torch.Tensor:
    """ord(c) - 32, clipped to [0, 63], names padded to 4, one-hot, then flattened."""
    codes = [
        min(max(ord(character) - 32, 0), ATOM_NAME_CLASSES - 1)
        for name in names
        for character in name.ljust(ATOM_NAME_LENGTH)[:ATOM_NAME_LENGTH]
    ]
    onehot = _one_hot(codes, ATOM_NAME_CLASSES)
    return onehot.reshape(len(names), ATOM_NAME_LENGTH * ATOM_NAME_CLASSES)


def _atom_pair_geometry(
    ref_pos: torch.Tensor, ref_space_uid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The windowed atom-pair features: d_lm, v_lm and the window mask.

    Mirrors ``rearrange_qk_to_dense_trunk``: queries are split into blocks of 32, and
    each block reads a 128-wide key window centred on it, so the window reaches 48 atoms
    back and 48 forward. Padding is zeros on both axes, which matters for parity rather
    than for meaning: a padded query row compares equal to reference space 0 and so
    v_lm reads 1 there. Those positions are exactly the ones mask_trunked zeroes.
    """
    atom_count = ref_pos.shape[0]
    trunks = int(math.ceil(atom_count / N_QUERIES))
    query_pad = trunks * N_QUERIES - atom_count
    pad_left = (N_KEYS - N_QUERIES) // 2
    pad_right = int((trunks - 1 / 2) * N_QUERIES + N_KEYS / 2 - atom_count + 1 / 2)

    def pad(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
        shape = (left, *x.shape[1:])
        tail = (right, *x.shape[1:])
        return torch.cat(
            [torch.zeros(shape, dtype=x.dtype), x, torch.zeros(tail, dtype=x.dtype)]
        )

    queries = pad(ref_pos, 0, query_pad).reshape(trunks, N_QUERIES, 3)
    keys = pad(ref_pos, pad_left, pad_right)
    keys = keys.unfold(0, N_KEYS, N_QUERIES).permute(0, 2, 1)

    query_uid = pad(ref_space_uid, 0, query_pad).reshape(trunks, N_QUERIES)
    key_uid = pad(ref_space_uid, pad_left, pad_right).unfold(0, N_KEYS, N_QUERIES)

    d_lm = queries.unsqueeze(2) - keys.unsqueeze(1)
    v_lm = (
        (query_uid.int().unsqueeze(2) == key_uid.int().unsqueeze(1))
        .unsqueeze(-1)
        .float()
    )

    # A window position is real when its query is a real atom and its key is too. The
    # key axis carries `pad_left` of padding in front, so key j of block b is atom
    # b * 32 + j - 48.
    query_index = (
        torch.arange(trunks).unsqueeze(1) * N_QUERIES + torch.arange(N_QUERIES)
    )
    key_index = (
        torch.arange(trunks).unsqueeze(1) * N_QUERIES
        + torch.arange(N_KEYS)
        - pad_left
    )
    query_real = (query_index < atom_count).unsqueeze(-1)
    key_real = ((key_index >= 0) & (key_index < atom_count)).unsqueeze(1)
    mask_trunked = (query_real & key_real).float()
    return d_lm, v_lm, mask_trunked


def _relative_position(
    *,
    asym_id: Sequence[int],
    entity_id: Sequence[int],
    sym_id: Sequence[int],
    residue_index: Sequence[int],
) -> torch.Tensor:
    """The relp one-hot block: [rel_pos(66) | rel_token(66) | same_entity(1) | rel_chain(6)].

    One token is one residue for a standard amino acid, so the rel_token block is the
    degenerate case of the rel_pos one: tokens within a residue would index it, and here
    every residue has exactly one token, leaving 32 on the diagonal and the
    "different residue" sentinel everywhere else.
    """
    asym = torch.tensor(asym_id, dtype=torch.long)
    entity = torch.tensor(entity_id, dtype=torch.long)
    symmetry = torch.tensor(sym_id, dtype=torch.long)
    residue = torch.tensor(residue_index, dtype=torch.long)
    tokens = torch.arange(len(asym_id), dtype=torch.long)

    same_chain = (asym[:, None] == asym[None, :]).long()
    same_residue = (residue[:, None] == residue[None, :]).long() * same_chain
    same_entity = (entity[:, None] == entity[None, :]).long()

    d_residue = torch.clip(
        residue[:, None] - residue[None, :] + R_MAX, min=0, max=2 * R_MAX
    ) * same_chain + (1 - same_chain) * (2 * R_MAX + 1)
    d_token = torch.clip(
        tokens[:, None] - tokens[None, :] + R_MAX, min=0, max=2 * R_MAX
    ) * same_residue + (1 - same_residue) * (2 * R_MAX + 1)
    d_chain = torch.clip(
        symmetry[:, None] - symmetry[None, :] + S_MAX, min=0, max=2 * S_MAX
    ) * same_entity + (1 - same_entity) * (2 * S_MAX + 1)

    return torch.cat(
        [
            torch.nn.functional.one_hot(d_residue, 2 * (R_MAX + 1)),
            torch.nn.functional.one_hot(d_token, 2 * (R_MAX + 1)),
            same_entity[..., None],
            torch.nn.functional.one_hot(d_chain, 2 * (S_MAX + 1)),
        ],
        dim=-1,
    ).float()
