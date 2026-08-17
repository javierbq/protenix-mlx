import Foundation
import MLX

/// Builds a `FeatureBundle` from sequences alone — the half of the pipeline that used to
/// require Python.
///
/// Upstream's featurizer is a data pipeline over the CCD: tokenize, look up reference
/// conformers, compute geometry, build the relative-position encoding. For the canonical
/// 20 all of the chemistry is a constant (see ``ResidueTemplates``), and what remains is
/// index arithmetic — which is what this type is. The result is compared **bitwise**
/// against `protenix-mlx export-features` by `FeaturizerParityTests`; the Python
/// reference it transliterates is `template_features.py`.
///
/// What it does not do, and refuses rather than approximates: ligands, nucleic acids,
/// modified residues, covalent bonds between tokens, structural templates, and real
/// alignments. `msa_features` here is upstream's depth-1 dummy — the query one-hot, no
/// alignment — which is exactly what `export-features` produces and must not be
/// mistaken for MSA support.
public enum Featurizer {
  /// The atom-attention window, as `update_input_feature_dict` hardcodes it.
  public static let queryWindow = 32
  public static let keyWindow = 128

  /// `RelativePositionEncoding`'s clamps, identical in every released variant.
  static let rMax = 32
  static let sMax = 2

  /// Chain labels handed out when the caller names none. Labels reach the written PDB
  /// and nothing else — no feature depends on them.
  static let defaultChainIDs = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

  /// One chain to fold.
  public struct Chain: Sendable, Equatable {
    public let id: String
    public let sequence: String

    public init(id: String, sequence: String) {
      self.id = id
      self.sequence = sequence
    }
  }

  /// Featurize a `"/"`-separated spec, the form RayMol's own chain string takes.
  public static func bundle(
    sequence: String, name: String = "prediction",
    templates: ResidueTemplates? = nil
  ) throws -> FeatureBundle {
    let parts = sequence.split(separator: "/").map(String.init)
    let chains = parts.enumerated().map { index, part in
      Chain(id: String(defaultChainIDs[index % defaultChainIDs.count]), sequence: part)
    }
    return try bundle(chains: chains, name: name, templates: templates)
  }

  /// Featurize explicit chains.
  ///
  /// Chains are grouped into entities by sequence before anything is numbered: upstream
  /// keys `entity_id` off position, so two copies of one sequence submitted as two
  /// entities produce a `relp` block that tells the model its identical chains are
  /// unrelated. Grouping them is the AF3 semantics and what the exporter does.
  public static func bundle(
    chains requested: [Chain], name: String = "prediction",
    templates providedTemplates: ResidueTemplates? = nil
  ) throws -> FeatureBundle {
    let templates = try providedTemplates ?? ResidueTemplates.bundled()
    let chains = try normalize(requested)

    // Entity by entity, copies consecutive: the order upstream lays chains out in, and
    // therefore the order every tensor is indexed by.
    var entityOfSequence: [String: Int] = [:]
    var order: [String] = []
    var members: [String: [Chain]] = [:]
    for chain in chains {
      if members[chain.sequence] == nil {
        order.append(chain.sequence)
        entityOfSequence[chain.sequence] = entityOfSequence.count
        members[chain.sequence] = []
      }
      members[chain.sequence]?.append(chain)
    }
    let expanded = order.flatMap { members[$0] ?? [] }

    var restype: [Int32] = []
    var asymID: [Int32] = []
    var entityID: [Int32] = []
    var symID: [Int32] = []
    var residueIndex: [Int32] = []

    var positions: [Float] = []
    var charges: [Float] = []
    var elementIndices: [Int32] = []
    var nameCharacters: [Int32] = []
    var tokenOfAtom: [Int32] = []
    var indexInToken: [Int32] = []
    var representative: [Float] = []
    var atoms: [FeatureBundle.Atom] = []

    var copiesSeen: [String: Int] = [:]
    for (chainNumber, chain) in expanded.enumerated() {
      let entity = entityOfSequence[chain.sequence] ?? 0
      let copyIndex = copiesSeen[chain.sequence] ?? 0
      copiesSeen[chain.sequence] = copyIndex + 1

      let letters = Array(chain.sequence)
      for (offset, letter) in letters.enumerated() {
        let residue = try templates.residue(for: letter)
        // The C-terminal residue carries an extra OXT and a conformer re-centred over
        // it. That is the only way position changes a residue; the N-terminus is not
        // special, which the table's writer verifies rather than assumes.
        let conformer = offset == letters.count - 1 ? residue.terminalAtoms : residue.atoms
        let token = Int32(restype.count)
        restype.append(Int32(residue.restypeIndex))
        asymID.append(Int32(chainNumber))
        entityID.append(Int32(entity))
        symID.append(Int32(copyIndex))
        residueIndex.append(Int32(offset))
        // The token's representative atom for distance purposes: CB, or CA for the one
        // residue with no side chain. Glycine is not a special case bolted on — it is
        // what "the first side-chain atom" degenerates to. Only the confidence head
        // reads this; the structure path never does.
        let representativeName = conformer.contains { $0.name == "CB" } ? "CB" : "CA"
        for (positionInToken, atom) in conformer.enumerated() {
          positions.append(contentsOf: atom.pos)
          charges.append(atom.charge)
          elementIndices.append(Int32(atom.elementIndex))
          nameCharacters.append(contentsOf: encodeName(atom.name))
          tokenOfAtom.append(token)
          indexInToken.append(Int32(positionInToken))
          representative.append(atom.name == representativeName ? 1 : 0)
          atoms.append(
            FeatureBundle.Atom(
              element: atom.element, atomName: atom.name, resName: residue.code,
              resId: offset + 1, chainId: chain.id))
        }
      }
    }

    let tokenCount = restype.count
    let atomCount = tokenOfAtom.count

    let refPos = MLXArray(positions, [atomCount, 3])
    let atomToToken = MLXArray(tokenOfAtom).asType(.float32)
    // ref_space_uid numbers each (chain, residue) pair on first appearance, which for a
    // polymer laid out in order is the token index. So it is atom_to_token_idx by
    // another name, and it is what makes v_lm mean "same residue".
    let (dLM, vLM, maskTrunked) = atomPairGeometry(
      refPos: refPos, referenceSpace: MLXArray(tokenOfAtom), atomCount: atomCount)

    let restypeOneHot = oneHot(restype, classes: ResidueTemplates.restypeClasses)
    let tensors: [String: MLXArray] = [
      "ref_pos": refPos,
      "ref_charge": MLXArray(charges),
      "ref_mask": MLXArray.ones([atomCount]),
      "ref_element": oneHot(elementIndices, classes: ResidueTemplates.elementClasses),
      "ref_atom_name_chars": oneHot(
        nameCharacters, classes: ResidueTemplates.atomNameClasses
      ).reshaped([atomCount, ResidueTemplates.atomNameLength * ResidueTemplates.atomNameClasses]),
      "atom_to_token_idx": atomToToken,
      // The two the confidence head needs and the structure path does not.
      "distogram_rep_atom_mask": MLXArray(representative),
      "atom_to_tokatom_idx": MLXArray(indexInToken).asType(.float32),
      "d_lm": dLM,
      "v_lm": vLM,
      "mask_trunked": maskTrunked,
      "restype": restypeOneHot,
      // With a dummy alignment the profile IS the query's one-hot and the mean deletion
      // count is zero. Both become real the day an a3m is plumbed through, and neither
      // is a claim that one was.
      "profile": restypeOneHot,
      "deletion_mean": MLXArray.zeros([tokenCount, 1]),
      "relp": relativePosition(
        asymID: asymID, entityID: entityID, symID: symID, residueIndex: residueIndex),
      // True for a polypeptide chain and false for anything with a ligand, a disulfide
      // or a covalent modification -- none of which reach here.
      "token_bonds": MLXArray.zeros([tokenCount, tokenCount]),
      "msa_features": MLX.concatenated(
        [restypeOneHot.expandedDimensions(axis: 0), MLXArray.zeros([1, tokenCount, 2])],
        axis: -1),
    ]

    let metadata = FeatureBundle.Metadata(
      schemaVersion: 1, name: name,
      sequence: expanded.map(\.sequence).joined(separator: "/"),
      tokenCount: tokenCount, atomCount: atomCount, atoms: atoms)
    return FeatureBundle(metadata: metadata, tensors: tensors)
  }

  /// Trim, upper-case and label the chains, refusing what cannot be folded.
  static func normalize(_ requested: [Chain]) throws -> [Chain] {
    guard !requested.isEmpty else { throw ProtenixError.emptyChain("(none given)") }
    guard requested.count <= defaultChainIDs.count else {
      throw ProtenixError.tooManyChains(
        count: requested.count, limit: defaultChainIDs.count)
    }
    return try requested.map { chain in
      let sequence = chain.sequence.trimmingCharacters(in: .whitespacesAndNewlines)
        .uppercased()
      guard !sequence.isEmpty else { throw ProtenixError.emptyChain(chain.id) }
      return Chain(id: chain.id, sequence: sequence)
    }
  }

  /// `ord(c) - 32`, clipped to [0, 63], names padded to four characters.
  static func encodeName(_ name: String) -> [Int32] {
    var codes: [Int32] = []
    codes.reserveCapacity(ResidueTemplates.atomNameLength)
    for scalar in name.unicodeScalars.prefix(ResidueTemplates.atomNameLength) {
      codes.append(Int32(min(max(Int(scalar.value) - 32, 0), ResidueTemplates.atomNameClasses - 1)))
    }
    // A name shorter than four characters is padded with spaces, which encode to 0.
    while codes.count < ResidueTemplates.atomNameLength { codes.append(0) }
    return codes
  }

  static func oneHot(_ indices: [Int32], classes: Int) -> MLXArray {
    guard !indices.isEmpty else { return MLXArray.zeros([0, classes]) }
    let rows = indices.count
    var dense = [Float](repeating: 0, count: rows * classes)
    for (row, index) in indices.enumerated() {
      dense[row * classes + Int(index)] = 1
    }
    return MLXArray(dense, [rows, classes])
  }

  /// The windowed atom-pair features: `d_lm`, `v_lm` and the window mask.
  ///
  /// Each block of 32 query atoms reads a 128-wide key window centred on it, so the
  /// window reaches 48 atoms back and 48 forward. Padding is zeros on both axes, which
  /// matters for parity rather than for meaning: a padded query row compares equal to
  /// reference space 0, so `v_lm` reads 1 there. Those are exactly the positions
  /// `mask_trunked` zeroes, and they must still be *bitwise* what Python produces.
  static func atomPairGeometry(
    refPos: MLXArray, referenceSpace: MLXArray, atomCount: Int
  ) -> (MLXArray, MLXArray, MLXArray) {
    let trunking = LocalAttention.Trunking(
      atomCount: atomCount, queryWindow: queryWindow, keyWindow: keyWindow)

    let queries = LocalAttention.trunkQueries(refPos, trunking)  // [trunks, 32, 3]
    let keys = LocalAttention.trunkKeys(refPos, trunking)  // [trunks, 128, 3]
    let dLM = queries.expandedDimensions(axis: 2) - keys.expandedDimensions(axis: 1)

    // The same trunking, over a [n, 1] column so the shared helpers apply.
    let space = referenceSpace.reshaped([atomCount, 1]).asType(.float32)
    let querySpace = LocalAttention.trunkQueries(space, trunking).squeezed(axis: -1)
    let keySpace = LocalAttention.trunkKeys(space, trunking).squeezed(axis: -1)
    let vLM =
      (querySpace.asType(.int32).expandedDimensions(axis: 2)
        .== keySpace.asType(.int32).expandedDimensions(axis: 1))
      .expandedDimensions(axis: -1).asType(.float32)

    // The same geometry `windowBias` encodes as -inf, as a 0/1 mask.
    let mask = MLX.where(
      LocalAttention.windowBias(trunking) .== MLXArray(Float(0)),
      MLXArray(Float(1)), MLXArray(Float(0)))
    return (dLM, vLM, mask)
  }

  /// `relp`: `[rel_pos(66) | rel_token(66) | same_entity(1) | rel_chain(6)]`.
  ///
  /// One token is one residue for a standard amino acid, so the `rel_token` block is the
  /// degenerate case of the `rel_pos` one — 32 on the diagonal and the "different
  /// residue" sentinel everywhere else. It stops being degenerate the moment a ligand,
  /// which is tokenized per atom, enters.
  static func relativePosition(
    asymID: [Int32], entityID: [Int32], symID: [Int32], residueIndex: [Int32]
  ) -> MLXArray {
    let count = asymID.count
    let asym = MLXArray(asymID)
    let entity = MLXArray(entityID)
    let symmetry = MLXArray(symID)
    let residue = MLXArray(residueIndex)
    let tokens = MLXArray(Array(0..<Int32(count)))

    func pairwise(_ x: MLXArray) -> (MLXArray, MLXArray) {
      let rows = x.expandedDimensions(axis: 1)
      let columns = x.expandedDimensions(axis: 0)
      return (rows - columns, (rows .== columns).asType(.int32))
    }

    let (residueDelta, sameResidueIndex) = pairwise(residue)
    let (_, sameChain) = pairwise(asym)
    let (symmetryDelta, _) = pairwise(symmetry)
    let (tokenDelta, _) = pairwise(tokens)
    let (_, sameEntity) = pairwise(entity)
    // Two residues with the same index in *different* chains are not the same residue.
    let sameResidue = sameResidueIndex * sameChain

    let dResidue =
      MLX.clip(residueDelta + MLXArray(Int32(rMax)), min: MLXArray(Int32(0)), max: MLXArray(Int32(2 * rMax)))
      * sameChain + (MLXArray(Int32(1)) - sameChain) * MLXArray(Int32(2 * rMax + 1))
    let dToken =
      MLX.clip(tokenDelta + MLXArray(Int32(rMax)), min: MLXArray(Int32(0)), max: MLXArray(Int32(2 * rMax)))
      * sameResidue + (MLXArray(Int32(1)) - sameResidue) * MLXArray(Int32(2 * rMax + 1))
    let dChain =
      MLX.clip(symmetryDelta + MLXArray(Int32(sMax)), min: MLXArray(Int32(0)), max: MLXArray(Int32(2 * sMax)))
      * sameEntity + (MLXArray(Int32(1)) - sameEntity) * MLXArray(Int32(2 * sMax + 1))

    return MLX.concatenated(
      [
        pairOneHot(dResidue, classes: 2 * (rMax + 1), count: count),
        pairOneHot(dToken, classes: 2 * (rMax + 1), count: count),
        sameEntity.expandedDimensions(axis: -1).asType(.float32),
        pairOneHot(dChain, classes: 2 * (sMax + 1), count: count),
      ], axis: -1)
  }

  /// One-hot a `[n, n]` index matrix into `[n, n, classes]`.
  static func pairOneHot(_ indices: MLXArray, classes: Int, count: Int) -> MLXArray {
    let flat = indices.reshaped([count * count]).asType(.int32)
    let eye = MLXArray.eye(classes, dtype: .float32)
    return eye.take(flat, axis: 0).reshaped([count, count, classes])
  }
}
