import MLX

/// Algorithm 31: the confidence head — pLDDT, PAE, PDE and resolved, per prediction.
///
/// Runs *after* the sampler, on the coordinates it produced: it re-embeds the predicted
/// token-token distances, pushes them through its own 4-block Pairformer, and reads four
/// heads off the result. So it is not a second opinion about the fold — it is a
/// prediction about the fold that was just made, which is why the coordinates are an
/// input rather than something it recomputes.
///
/// Two details worth knowing, both of which a port can get silently wrong:
///
/// * **The distance matrix is over *representative* atoms, not all atoms.** One atom
///   stands for each token — CB, or CA for glycine — selected by
///   `distogram_rep_atom_mask`. Using every atom would produce an `[N_atom, N_atom]`
///   matrix where the network expects `[N_token, N_token]`, which for a real protein is
///   an eightfold error in the wrong direction.
/// * **pLDDT and resolved are per ATOM, and each atom reads its own weight matrix.**
///   `plddt_weight` is `[max_atoms_per_token, c_s, bins]`, indexed by
///   `atom_to_tokatom_idx` — an atom's position *within its residue*. The single
///   representation is broadcast from tokens to atoms first, so a residue's fifth atom
///   is scored by the fifth weight matrix. Indexing that by anything else (the atom's
///   global index, its element) yields plausible-looking confidence that means nothing.
///
/// The bin edges come from the artifact rather than being recomputed here: `upper_bins`
/// ends in a 1e6 sentinel that the exporter deliberately keeps at float32 because it
/// becomes `inf` in float16, and re-deriving it would quietly reintroduce that.
public struct ConfidenceHead {
  let linearS1: AffineLinear
  let linearS2: AffineLinear
  let linearDistance: AffineLinear
  let linearDistanceNoOneHot: AffineLinear
  let pairformer: PairformerStack
  let linearPAE: AffineLinear
  let linearPDE: AffineLinear
  let inputSTrunkNorm: LayerNormWeights
  let paeNorm: LayerNormWeights
  let pdeNorm: LayerNormWeights
  let plddtNorm: LayerNormWeights
  let resolvedNorm: LayerNormWeights
  /// `[max_atoms_per_token, c_s, plddt_bins]`.
  let plddtWeight: MLXArray
  /// `[max_atoms_per_token, c_s, 2]`.
  let resolvedWeight: MLXArray
  let lowerBins: MLXArray
  let upperBins: MLXArray

  /// The head's four raw outputs, before any bin arithmetic.
  ///
  /// Exposed separately from ``Scores`` because this is the boundary PyTorch is compared
  /// against: converting to Angstroms and 0-100 first would fold two steps into one
  /// number and make a disagreement ambiguous between the network and the conversion.
  public struct Logits {
    /// `[N_atom, plddt_bins]`.
    public let plddt: MLXArray
    /// `[N_token, N_token, pae_bins]`.
    public let pae: MLXArray
    /// `[N_token, N_token, pde_bins]`.
    public let pde: MLXArray
    /// `[N_atom, 2]`.
    public let resolved: MLXArray
  }

  /// What the head reports, in the units a caller wants rather than as logits.
  public struct Scores {
    /// Per-atom pLDDT on 0-100, the array a viewer colours by.
    public let plddt: MLXArray
    /// Mean of `plddt` — the single number that summarises a fold.
    public let meanPLDDT: Float
    /// `[N_token, N_token]` predicted aligned error in Angstroms.
    public let pae: MLXArray
    /// `[N_token, N_token]` predicted distance error in Angstroms.
    public let pde: MLXArray
    /// Per-atom probability that the atom is resolved.
    public let resolved: MLXArray
  }

  public init(
    store: WeightStore, path: String,
    configuration: ProtenixModelConfiguration.ConfidenceSection,
    headCount: Int, pairHeadCount: Int
  ) throws {
    linearS1 = try store.linear("\(path).linear_no_bias_s1")
    linearS2 = try store.linear("\(path).linear_no_bias_s2")
    linearDistance = try store.linear("\(path).linear_no_bias_d")
    linearDistanceNoOneHot = try store.linear("\(path).linear_no_bias_d_wo_onehot")
    pairformer = try PairformerStack(
      store: store, path: "\(path).pairformer_stack",
      blockCount: configuration.nBlocks, headCount: headCount,
      pairHeadCount: pairHeadCount)
    linearPAE = try store.linear("\(path).linear_no_bias_pae")
    linearPDE = try store.linear("\(path).linear_no_bias_pde")
    inputSTrunkNorm = try store.layerNorm("\(path).input_strunk_ln")
    paeNorm = try store.layerNorm("\(path).pae_ln")
    pdeNorm = try store.layerNorm("\(path).pde_ln")
    plddtNorm = try store.layerNorm("\(path).plddt_ln")
    resolvedNorm = try store.layerNorm("\(path).resolved_ln")
    plddtWeight = try store.raw("\(path).plddt_weight")
    resolvedWeight = try store.raw("\(path).resolved_weight")
    lowerBins = try store.raw("\(path).lower_bins")
    upperBins = try store.raw("\(path).upper_bins")
  }

  /// Score one predicted structure.
  ///
  /// - Parameters:
  ///   - coordinates: `[N_atom, 3]`, as `ProtenixPredictor.fold` returns them.
  ///   - sInputs: `[1, N_token, c_s_inputs]` from the input embedder.
  ///   - sTrunk: `[1, N_token, c_s]` and `zTrunk` `[1, N_token, N_token, c_z]` from the trunk.
  ///   - representativeAtoms: `distogram_rep_atom_mask`, `[N_atom]`.
  ///   - atomToTokenAtom: `atom_to_tokatom_idx`, `[N_atom]`.
  public func callAsFunction(
    coordinates: MLXArray, sInputs: MLXArray, sTrunk: MLXArray, zTrunk: MLXArray,
    atomToToken: MLXArray, representativeAtoms: MLXArray, atomToTokenAtom: MLXArray,
    tokenCount: Int
  ) -> Scores {
    let logits = self.logits(
      coordinates: coordinates, sInputs: sInputs, sTrunk: sTrunk, zTrunk: zTrunk,
      atomToToken: atomToToken, representativeAtoms: representativeAtoms,
      atomToTokenAtom: atomToTokenAtom)

    // Logits to the numbers a caller wants. pLDDT's bins span 0-1 and are reported on
    // 0-100, which is the convention every viewer's B-factor column assumes.
    let plddt = expectation(logits.plddt, minBin: 0, maxBin: 1) * MLXArray(Float(100))
    let pae = expectation(logits.pae, minBin: 0, maxBin: 32)
    let pde = expectation(logits.pde, minBin: 0, maxBin: 32)
    let resolved = MLX.softMax(logits.resolved, axis: -1)[.ellipsis, 1]
    MLX.eval(plddt, pae, pde, resolved)

    return Scores(
      plddt: plddt.reshaped([plddt.size]),
      meanPLDDT: plddt.mean().item(Float.self),
      pae: pae.reshaped([tokenCount, tokenCount]),
      pde: pde.reshaped([tokenCount, tokenCount]),
      resolved: resolved.reshaped([resolved.size]))
  }

  /// The head's raw logits — everything up to the bin arithmetic.
  public func logits(
    coordinates: MLXArray, sInputs: MLXArray, sTrunk: MLXArray, zTrunk: MLXArray,
    atomToToken: MLXArray, representativeAtoms: MLXArray, atomToTokenAtom: MLXArray
  ) -> Logits {
    // Upstream clamps s_trunk to +-512 before its own layer norm. Not cosmetic: the
    // trunk's single representation is unbounded, and a recycled outlier would otherwise
    // dominate the norm's variance.
    let clamped = MLX.clip(sTrunk, min: MLXArray(Float(-512)), max: MLXArray(Float(512)))
    let single = TensorOps.layerNorm(clamped, inputSTrunkNorm)

    // One atom stands for each token. Resolved to indices on the host: the mask is
    // [N_atom] of 0/1, MLX has no nonzero, and the order matters — token i's
    // representative must land at row i, which it does because atoms are laid out in
    // token order.
    let flags = representativeAtoms.asType(.float32).asArray(Float.self)
    let representativeIndices = MLXArray(
      flags.enumerated().compactMap { $0.element > 0.5 ? Int32($0.offset) : nil })
    let representativeCoordinates = coordinates.take(representativeIndices, axis: 0)
    let distances = pairwiseDistances(representativeCoordinates)

    // z starts from the input embedding, outer-summed into a pair tensor, plus the trunk.
    //
    // The axes are not interchangeable: upstream broadcasts s1 over the ROW axis
    // (`[..., None, :, :]`) and s2 over the column axis (`[..., None, :]`), so entry
    // (i, j) is s1[j] + s2[i]. Swapping them transposes the pair initialisation, which
    // is not symmetric, and every one of the head's four outputs comes out wrong
    // together — the symptom that led here.
    var pair =
      linearS1(sInputs).expandedDimensions(axis: -3)
      + linearS2(sInputs).expandedDimensions(axis: -2)
    pair = pair + zTrunk
    // The distance embedding, both ways upstream embeds it: soft-binned one-hot, and the
    // raw distance through a width-1 projection.
    let binned = distanceOneHot(distances)
    pair = pair + linearDistance(binned)
    pair = pair + linearDistanceNoOneHot(distances.expandedDimensions(axis: -1))

    let (updatedSingle, updatedPair) = pairformer(single, pair)
    let s = updatedSingle ?? single
    MLX.eval(s, updatedPair)

    let paeLogits = linearPAE(TensorOps.layerNorm(updatedPair, paeNorm))
    // PDE is symmetrized first: an error in the DISTANCE between i and j is one number,
    // not two, so the head is fed z + z^T rather than z.
    let pdeLogits = linearPDE(
      TensorOps.layerNorm(updatedPair + updatedPair.swappedAxes(-2, -3), pdeNorm))

    // Tokens to atoms, so every atom is scored by the weight matrix for its own position
    // within its residue.
    let atomSingle = s.take(atomToToken.asType(.int32), axis: -2)
    let plddtLogits = perAtomHead(
      TensorOps.layerNorm(atomSingle, plddtNorm), weight: plddtWeight,
      index: atomToTokenAtom)
    let resolvedLogits = perAtomHead(
      TensorOps.layerNorm(atomSingle, resolvedNorm), weight: resolvedWeight,
      index: atomToTokenAtom)
    MLX.eval(plddtLogits, paeLogits, pdeLogits, resolvedLogits)

    return Logits(
      plddt: plddtLogits, pae: paeLogits, pde: pdeLogits, resolved: resolvedLogits)
  }

  /// Euclidean distances between every pair of rows, upstream's `torch.cdist`.
  ///
  /// Computed from the squared-norm expansion rather than by materialising the
  /// difference tensor: for 2000 tokens the latter is a `[2000, 2000, 3]` intermediate.
  /// Clamped at zero before the root because the expansion can go slightly negative on
  /// the diagonal in float32, and `sqrt` of -1e-7 is NaN, which then poisons every bin.
  func pairwiseDistances(_ x: MLXArray) -> MLXArray {
    let squared = (x * x).sum(axis: -1)
    let cross = x.matmul(x.swappedAxes(-2, -1))
    let expanded =
      squared.expandedDimensions(axis: -1) + squared.expandedDimensions(axis: -2)
      - 2 * cross
    return MLX.sqrt(MLX.maximum(expanded, MLXArray(Float(0))))
  }

  /// Upstream's `one_hot(x, lower_bins, upper_bins)`: strictly inside each bin.
  ///
  /// Deliberately not `>=` on the lower edge — upstream uses `>` on both sides, so a
  /// distance exactly on a bin edge falls in NEITHER bin and that row is all zeros. Kept
  /// as-is: the weights were trained against this function, edge case included.
  func distanceOneHot(_ distances: MLXArray) -> MLXArray {
    let x = distances.expandedDimensions(axis: -1)
    return MLX.logicalAnd(x .> lowerBins, x .< upperBins).asType(.float32)
  }

  /// `einsum("...nc,ncb->...nb", x, weight[index])` — each atom against its own matrix.
  func perAtomHead(_ x: MLXArray, weight: MLXArray, index: MLXArray) -> MLXArray {
    let gathered = weight.take(index.asType(.int32), axis: 0)  // [N_atom, c_s, bins]
    let rows = x.reshaped([gathered.shape[0], 1, gathered.shape[1]])  // [N_atom, 1, c_s]
    return rows.matmul(gathered).squeezed(axis: -2)  // [N_atom, bins]
  }

  /// Softmax over bins, then the expectation under the bin centres.
  ///
  /// Bin centres are `min + (i + 0.5) * width` — the midpoints of `no_bins` equal bins
  /// spanning `[min, max)`, which is what upstream's `get_bin_centers` builds.
  func expectation(_ logits: MLXArray, minBin: Float, maxBin: Float) -> MLXArray {
    let bins = logits.shape[logits.ndim - 1]
    let width = (maxBin - minBin) / Float(bins)
    let centres = MLXArray(
      (0..<bins).map { minBin + (Float($0) + 0.5) * width })
    let probabilities = MLX.softMax(logits, axis: -1)
    return (probabilities * centres).sum(axis: -1)
  }
}
