import MLX

/// Algorithm 10: MSA pair-weighted averaging.
///
/// The pair representation supplies per-head weights over token pairs; each MSA row is
/// then re-mixed across tokens by those weights. The softmax is over the *second* token
/// axis of `b` (`dim=-2` on `[..., i, j, h]`), not over heads — normalizing the wrong
/// axis produces a plausible tensor with the wrong meaning.
public struct MSAPairWeightedAveraging {
  public let headCount: Int
  let hiddenWidth: Int

  let normM: LayerNormWeights
  let normZ: LayerNormWeights
  let linearV: AffineLinear
  let linearZ: AffineLinear
  let linearG: AffineLinear
  let linearOut: AffineLinear

  public init(store: WeightStore, path: String, headCount: Int) throws {
    self.headCount = headCount
    normM = try store.layerNorm("\(path).layernorm_m")
    normZ = try store.layerNorm("\(path).layernorm_z")
    linearV = try store.linear("\(path).linear_no_bias_mv")
    linearZ = try store.linear("\(path).linear_no_bias_z")
    linearG = try store.linear("\(path).linear_no_bias_mg")
    linearOut = try store.linear("\(path).linear_no_bias_out")
    hiddenWidth = linearV.weight.shape[0] / headCount
  }

  /// - Parameters:
  ///   - m: `[..., S, N, c_m]` MSA representation.
  ///   - z: `[..., N, N, c_z]` pair representation.
  public func callAsFunction(_ m: MLXArray, _ z: MLXArray) -> MLXArray {
    let normalized = TensorOps.layerNorm(m, normM)
    let shape = Array(normalized.shape.dropLast())

    // [..., S, N, H, C]
    let v = linearV(normalized).reshaped(shape + [headCount, hiddenWidth])
    let g = MLX.sigmoid(linearG(normalized))
      .reshaped(shape + [headCount, hiddenWidth])

    // [..., N, N, H], normalized over j.
    let b = linearZ(TensorOps.layerNorm(z, normZ))
    let w = MLX.softmax(b.asType(.float32), axis: -2)

    // einsum("...ijh,...mjhc->...mihc"): contract over j.
    // Reordered into one matmul per head: [..., H, i, j] @ [..., H, j, (m c)].
    let weights = TensorOps.permuteFinalDims(w, [2, 0, 1])  // [..., H, i, j]
    let values = v.asType(.float32)
    let leading = Array(values.shape.dropLast(4))
    let depth = values.shape[values.ndim - 4]
    let tokens = values.shape[values.ndim - 3]
    // [..., S, N, H, C] -> [..., H, N, S*C]
    let valueByHead = values.transposed(
      axes: axesToHeadMajor(rank: values.ndim)
    ).reshaped(leading + [headCount, tokens, depth * hiddenWidth])

    // [..., H, i, N] @ [..., H, N, S*C] -> [..., H, i, S*C]
    var mixed = MLX.matmul(weights, valueByHead)
    // -> [..., H, N, S, C] -> [..., S, N, H, C]
    mixed = mixed.reshaped(leading + [headCount, tokens, depth, hiddenWidth])
      .transposed(axes: axesFromHeadMajor(rank: mixed.ndim + 1))

    let gated = g.asType(.float32) * mixed
    let merged = gated.reshaped(shape + [headCount * hiddenWidth])
    return linearOut(merged.asType(m.dtype))
  }

  /// `[..., S, N, H, C] -> [..., H, S, N, C]`, then flattened by the caller.
  private func axesToHeadMajor(rank: Int) -> [Int] {
    let base = rank - 4  // axes: base=S, base+1=N, base+2=H, base+3=C
    return Array(0..<base) + [base + 2, base + 1, base, base + 3]
  }

  /// `[..., H, N, S, C] -> [..., S, N, H, C]`.
  private func axesFromHeadMajor(rank: Int) -> [Int] {
    let base = rank - 4  // axes: base=H, base+1=N, base+2=S, base+3=C
    return Array(0..<base) + [base + 2, base + 1, base, base + 3]
  }
}

/// Algorithm 8, lines 7-8: the per-row MSA update.
public struct MSAStack {
  let pairWeightedAveraging: MSAPairWeightedAveraging
  let transition: Transition

  public init(store: WeightStore, path: String, headCount: Int) throws {
    pairWeightedAveraging = try MSAPairWeightedAveraging(
      store: store, path: "\(path).msa_pair_weighted_averaging",
      headCount: headCount)
    transition = try Transition(store: store, path: "\(path).transition_m")
  }

  public func callAsFunction(_ m: MLXArray, _ z: MLXArray) -> MLXArray {
    var value = m + pairWeightedAveraging(m, z)
    value = value + transition(value)
    return value
  }
}

/// One MSA block: communicate MSA into pairs, update the MSA, then update the pairs.
///
/// The final block has no MSA stack at all — upstream drops it because `m` is never
/// read again, so those weights are simply absent from the checkpoint. Detected here
/// from the weights rather than from an index, so a stack of any depth loads correctly.
public struct MSABlock {
  let outerProductMean: OuterProductMean
  let msaStack: MSAStack?
  let pairStack: PairformerBlock

  public var isLastBlock: Bool { msaStack == nil }

  public init(store: WeightStore, path: String, pairHeadCount: Int, msaHeadCount: Int)
    throws
  {
    outerProductMean = try OuterProductMean(
      store: store, path: "\(path).outer_product_mean_msa")
    let stackPath = "\(path).msa_stack"
    msaStack =
      store.names(withPrefix: "\(stackPath).").isEmpty
      ? nil
      : try MSAStack(store: store, path: stackPath, headCount: msaHeadCount)
    // c_s = 0 here: the pair stack has no single track, so its AttentionPairBias and
    // single transition are absent. `PairformerBlock` detects that from the weights.
    pairStack = try PairformerBlock(
      store: store, path: "\(path).pair_stack", headCount: pairHeadCount,
      pairHeadCount: pairHeadCount)
  }

  /// - Returns: the updated MSA (nil on the last block) and pair representations.
  public func callAsFunction(_ m: MLXArray, _ z: MLXArray) -> (MLXArray?, MLXArray) {
    var pair = z + outerProductMean(m)
    var msa: MLXArray? = nil
    if let msaStack {
      msa = msaStack(m, pair)
    }
    (_, pair) = pairStack(nil, pair)
    return (msa, pair)
  }
}

/// Algorithm 8: the MSA module — 1 block in tiny/mini, 4 in base and v2.
///
/// Only the trunk half is ported. Building `m` from raw MSA features (one-hot residues,
/// deletion counts, row sampling) belongs to the featurizer, which is not written yet;
/// this takes an already-embedded `m`.
public struct MSAModule {
  public let blocks: [MSABlock]
  let linearM: AffineLinear
  let linearS: AffineLinear

  public var blockCount: Int { blocks.count }

  public init(
    store: WeightStore, path: String, blockCount: Int, pairHeadCount: Int,
    msaHeadCount: Int
  ) throws {
    linearM = try store.linear("\(path).linear_no_bias_m")
    linearS = try store.linear("\(path).linear_no_bias_s")
    blocks = try (0..<blockCount).map { index in
      try MSABlock(
        store: store, path: "\(path).blocks.\(index)", pairHeadCount: pairHeadCount,
        msaHeadCount: msaHeadCount)
    }
  }

  /// Embed the MSA feature block and the single inputs into the MSA representation.
  ///
  /// - Parameters:
  ///   - msaFeatures: `[..., S, N, 34]` — 32 one-hot residue classes, has-deletion,
  ///     deletion-value, in that order.
  ///   - singleInputs: `[..., N, c_s_inputs]`, broadcast across every MSA row.
  public func embed(msaFeatures: MLXArray, singleInputs: MLXArray) -> MLXArray {
    linearM(msaFeatures) + linearS(singleInputs).expandedDimensions(axis: -3)
  }

  public func callAsFunction(_ m: MLXArray, _ z: MLXArray) -> MLXArray {
    var msa = m
    var pair = z
    for block in blocks {
      let (updated, updatedPair) = block(msa, pair)
      pair = updatedPair
      if let updated { msa = updated }
      MLX.eval(pair, msa)
    }
    return pair
  }
}
