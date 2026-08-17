import MLX

/// ReLU as MLX exposes no free `relu` function on the module.
private func relu(_ x: MLXArray) -> MLXArray { MLX.maximum(x, MLXArray(Float(0))) }

/// Atom↔token scatter/gather shared by the atom encoder and decoder.
public enum AtomOps {
  /// Broadcast a per-token tensor to per-atom by gathering each atom's token row.
  ///
  /// `x_token[..., atom_to_token_idx, :]` — a plain gather along the token axis.
  public static func broadcastTokenToAtom(
    _ xToken: MLXArray, atomToToken: MLXArray
  ) -> MLXArray {
    xToken.take(atomToToken.asType(.int32), axis: -2)
  }

  /// Mean-aggregate a per-atom tensor into per-token.
  ///
  /// Expressed as a one-hot contraction rather than a scatter: `token[t]` is the mean
  /// over atoms assigned to `t`. Correct and branch-free, at the cost of an
  /// `[N_atom, N_token]` one-hot — acceptable at the token counts targeted here.
  public static func aggregateAtomToToken(
    _ xAtom: MLXArray, atomToToken: MLXArray, tokenCount: Int
  ) -> MLXArray {
    let classes = MLXArray((0..<tokenCount).map { Int32($0) })
    // [N_atom, N_token]
    let oneHot = (atomToToken.asType(.int32).expandedDimensions(axis: -1) .== classes)
      .asType(.float32)
    let counts = oneHot.sum(axis: 0)  // [N_token]
    // [..., N_token, d] = [N_token, N_atom] @ [..., N_atom, d]
    let summed = MLX.matmul(oneHot.swappedAxes(-1, -2), xAtom.asType(.float32))
    let safeCounts = MLX.maximum(counts, MLXArray(Float(1)))
    return (summed / safeCounts.expandedDimensions(axis: -1)).asType(xAtom.dtype)
  }

  /// Gather a token-pair tensor into the atom-pair blocked layout.
  ///
  /// `y[..., t, i, j, :] = z_token[..., idxQ[t, i], idxK[t, j], :]`, where the trunked
  /// index arrays come from windowing `atom_to_token_idx`. Padding slots index token 0
  /// and are masked out downstream, exactly as upstream leaves them.
  public static func broadcastTokenToLocalAtomPair(
    _ zToken: MLXArray, atomToToken: MLXArray, trunking: LocalAttention.Trunking
  ) -> MLXArray {
    let indices = atomToToken.asType(.int32).expandedDimensions(axis: -1)
    let idxQ = LocalAttention.trunkQueries(indices, trunking).squeezed(axis: -1)
    let idxK = LocalAttention.trunkKeys(indices, trunking).squeezed(axis: -1)
    let trunks = trunking.trunkCount
    let queryWindow = trunking.queryWindow
    let keyWindow = trunking.keyWindow
    let tokenCount = zToken.shape[zToken.ndim - 2]
    let width = zToken.shape[zToken.ndim - 1]

    // Flatten the [N_token, N_token] pair grid so a single flat index gathers it.
    let leading = Array(zToken.shape.dropLast(3))
    let flat = zToken.reshaped(leading + [tokenCount * tokenCount, width])

    let qExpanded = idxQ.expandedDimensions(axis: 2)  // [trunks, nQ, 1]
    let kExpanded = idxK.expandedDimensions(axis: 1)  // [trunks, 1, nK]
    let flatIndex = (qExpanded * Int32(tokenCount) + kExpanded)  // [trunks, nQ, nK]
      .reshaped([trunks * queryWindow * keyWindow])
    let gathered = flat.take(flatIndex, axis: -2)
    return gathered.reshaped(leading + [trunks, queryWindow, keyWindow, width])
  }
}

/// Algorithm 5: the atom attention encoder.
///
/// Runs in two configurations distinguished by whether coordinates are supplied:
///
/// * `has_coords == true` (inside the diffusion module) folds in the trunk single/pair
///   embeddings and the noisy atom positions `r_l`, and carries a sample axis.
/// * `has_coords == false` (input feature embedder) builds a coordinate-free atom
///   embedding; the `layernorm_s`/`linear_r`/etc. weights are simply absent.
///
/// `d_lm`, `v_lm` and `maskTrunked` are geometry features (pairwise offsets, same-space
/// mask, window validity) with no learned parameters — they arrive precomputed from the
/// feature bundle, as in the boltz-mlx pattern.
public struct AtomAttentionEncoder {
  public let hasCoords: Bool
  public let queryWindow: Int
  public let keyWindow: Int

  let linearRefPos: AffineLinear
  let linearRefCharge: AffineLinear
  let linearF: AffineLinear
  let linearD: AffineLinear
  let linearInvD: AffineLinear
  let linearV: AffineLinear
  let linearCL: AffineLinear
  let linearCM: AffineLinear
  let smallMLP: [AffineLinear]
  let linearQ: AffineLinear
  let transformer: AtomTransformer

  // has_coords only:
  let normS: LayerNormWeights?
  let linearS: AffineLinear?
  let normZ: LayerNormWeights?
  let linearZ: AffineLinear?
  let linearR: AffineLinear?

  public init(
    store: WeightStore, path: String, blockCount: Int, headCount: Int,
    queryWindow: Int = 32, keyWindow: Int = 128
  ) throws {
    self.queryWindow = queryWindow
    self.keyWindow = keyWindow
    linearRefPos = try store.linear("\(path).linear_no_bias_ref_pos")
    linearRefCharge = try store.linear("\(path).linear_no_bias_ref_charge")
    linearF = try store.linear("\(path).linear_no_bias_f")
    linearD = try store.linear("\(path).linear_no_bias_d")
    linearInvD = try store.linear("\(path).linear_no_bias_invd")
    linearV = try store.linear("\(path).linear_no_bias_v")
    linearCL = try store.linear("\(path).linear_no_bias_cl")
    linearCM = try store.linear("\(path).linear_no_bias_cm")
    // small_mlp is nn.Sequential(ReLU, Linear, ReLU, Linear, ReLU, Linear) — the
    // three Linears live at indices 1, 3, 5.
    smallMLP = try [1, 3, 5].map { try store.linear("\(path).small_mlp.\($0)") }
    linearQ = try store.linear("\(path).linear_no_bias_q")
    transformer = try AtomTransformer(
      store: store, path: "\(path).atom_transformer", blockCount: blockCount,
      headCount: headCount, queryWindow: queryWindow, keyWindow: keyWindow)

    self.hasCoords = !store.names(
      withPrefix: "\(path).linear_no_bias_r.weight").isEmpty
    if hasCoords {
      normS = try store.layerNorm("\(path).layernorm_s")
      linearS = try store.linear("\(path).linear_no_bias_s")
      normZ = try store.layerNorm("\(path).layernorm_z")
      linearZ = try store.linear("\(path).linear_no_bias_z")
      linearR = try store.linear("\(path).linear_no_bias_r")
    } else {
      normS = nil; linearS = nil; normZ = nil; linearZ = nil; linearR = nil
    }
  }

  /// The atom encoder's outputs, in upstream's order.
  public struct Output {
    public let a: MLXArray      // [..., (S), N_token, c_token]
    public let qSkip: MLXArray  // [..., (S), N_atom, c_atom]
    public let cSkip: MLXArray  // [..., (S), N_atom, c_atom]
    public let pSkip: MLXArray  // [..., (S), trunks, nQ, nK, c_atompair]
  }

  public struct Features {
    public let refPos: MLXArray
    public let refCharge: MLXArray
    public let refMask: MLXArray
    public let refElement: MLXArray
    public let refAtomNameChars: MLXArray
    public let atomToToken: MLXArray
    public let dLM: MLXArray
    public let vLM: MLXArray
    public let maskTrunked: MLXArray

    public init(
      refPos: MLXArray, refCharge: MLXArray, refMask: MLXArray, refElement: MLXArray,
      refAtomNameChars: MLXArray, atomToToken: MLXArray, dLM: MLXArray, vLM: MLXArray,
      maskTrunked: MLXArray
    ) {
      self.refPos = refPos; self.refCharge = refCharge; self.refMask = refMask
      self.refElement = refElement; self.refAtomNameChars = refAtomNameChars
      self.atomToToken = atomToToken; self.dLM = dLM; self.vLM = vLM
      self.maskTrunked = maskTrunked
    }
  }

  /// - Parameters:
  ///   - r: `[..., S, N_atom, 3]` noisy positions (has_coords only).
  ///   - s: `[..., S, N_token, c_s]` trunk single (has_coords only).
  ///   - z: `[..., N_token, N_token, c_z]` trunk pair (has_coords only).
  public func callAsFunction(
    _ features: Features, r: MLXArray? = nil, s: MLXArray? = nil, z: MLXArray? = nil,
    tokenCount: Int
  ) -> Output {
    let refMaskColumn = features.refMask.expandedDimensions(axis: -1)
    let n = features.refPos.shape[features.refPos.ndim - 2]
    let trunking = LocalAttention.Trunking(
      atomCount: n, queryWindow: queryWindow, keyWindow: keyWindow)

    // Per-atom conditioning c_l.
    var cL =
      linearRefPos(features.refPos)
      + linearRefCharge(MLX.asinh(features.refCharge).expandedDimensions(axis: -1))
    let atomFeatures = MLX.concatenated(
      [refMaskColumn, features.refElement, features.refAtomNameChars], axis: -1)
    cL = cL + linearF(atomFeatures.asType(cL.dtype))
    cL = cL * refMaskColumn

    // Per-atom-pair conditioning p_lm, in blocked window layout.
    let maskColumn = features.maskTrunked.expandedDimensions(axis: -1)
    var pLM = (linearD(features.dLM) * features.vLM) * maskColumn
    let invSquare = 1 / (1 + (features.dLM * features.dLM).sum(axis: -1, keepDims: true))
    pLM = pLM + linearInvD(invSquare) * features.vLM
    pLM = pLM + linearV(features.vLM.asType(pLM.dtype))

    if hasCoords, let z, let linearZ, let normZ {
      let tokenPair = linearZ(TensorOps.layerNorm(z, normZ))
      let atomPair = AtomOps.broadcastTokenToLocalAtomPair(
        tokenPair, atomToToken: features.atomToToken, trunking: trunking)
      pLM = pLM.expandedDimensions(axis: -5) + atomPair
    }

    var qL: MLXArray
    if hasCoords, let r, let s, let normS, let linearS, let linearR {
      let single = AtomOps.broadcastTokenToAtom(
        linearS(TensorOps.layerNorm(s, normS)), atomToToken: features.atomToToken)
      cL = cL.expandedDimensions(axis: -3) + single
      qL = cL + linearR(r)
    } else {
      qL = cL
    }

    // Fold the single conditioning into the pair, over the window.
    let cLQ = LocalAttention.trunkQueries(cL, trunking).expandedDimensions(axis: -2)
    let cLK = LocalAttention.trunkKeys(cL, trunking).expandedDimensions(axis: -3)
    pLM = pLM + linearCL(relu(cLQ)) + linearCM(relu(cLK))
    pLM = pLM + runSmallMLP(pLM)

    let updated = transformer(qL, cL, pLM)
    let a = AtomOps.aggregateAtomToToken(
      relu(linearQ(updated)), atomToToken: features.atomToToken,
      tokenCount: tokenCount)
    return Output(a: a, qSkip: updated, cSkip: cL, pSkip: pLM)
  }

  private func runSmallMLP(_ x: MLXArray) -> MLXArray {
    var value = smallMLP[0](relu(x))
    value = smallMLP[1](relu(value))
    value = smallMLP[2](relu(value))
    return value
  }
}

/// Algorithm 6: the atom attention decoder — turns token activations back into a
/// per-atom coordinate update.
public struct AtomAttentionDecoder {
  public let queryWindow: Int
  public let keyWindow: Int

  let linearA: AffineLinear
  let normQ: LayerNormWeights
  let linearOut: AffineLinear
  let transformer: AtomTransformer

  public init(
    store: WeightStore, path: String, blockCount: Int, headCount: Int,
    queryWindow: Int = 32, keyWindow: Int = 128
  ) throws {
    self.queryWindow = queryWindow
    self.keyWindow = keyWindow
    linearA = try store.linear("\(path).linear_no_bias_a")
    normQ = try store.layerNorm("\(path).layernorm_q")
    linearOut = try store.linear("\(path).linear_no_bias_out")
    transformer = try AtomTransformer(
      store: store, path: "\(path).atom_transformer", blockCount: blockCount,
      headCount: headCount, queryWindow: queryWindow, keyWindow: keyWindow)
  }

  /// - Parameters:
  ///   - a: `[..., S, N_token, c_token]` token activations.
  ///   - qSkip/cSkip/pSkip: the encoder's carried skip connections.
  /// - Returns: `[..., S, N_atom, 3]` coordinate update.
  public func callAsFunction(
    _ a: MLXArray, atomToToken: MLXArray, qSkip: MLXArray, cSkip: MLXArray,
    pSkip: MLXArray
  ) -> MLXArray {
    let q =
      AtomOps.broadcastTokenToAtom(linearA(a), atomToToken: atomToToken) + qSkip
    let updated = transformer(q, cSkip, pSkip)
    return linearOut(TensorOps.layerNorm(updated, normQ))
  }
}
