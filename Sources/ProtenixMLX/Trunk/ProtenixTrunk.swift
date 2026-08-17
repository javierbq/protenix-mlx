import MLX

/// Algorithm 1, lines 1-13: the trunk that turns input features into the single and
/// pair embeddings the diffusion module folds against.
///
/// Structure-path only. The template embedder is skipped when it has no blocks (the
/// v0.5.0 mini/tiny variants) — for those, `templateEmbedder` is nil and templates are
/// simply not part of the graph. Constraints and ESM are out of scope.
public struct ProtenixTrunk {
  public let cycleCount: Int

  let inputEmbedder: InputFeatureEmbedder
  let relativePositionEncoding: RelativePositionEncoding
  let msaModule: MSAModule
  let pairformer: PairformerStack

  let linearSInit: AffineLinear
  let linearZInit1: AffineLinear
  let linearZInit2: AffineLinear
  let linearTokenBond: AffineLinear
  let linearZCycle: AffineLinear
  let linearS: AffineLinear
  let normZCycle: LayerNormWeights
  let normS: LayerNormWeights

  /// Build the trunk from the artifact's own architecture contract. This is what a
  /// runtime uses in production — every dimension comes from `config.json`.
  public init(store: WeightStore, cycleCount: Int) throws {
    let config = store.configuration.model
    // Triangle attention's head count is NOT the block's attention head count. Upstream
    // uses no_heads_pair, which defaults to 4 and becomes c_z / c_hidden_pair_att (32)
    // only when hidden_scale_up is on (protenix-v2). Passing the 16 attention heads
    // here builds triangle-bias tensors of the wrong width.
    let pairHeads = config.pairformer.hiddenScaleUp ? config.pairformer.cZ / 32 : 4
    try self.init(
      store: store, cycleCount: cycleCount,
      inputEmbedderBlocks: config.diffusionModule.atomEncoder.nBlocks,
      inputEmbedderHeads: config.diffusionModule.atomEncoder.nHeads,
      msaBlocks: config.msaModule.nBlocks,
      pairformerBlocks: config.pairformer.nBlocks,
      pairformerHeads: config.pairformer.nHeads,
      pairTriangleHeads: pairHeads,
      rMax: config.relativePositionEncoding.rMax,
      sMax: config.relativePositionEncoding.sMax)
  }

  /// Build the trunk from explicit dimensions. Used by tests, whose fixtures carry no
  /// full config, and by the convenience initializer above.
  public init(
    store: WeightStore, cycleCount: Int, inputEmbedderBlocks: Int,
    inputEmbedderHeads: Int, msaBlocks: Int, pairformerBlocks: Int,
    pairformerHeads: Int, pairTriangleHeads: Int, rMax: Int, sMax: Int
  ) throws {
    self.cycleCount = cycleCount

    inputEmbedder = try InputFeatureEmbedder(
      store: store, path: "input_embedder",
      blockCount: inputEmbedderBlocks, headCount: inputEmbedderHeads)
    relativePositionEncoding = try RelativePositionEncoding(
      store: store, path: "relative_position_encoding", rMax: rMax, sMax: sMax)
    msaModule = try MSAModule(
      store: store, path: "msa_module", blockCount: msaBlocks,
      pairHeadCount: pairTriangleHeads, msaHeadCount: 8)
    pairformer = try PairformerStack(
      store: store, path: "pairformer_stack", blockCount: pairformerBlocks,
      headCount: pairformerHeads, pairHeadCount: pairTriangleHeads)

    linearSInit = try store.linear("linear_no_bias_sinit")
    linearZInit1 = try store.linear("linear_no_bias_zinit1")
    linearZInit2 = try store.linear("linear_no_bias_zinit2")
    linearTokenBond = try store.linear("linear_no_bias_token_bond")
    linearZCycle = try store.linear("linear_no_bias_z_cycle")
    linearS = try store.linear("linear_no_bias_s")
    normZCycle = try store.layerNorm("layernorm_z_cycle")
    normS = try store.layerNorm("layernorm_s")
  }

  /// The trunk's outputs, feeding the diffusion module.
  public struct Output {
    public let sInputs: MLXArray  // [..., N_token, c_s_inputs]
    public let s: MLXArray        // [..., N_token, c_s]
    public let z: MLXArray        // [..., N_token, N_token, c_z]
  }

  /// - Parameters:
  ///   - atomFeatures: coordinate-free atom features for the input embedder.
  ///   - restype/profile/deletionMean: per-token input features.
  ///   - relativeFeatures: `[..., N, N, featureWidth]` relative-position one-hot.
  ///   - tokenBonds: `[..., N, N]` bonded-pair indicator.
  ///   - msaFeatures: `[..., S, N, 34]` sampled MSA feature block.
  ///   - progress: called after each recycle; return false to cancel. The trunk is
  ///     roughly half a base-model fold's wall clock, so a cancel that could only be
  ///     observed by the sampler would be ignored for minutes.
  public func callAsFunction(
    atomFeatures: AtomAttentionEncoder.Features, restype: MLXArray, profile: MLXArray,
    deletionMean: MLXArray, relativeFeatures: MLXArray, tokenBonds: MLXArray,
    msaFeatures: MLXArray, tokenCount: Int, progress: FoldProgressHandler? = nil
  ) throws -> Output {
    let sInputs = inputEmbedder(
      features: atomFeatures, restype: restype, profile: profile,
      deletionMean: deletionMean, tokenCount: tokenCount)

    let sInit = linearSInit(sInputs)
    var zInit =
      linearZInit1(sInit).expandedDimensions(axis: -2)
      + linearZInit2(sInit).expandedDimensions(axis: -3)
    zInit = zInit + relativePositionEncoding(relativeFeatures)
    zInit = zInit + linearTokenBond(tokenBonds.expandedDimensions(axis: -1))

    var z = MLXArray.zeros(like: zInit)
    var s = MLXArray.zeros(like: sInit)

    for cycle in 0..<cycleCount {
      z = zInit + linearZCycle(TensorOps.layerNorm(z, normZCycle))
      // Template embedder skipped: nil for the variants in scope.
      let m = msaModule.embed(msaFeatures: msaFeatures, singleInputs: sInputs)
      z = msaModule(m, z)
      s = sInit + linearS(TensorOps.layerNorm(s, normS))
      let (updatedS, updatedZ) = pairformer(s, z)
      s = updatedS!
      z = updatedZ
      MLX.eval(s, z)
      if let progress,
        !progress(FoldProgress(phase: .trunk, step: cycle + 1, total: cycleCount))
      {
        throw ProtenixError.cancelled(phase: "trunk", step: cycle + 1)
      }
    }
    return Output(sInputs: sInputs, s: s, z: z)
  }
}
