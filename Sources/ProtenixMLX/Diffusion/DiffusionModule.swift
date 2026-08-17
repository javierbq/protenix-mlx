import Foundation
import MLX

/// Algorithm 20: one EDM denoising step.
///
/// Assembles the whole diffusion network — conditioning, atom encoder, token-level
/// diffusion transformer, atom decoder — and wraps it in the EDM input/output scaling
/// that turns a raw network `F` into a denoiser `D`. A single call maps a noised
/// coordinate set at a given noise level to its denoised estimate; the sampler loops it.
public struct DiffusionModule {
  public let sigmaData: Float

  public let conditioning: DiffusionConditioning
  let encoder: AtomAttentionEncoder
  let decoder: AtomAttentionDecoder
  let transformer: DiffusionTransformer

  let normS: LayerNormWeights
  let linearS: AffineLinear
  let normA: LayerNormWeights

  public init(
    store: WeightStore, path: String, sigmaData: Float,
    atomEncoderBlocks: Int, atomEncoderHeads: Int,
    transformerBlocks: Int, transformerHeads: Int,
    atomDecoderBlocks: Int, atomDecoderHeads: Int,
    rMax: Int = 32, sMax: Int = 2
  ) throws {
    self.sigmaData = sigmaData
    conditioning = try DiffusionConditioning(
      store: store, path: "\(path).diffusion_conditioning", sigmaData: sigmaData,
      rMax: rMax, sMax: sMax)
    encoder = try AtomAttentionEncoder(
      store: store, path: "\(path).atom_attention_encoder",
      blockCount: atomEncoderBlocks, headCount: atomEncoderHeads)
    decoder = try AtomAttentionDecoder(
      store: store, path: "\(path).atom_attention_decoder",
      blockCount: atomDecoderBlocks, headCount: atomDecoderHeads)
    transformer = try DiffusionTransformer(
      store: store, path: "\(path).diffusion_transformer",
      blockCount: transformerBlocks, headCount: transformerHeads)
    normS = try store.layerNorm("\(path).layernorm_s")
    linearS = try store.linear("\(path).linear_no_bias_s")
    normA = try store.layerNorm("\(path).layernorm_a")
  }

  /// The conditioning that is constant across the sampler's steps: the noise-level-free
  /// pair embedding. Hoisting it out of the loop is upstream's `prepare_cache`, and at
  /// 200 steps it is a large saving.
  public func prepareConditioning(
    zTrunk: MLXArray, relativeFeatures: MLXArray
  ) -> MLXArray {
    conditioning.preparePair(zTrunk: zTrunk, relativeFeatures: relativeFeatures)
  }

  /// The trunk and feature inputs a denoising step needs, none of which change between
  /// steps. Bundled so the sampler threads one value through its loop.
  public struct Context {
    public let features: AtomAttentionEncoder.Features
    public let relativeFeatures: MLXArray
    public let sInputs: MLXArray
    public let sTrunk: MLXArray
    public let zTrunk: MLXArray
    public let tokenCount: Int
    /// Cached pair conditioning from `prepareConditioning`, if the sampler hoisted it.
    public let pairConditioning: MLXArray?

    public init(
      features: AtomAttentionEncoder.Features, relativeFeatures: MLXArray,
      sInputs: MLXArray, sTrunk: MLXArray, zTrunk: MLXArray, tokenCount: Int,
      pairConditioning: MLXArray? = nil
    ) {
      self.features = features
      self.relativeFeatures = relativeFeatures
      self.sInputs = sInputs
      self.sTrunk = sTrunk
      self.zTrunk = zTrunk
      self.tokenCount = tokenCount
      self.pairConditioning = pairConditioning
    }
  }

  /// One denoising step: `(x_noisy, sigma) -> x_denoised`.
  ///
  /// - Parameters:
  ///   - xNoisy: `[..., S, N_atom, 3]`.
  ///   - noiseLevel: `[..., S]` the current sigma (also the time step, since sigma = t).
  public func callAsFunction(
    xNoisy: MLXArray, noiseLevel: MLXArray, context: Context
  ) -> MLXArray {
    let sigmaColumn = noiseLevel.expandedDimensions(axis: -1).expandedDimensions(axis: -1)
    let cIn = 1 / MLX.sqrt(sigmaData * sigmaData + sigmaColumn * sigmaColumn)
    let rNoisy = xNoisy * cIn

    let pairZ =
      context.pairConditioning
      ?? conditioning.preparePair(
        zTrunk: context.zTrunk, relativeFeatures: context.relativeFeatures)
    let sSingle = conditioning.prepareSingle(
      noiseLevel: noiseLevel, sInputs: context.sInputs, sTrunk: context.sTrunk)

    // The atom encoder and token transformer carry a sample axis; the trunk embeddings
    // are broadcast into it.
    let sTrunkExpanded = context.sTrunk.expandedDimensions(axis: -3)
    let zPairExpanded = pairZ.expandedDimensions(axis: -4)

    let encoded = encoder(
      context.features, r: rNoisy, s: sTrunkExpanded, z: zPairExpanded,
      tokenCount: context.tokenCount)

    // Everything from here is float32, matching upstream's explicit upcast.
    var aToken =
      encoded.a.asType(.float32)
      + linearS(TensorOps.layerNorm(sSingle, normS)).asType(.float32)
    aToken = transformer(
      aToken, sSingle.asType(.float32), zPairExpanded.asType(.float32))
    aToken = TensorOps.layerNorm(aToken, normA)

    let rUpdate = decoder(
      aToken, atomToToken: context.features.atomToToken, qSkip: encoded.qSkip,
      cSkip: encoded.cSkip, pSkip: encoded.pSkip)

    // EDM output scaling: D = c_skip * x_noisy + c_out * r_update.
    let sRatio = sigmaColumn / sigmaData
    let cSkip = 1 / (1 + sRatio * sRatio)
    let cOut = sigmaColumn / MLX.sqrt(1 + sRatio * sRatio)
    return cSkip * xNoisy + cOut * rUpdate.asType(.float32)
  }
}
