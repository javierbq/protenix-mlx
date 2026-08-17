import Foundation
import MLX

/// Algorithm 22: random Fourier features of the log noise level.
///
/// `w` and `b` are drawn once from a seeded generator at construction and frozen
/// (`requires_grad=False`), but they ARE saved in the checkpoint — so they are loaded
/// here rather than regenerated. Regenerating them would require reproducing PyTorch's
/// Mersenne-Twister stream exactly, and any divergence would silently shift every
/// noise embedding.
public struct FourierEmbedding {
  let w: MLXArray
  let b: MLXArray

  public init(store: WeightStore, path: String) throws {
    w = try store.raw("\(path).w")
    b = try store.raw("\(path).b")
  }

  /// - Parameter noiseLevel: `[..., S]` already scaled to `log(sigma / sigma_data) / 4`.
  /// - Returns: `[..., S, c]`.
  public func callAsFunction(_ noiseLevel: MLXArray) -> MLXArray {
    MLX.cos(2 * Float.pi * (noiseLevel.expandedDimensions(axis: -1) * w + b))
  }
}

/// Algorithm 21: builds the noise-conditioned single and pair embeddings the diffusion
/// transformer runs against.
///
/// The pair half depends only on the trunk, not on the noise level, so upstream caches
/// it across diffusion steps (`prepare_cache`). `preparePair` here is that same
/// computation, exposed so a sampler can hoist it out of the step loop — at 200 steps
/// that is 200× the triangle-free but still substantial pair work.
public struct DiffusionConditioning {
  public let sigmaData: Float

  /// Owned here, as upstream does: the diffusion module has its own relative-position
  /// encoder, distinct from the trunk's, with its own weights.
  public let relativePositionEncoding: RelativePositionEncoding
  let normZ: LayerNormWeights
  let linearZ: AffineLinear
  let transitionZ1: Transition
  let transitionZ2: Transition

  let normS: LayerNormWeights
  let linearS: AffineLinear
  let fourier: FourierEmbedding
  let normN: LayerNormWeights
  let linearN: AffineLinear
  let transitionS1: Transition
  let transitionS2: Transition

  public init(
    store: WeightStore, path: String, sigmaData: Float, rMax: Int = 32, sMax: Int = 2
  ) throws {
    self.sigmaData = sigmaData
    relativePositionEncoding = try RelativePositionEncoding(
      store: store, path: "\(path).relpe", rMax: rMax, sMax: sMax)
    normZ = try store.layerNorm("\(path).layernorm_z")
    linearZ = try store.linear("\(path).linear_no_bias_z")
    transitionZ1 = try Transition(store: store, path: "\(path).transition_z1")
    transitionZ2 = try Transition(store: store, path: "\(path).transition_z2")

    normS = try store.layerNorm("\(path).layernorm_s")
    linearS = try store.linear("\(path).linear_no_bias_s")
    fourier = try FourierEmbedding(store: store, path: "\(path).fourier_embedding")
    normN = try store.layerNorm("\(path).layernorm_n")
    linearN = try store.linear("\(path).linear_no_bias_n")
    transitionS1 = try Transition(store: store, path: "\(path).transition_s1")
    transitionS2 = try Transition(store: store, path: "\(path).transition_s2")
  }

  /// Pair conditioning, independent of the noise level and so cacheable.
  ///
  /// - Parameters:
  ///   - zTrunk: `[..., N, N, c_z]` from the Pairformer.
  ///   - relativeFeatures: `[..., N, N, 4·r_max + 2·s_max + 7]` one-hot block, built by
  ///     `RelativePositionEncoding.relativeFeatures` from raw token indices.
  public func preparePair(
    zTrunk: MLXArray, relativeFeatures: MLXArray
  ) -> MLXArray {
    let concatenated = MLX.concatenated(
      [zTrunk, relativePositionEncoding(relativeFeatures)], axis: -1)
    var pair = linearZ(TensorOps.layerNorm(concatenated, normZ))
    pair = pair + transitionZ1(pair)
    pair = pair + transitionZ2(pair)
    return pair
  }

  /// Single conditioning at a given noise level.
  ///
  /// - Parameters:
  ///   - noiseLevel: `[..., S]` raw sigma, NOT pre-scaled — the log/sigma_data/4
  ///     transform is applied here, matching upstream's call site.
  ///   - sInputs: `[..., N, c_s_inputs]` from the input embedder.
  ///   - sTrunk: `[..., N, c_s]` from the Pairformer.
  /// - Returns: `[..., S, N, c_s]`.
  public func prepareSingle(
    noiseLevel: MLXArray, sInputs: MLXArray, sTrunk: MLXArray
  ) -> MLXArray {
    let concatenated = MLX.concatenated([sTrunk, sInputs], axis: -1)
    let projected = linearS(TensorOps.layerNorm(concatenated, normS))

    let scaled = MLX.log(noiseLevel / sigmaData) / 4
    let noise = fourier(scaled).asType(projected.dtype)
    // [..., N, c_s] broadcast over samples, plus [..., S, c_s] broadcast over tokens.
    var single =
      projected.expandedDimensions(axis: -3)
      + linearN(TensorOps.layerNorm(noise, normN)).expandedDimensions(axis: -2)
    single = single + transitionS1(single)
    single = single + transitionS2(single)
    return single
  }
}
