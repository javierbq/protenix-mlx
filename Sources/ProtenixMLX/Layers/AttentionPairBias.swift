import Foundation
import MLX

/// Algorithm 24: multi-head attention over the single representation, biased by pairs.
///
/// Used in two configurations that are different graphs, not different arguments:
///
/// * `hasS == true` (Pairformer, diffusion transformer) conditions the input norm on the
///   single embedding via `AdaptiveLayerNorm` and gates the output by
///   `sigmoid(linear_a_last(s))`.
/// * `hasS == false` (confidence head, template stack) uses a plain LayerNorm and has no
///   output gate at all -- `linear_a_last` does not exist in the checkpoint.
///
/// Which one applies is decided by the weights present, not by a caller-supplied flag,
/// so a variant that differs here cannot be mis-driven.
public struct AttentionPairBias {
  public let headCount: Int
  public let hasS: Bool

  let adaptiveNorm: AdaptiveLayerNorm?
  let plainNorm: LayerNormWeights?
  let normZ: LayerNormWeights
  let linearZ: AffineLinear
  let linearQ: AffineLinear
  let linearK: AffineLinear
  let linearV: AffineLinear
  let linearG: AffineLinear
  let linearO: AffineLinear
  let linearALast: AffineLinear?
  /// Per-head width. Upstream passes `c_a / n_heads` as `c_hidden`, and the 1/sqrt
  /// scaling uses that per-head width -- not the full embedding width.
  let headWidth: Int

  public init(store: WeightStore, path: String, headCount: Int) throws {
    self.headCount = headCount
    let adaptivePath = "\(path).layernorm_a"
    self.hasS = !store.names(withPrefix: "\(adaptivePath).linear_s.weight").isEmpty
    if hasS {
      adaptiveNorm = try AdaptiveLayerNorm(store: store, path: adaptivePath)
      plainNorm = nil
      linearALast = try store.linear("\(path).linear_a_last")
    } else {
      adaptiveNorm = nil
      plainNorm = try store.layerNorm(adaptivePath)
      linearALast = nil
    }
    normZ = try store.layerNorm("\(path).layernorm_z")
    linearZ = try store.linear("\(path).linear_nobias_z")
    linearQ = try store.linear("\(path).attention.linear_q")
    linearK = try store.linear("\(path).attention.linear_k")
    linearV = try store.linear("\(path).attention.linear_v")
    linearG = try store.linear("\(path).attention.linear_g")
    linearO = try store.linear("\(path).attention.linear_o")
    headWidth = linearQ.weight.shape[0] / headCount
  }

  /// - Parameters:
  ///   - a: `[..., N, c_a]` single representation.
  ///   - s: `[..., N, c_s]` conditioning; required when `hasS`, ignored otherwise.
  ///   - z: `[..., N, N, c_z]` pair representation supplying the attention bias.
  public func callAsFunction(_ a: MLXArray, s: MLXArray?, z: MLXArray) -> MLXArray {
    var value: MLXArray
    if let adaptiveNorm, let s {
      value = adaptiveNorm(a, s)
    } else if let plainNorm {
      value = TensorOps.layerNorm(a, plainNorm)
    } else {
      // hasS with no s supplied: normalizing without the conditioning would silently
      // produce a different model, so fall back to the unconditioned norm only when
      // there is genuinely no adaptive norm to apply.
      value = TensorOps.layerNorm(a, weight: nil, bias: nil)
    }

    // Bias is [..., N, N, heads] projected from z, then moved to [..., heads, N, N].
    let bias = TensorOps.permuteFinalDims(
      linearZ(TensorOps.layerNorm(z, normZ)), [2, 0, 1])

    let query = TensorOps.splitHeads(linearQ(value), heads: headCount)
      / MLXArray(sqrt(Float(headWidth)))
    let key = TensorOps.splitHeads(linearK(value), heads: headCount)
    let element = TensorOps.splitHeads(linearV(value), heads: headCount)

    let attended = TensorOps.attention(
      query: query, key: key, value: element, bias: bias)
    // Gate is computed from the NORMALIZED input, per-head, before the heads are
    // merged -- gating the merged vector would apply the wrong slice to each head.
    let gate = MLX.sigmoid(
      TensorOps.splitHeads(linearG(value), heads: headCount))
    var output = linearO(TensorOps.mergeHeads(attended.asType(a.dtype) * gate))

    if let linearALast, let s {
      output = MLX.sigmoid(linearALast(s)) * output
    }
    return output
  }
}
