import MLX

/// Algorithm 11: a gated SwiGLU feed-forward block.
///
/// Upstream's inference path chunks this over the token axis to bound peak memory, but
/// the chunking is mathematically transparent -- each row is independent -- so it is not
/// reproduced here. MLX's lazy evaluation gives the same memory behaviour without it.
public struct Transition {
  let layerNorm: LayerNormWeights
  let linearA: AffineLinear
  let linearB: AffineLinear
  let linearOut: AffineLinear

  public init(store: WeightStore, path: String) throws {
    layerNorm = try store.layerNorm("\(path).layernorm1")
    linearA = try store.linear("\(path).linear_no_bias_a")
    linearB = try store.linear("\(path).linear_no_bias_b")
    linearOut = try store.linear("\(path).linear_no_bias")
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let normalized = TensorOps.layerNorm(x, layerNorm)
    // silu(a) * b, not silu(a * b): the two branches are separate projections and the
    // nonlinearity applies to only one of them.
    return linearOut(TensorOps.silu(linearA(normalized)) * linearB(normalized))
  }
}

/// Algorithm 26: adaptive layer norm, conditioning `a` on the single embedding `s`.
///
/// The norm over `a` has neither scale nor offset, and the norm over `s` has scale but
/// no offset -- the conditioning supplies what those would have provided. Both facts are
/// visible in the checkpoint: there is no `layernorm_a.weight`, and no `layernorm_s.bias`.
public struct AdaptiveLayerNorm {
  let normS: LayerNormWeights
  let linearS: AffineLinear
  let linearNoBiasS: AffineLinear

  public init(store: WeightStore, path: String) throws {
    normS = try store.layerNorm("\(path).layernorm_s")
    linearS = try store.linear("\(path).linear_s")
    linearNoBiasS = try store.linear("\(path).linear_nobias_s")
  }

  public func callAsFunction(_ a: MLXArray, _ s: MLXArray) -> MLXArray {
    let normalizedA = TensorOps.layerNorm(a, weight: nil, bias: nil)
    let normalizedS = TensorOps.layerNorm(s, normS)
    return MLX.sigmoid(linearS(normalizedS)) * normalizedA + linearNoBiasS(normalizedS)
  }
}
