import MLX

/// Algorithm 10: outer product mean, folding the MSA back into the pair representation.
public struct OuterProductMean {
  let layerNorm: LayerNormWeights
  let linear1: AffineLinear
  let linear2: AffineLinear
  let linearOut: AffineLinear
  /// Upstream's numerical guard on the row-count normalizer. Part of the arithmetic,
  /// not a tolerance: it shifts every output slightly and must match.
  let eps: Float = 1e-3

  public init(store: WeightStore, path: String) throws {
    layerNorm = try store.layerNorm("\(path).layer_norm")
    linear1 = try store.linear("\(path).linear_1")
    linear2 = try store.linear("\(path).linear_2")
    linearOut = try store.linear("\(path).linear_out")
  }

  /// - Parameter m: `[..., S, N, c_m]` MSA representation.
  /// - Returns: `[..., N, N, c_z]` pair update.
  public func callAsFunction(_ m: MLXArray) -> MLXArray {
    let normalized = TensorOps.layerNorm(m, layerNorm)
    // [..., S, N, C] -> [..., N, S, C]: the sequence axis is the one contracted over,
    // so it has to be innermost-but-one before the outer product.
    let a = linear1(normalized).swappedAxes(-2, -3).asType(.float32)
    let b = linear2(normalized).swappedAxes(-2, -3).asType(.float32)

    // einsum("...bac,...dae->...bdce"): for each pair of token positions (b, d), the
    // outer product of their per-sequence features, summed over sequences a.
    // Written as a matmul over the flattened channel pair so it stays one GEMM:
    //   [..., N, 1, S, C, 1] * [..., 1, N, S, 1, E] summed over S.
    let sequenceCount = a.shape[a.ndim - 2]
    let widthC = a.shape[a.ndim - 1]
    let widthE = b.shape[b.ndim - 1]

    // [..., N, S*C] with the outer-product axes laid out so one matmul contracts S.
    let leading = Array(a.shape.dropLast(2))
    let tokens = leading[leading.count - 1]
    let batch = Array(leading.dropLast())

    // a: [..., N, S, C] -> [..., N, C, S]; b: [..., N, S, E] -> [..., N, S, E]
    // matmul over S gives [..., N, C, ... ] per token pair, so instead reshape to
    // contract explicitly: fold (N, C) against (N, E) over S.
    let aFlat = a.swappedAxes(-2, -1)
      .reshaped(batch + [tokens * widthC, sequenceCount])
    let bFlat = b.reshaped(batch + [tokens * sequenceCount, widthE])
      .reshaped(batch + [tokens, sequenceCount, widthE])
      .swappedAxes(-3, -2)
      .reshaped(batch + [sequenceCount, tokens * widthE])
    // [..., N*C, S] @ [..., S, N*E] -> [..., N*C, N*E]
    var outer = MLX.matmul(aFlat, bFlat)
    // -> [..., N, C, N, E] -> [..., N, N, C, E] -> [..., N, N, C*E]
    outer = outer.reshaped(batch + [tokens, widthC, tokens, widthE])
      .transposed(axes: axesForOuter(rank: outer.ndim + 2))
      .reshaped(batch + [tokens, tokens, widthC * widthE])

    // Normalizer is the number of sequences contributing to each pair. With no mask
    // every entry is the full depth, but the eps is still added.
    let norm = MLXArray(Float(sequenceCount)) + MLXArray(eps)
    return (linearOut(outer.asType(m.dtype)).asType(.float32) / norm).asType(m.dtype)
  }

  /// Axis order taking `[..., N, C, N, E]` to `[..., N, N, C, E]`.
  private func axesForOuter(rank: Int) -> [Int] {
    let base = rank - 4
    return Array(0..<base) + [base, base + 2, base + 1, base + 3]
  }
}
