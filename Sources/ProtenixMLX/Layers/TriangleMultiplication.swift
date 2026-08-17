import MLX

/// Algorithms 11/12: triangular multiplicative update, outgoing or incoming.
///
/// The two directions differ only in how the two projections are permuted before the
/// matmul, which is why one type serves both.
public struct TriangleMultiplication {
  public let outgoing: Bool

  let normIn: LayerNormWeights
  let normOut: LayerNormWeights
  let linearAP: AffineLinear
  let linearAG: AffineLinear
  let linearBP: AffineLinear
  let linearBG: AffineLinear
  let linearZ: AffineLinear
  let linearG: AffineLinear

  public init(store: WeightStore, path: String, outgoing: Bool) throws {
    self.outgoing = outgoing
    normIn = try store.layerNorm("\(path).layer_norm_in")
    normOut = try store.layerNorm("\(path).layer_norm_out")
    linearAP = try store.linear("\(path).linear_a_p")
    linearAG = try store.linear("\(path).linear_a_g")
    linearBP = try store.linear("\(path).linear_b_p")
    linearBG = try store.linear("\(path).linear_b_g")
    linearZ = try store.linear("\(path).linear_z")
    linearG = try store.linear("\(path).linear_g")
  }

  /// - Parameter z: `[..., N, N, c_z]`.
  public func callAsFunction(_ z: MLXArray) -> MLXArray {
    let normalized = TensorOps.layerNorm(z, normIn)
    let a = MLX.sigmoid(linearAG(normalized)) * linearAP(normalized)
    let b = MLX.sigmoid(linearBG(normalized)) * linearBP(normalized)

    var product = combine(a, b)
    product = TensorOps.layerNorm(product, normOut)
    product = linearZ(product)
    return product * MLX.sigmoid(linearG(normalized))
  }

  /// Contract `a` and `b` over the shared index, in the direction this update encodes.
  ///
  /// Outgoing sums over the *outgoing* edge k: `p[i][j] = sum_k a[i][k] * b[j][k]`.
  /// Incoming sums over the incoming one: `p[i][j] = sum_k a[k][i] * b[k][j]`. Upstream
  /// expresses this by permuting both operands to `[c, ., .]` and taking a plain matmul;
  /// the same permutations are used here so the two implementations can be compared
  /// term by term.
  private func combine(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
    let left: MLXArray
    let right: MLXArray
    if outgoing {
      left = TensorOps.permuteFinalDims(a, [2, 0, 1])
      right = TensorOps.permuteFinalDims(b, [2, 1, 0])
    } else {
      left = TensorOps.permuteFinalDims(a, [2, 1, 0])
      right = TensorOps.permuteFinalDims(b, [2, 0, 1])
    }
    // In float32: this is a sum over the full token axis of products of unbounded
    // activations, and it is the step upstream guards against overflow in reduced
    // precision.
    let product = MLX.matmul(left.asType(.float32), right.asType(.float32))
    return TensorOps.permuteFinalDims(product, [1, 2, 0]).asType(a.dtype)
  }
}

/// Algorithms 13/14: triangular self-attention, starting or ending node.
public struct TriangleAttention {
  public let starting: Bool
  public let headCount: Int

  let layerNorm: LayerNormWeights
  let linearBias: AffineLinear
  let linearQ: AffineLinear
  let linearK: AffineLinear
  let linearV: AffineLinear
  let linearG: AffineLinear
  let linearO: AffineLinear
  let headWidth: Int

  public init(store: WeightStore, path: String, starting: Bool, headCount: Int) throws {
    self.starting = starting
    self.headCount = headCount
    layerNorm = try store.layerNorm("\(path).layer_norm")
    linearBias = try store.linear("\(path).linear")
    linearQ = try store.linear("\(path).mha.linear_q")
    linearK = try store.linear("\(path).mha.linear_k")
    linearV = try store.linear("\(path).mha.linear_v")
    linearG = try store.linear("\(path).mha.linear_g")
    linearO = try store.linear("\(path).mha.linear_o")
    headWidth = linearQ.weight.shape[0] / headCount
  }

  /// - Parameter x: `[..., N, N, c_z]`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Ending-node attention is starting-node attention on the transposed pair matrix.
    // Upstream transposes on the way in and back on the way out; doing the same keeps
    // one attention body for both.
    var value = starting ? x : x.swappedAxes(-2, -3)
    value = TensorOps.layerNorm(value, layerNorm)

    // [..., N, N, H] -> [..., H, N, N], then broadcast over the leading token axis
    // that attention iterates: the bias is shared across rows, which is what makes
    // this *triangular* rather than plain axial attention.
    let bias = TensorOps.permuteFinalDims(linearBias(value), [2, 0, 1])
      .expandedDimensions(axis: -4)

    let query = TensorOps.splitHeads(linearQ(value), heads: headCount)
      / MLXArray(Float(headWidth).squareRoot())
    let key = TensorOps.splitHeads(linearK(value), heads: headCount)
    let element = TensorOps.splitHeads(linearV(value), heads: headCount)

    let attended = TensorOps.attention(
      query: query, key: key, value: element, bias: bias)
    let gate = MLX.sigmoid(TensorOps.splitHeads(linearG(value), heads: headCount))
    var output = linearO(TensorOps.mergeHeads(attended.asType(x.dtype) * gate))

    if !starting { output = output.swappedAxes(-2, -3) }
    return output
  }
}
