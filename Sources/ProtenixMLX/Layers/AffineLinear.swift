import MLX

/// How a matrix parameter is stored in the artifact it was loaded from.
///
/// An int8 pack quantizes each matrix affinely and pads its input width up to the
/// group size; a dense pack stores the matrix at its own float width, unpadded. The
/// two are never mixed within one artifact -- see the exporter's `Precision`.
enum MatrixStorage {
  case affineInt8(scales: MLXArray, quantizationBiases: MLXArray?, groupSize: Int, bits: Int)
  case dense
}

/// An MLX matrix multiplication -- affine-int8 with logical-width padding, or dense.
///
/// Protenix is built almost entirely from `LinearNoBias`, so `linearBias` is nil for
/// the large majority of layers; it exists for the handful of `nn.Linear` sites.
public final class AffineLinear {
  public let weight: MLXArray
  public let linearBias: MLXArray?
  /// Input width the graph uses. For a quantized matrix this is narrower than the
  /// stored width, which was rounded up to a multiple of the group size.
  public let logicalInputWidth: Int
  public let physicalInputWidth: Int
  let storage: MatrixStorage

  init(
    weight: MLXArray,
    linearBias: MLXArray?,
    logicalInputWidth: Int,
    physicalInputWidth: Int,
    storage: MatrixStorage
  ) {
    self.weight = weight
    self.linearBias = linearBias
    self.logicalInputWidth = logicalInputWidth
    self.physicalInputWidth = physicalInputWidth
    self.storage = storage
  }

  /// A matrix stored at full float width.
  public convenience init(denseWeight: MLXArray, linearBias: MLXArray? = nil) {
    self.init(
      weight: denseWeight,
      linearBias: linearBias,
      logicalInputWidth: denseWeight.shape[1],
      physicalInputWidth: denseWeight.shape[1],
      storage: .dense
    )
  }

  public func callAsFunction(_ input: MLXArray) -> MLXArray {
    var value: MLXArray
    switch storage {
    case .affineInt8(let scales, let quantizationBiases, let groupSize, let bits):
      // The stored matrix is zero-padded on the contracted axis, so the activation
      // must be padded to match. Zeros contribute nothing to the sum, which is what
      // makes the padding invisible to the result.
      let padded =
        physicalInputWidth == logicalInputWidth
        ? input
        : MLX.padded(
          input,
          widths: paddingWidths(rank: input.ndim),
          value: MLXArray(0, dtype: input.dtype)
        )
      value = MLX.quantizedMatmul(
        padded,
        weight,
        scales: scales,
        biases: quantizationBiases,
        transpose: true,
        groupSize: groupSize,
        bits: bits,
        mode: .affine
      )
    case .dense:
      value = MLX.matmul(input, weight.transposed())
    }
    if let linearBias {
      value = value + linearBias
    }
    return value
  }

  private func paddingWidths(rank: Int) -> [IntOrPair] {
    var widths = Array(repeating: IntOrPair((0, 0)), count: rank)
    widths[rank - 1] = IntOrPair((0, physicalInputWidth - logicalInputWidth))
    return widths
  }
}

/// Row-wise lookup for an embedding matrix, affine-int8 or dense.
///
/// Padding runs along the *output* width here rather than the input width, because an
/// embedding is indexed by row and contracted along nothing -- so the padded columns
/// have to be sliced away instead of being cancelled by zeros.
public struct AffineEmbedding {
  public let weight: MLXArray
  public let logicalOutputWidth: Int
  let storage: MatrixStorage

  public init(denseWeight: MLXArray) {
    self.weight = denseWeight
    self.logicalOutputWidth = denseWeight.shape[1]
    self.storage = .dense
  }

  init(
    weight: MLXArray,
    logicalOutputWidth: Int,
    storage: MatrixStorage
  ) {
    self.weight = weight
    self.logicalOutputWidth = logicalOutputWidth
    self.storage = storage
  }

  public func callAsFunction(_ indices: MLXArray) -> MLXArray {
    let flattened = indices.flattened()
    let rows: MLXArray
    switch storage {
    case .affineInt8(let scales, let quantizationBiases, let groupSize, let bits):
      rows = MLX.dequantized(
        weight[flattened],
        scales: scales[flattened],
        biases: quantizationBiases.map { $0[flattened] },
        groupSize: groupSize,
        bits: bits,
        mode: .affine
      )
    case .dense:
      rows = weight[flattened]
    }
    let logicalRows = rows[.ellipsis, 0..<logicalOutputWidth]
    return logicalRows.reshaped(indices.shape + [logicalOutputWidth])
  }
}
