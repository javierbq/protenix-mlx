import Foundation
import MLX

/// Shared tensor helpers, matching upstream's PyTorch semantics exactly.
public enum TensorOps {

  /// Layer normalization over the last axis.
  ///
  /// `weight`/`bias` are both optional because upstream's `LayerNorm` factory takes
  /// `create_scale` and `create_offset` independently -- `AdaptiveLayerNorm` builds one
  /// norm with neither and one with scale but no offset, and a port that assumed both
  /// exist would look for weights the checkpoint does not contain.
  ///
  /// eps is 1e-5, upstream's default, and is applied inside the square root.
  public static func layerNorm(
    _ x: MLXArray,
    weight: MLXArray?,
    bias: MLXArray?,
    eps: Float = 1e-5
  ) -> MLXArray {
    // Computed in float32 regardless of the pack's width: the variance of a
    // 384-wide float16 row loses enough precision to shift the normalized output
    // visibly, and upstream computes this in float32 too.
    let value = x.asType(.float32)
    let mean = value.mean(axis: -1, keepDims: true)
    let centered = value - mean
    let variance = (centered * centered).mean(axis: -1, keepDims: true)
    var normalized = centered * MLX.rsqrt(variance + eps)
    if let weight { normalized = normalized * weight.asType(.float32) }
    if let bias { normalized = normalized + bias.asType(.float32) }
    return normalized.asType(x.dtype)
  }

  /// `LayerNormWeights` overload, for call sites that fetched from a `WeightStore`.
  public static func layerNorm(
    _ x: MLXArray, _ weights: LayerNormWeights, eps: Float = 1e-5
  ) -> MLXArray {
    layerNorm(x, weight: weights.weight, bias: weights.bias, eps: eps)
  }

  /// Split the trailing axis into `heads` and move it ahead of the sequence axis.
  ///
  /// `[..., N, H * C] -> [..., H, N, C]`, upstream's `view` + `transpose(-2, -3)`.
  public static func splitHeads(_ x: MLXArray, heads: Int) -> MLXArray {
    let shape = x.shape
    let width = shape[shape.count - 1] / heads
    let reshaped = x.reshaped(Array(shape.dropLast()) + [heads, width])
    return reshaped.swappedAxes(-2, -3)
  }

  /// Inverse of `splitHeads`: `[..., H, N, C] -> [..., N, H * C]`.
  public static func mergeHeads(_ x: MLXArray) -> MLXArray {
    let merged = x.swappedAxes(-2, -3)
    let shape = merged.shape
    let flattened = shape[shape.count - 2] * shape[shape.count - 1]
    return merged.reshaped(Array(shape.dropLast(2)) + [flattened])
  }

  /// Scaled dot-product attention with an additive bias, over the last two axes.
  ///
  /// `q` arrives pre-scaled -- upstream divides by `sqrt(c_hidden)` in `_prep_qkv` and
  /// then calls the kernel with `scale: 1.0`, so scaling again here would halve the
  /// logits' magnitude and flatten every attention distribution.
  public static func attention(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    bias: MLXArray?
  ) -> MLXArray {
    // Logits and softmax in float32: upstream explicitly upcasts q, k and the bias
    // before the product, and a float16 softmax over a long sequence saturates.
    let q = query.asType(.float32)
    let k = key.asType(.float32)
    var logits = MLX.matmul(q, k.swappedAxes(-1, -2))
    if let bias { logits = logits + bias.asType(.float32) }
    let weights = MLX.softmax(logits, axis: -1)
    return MLX.matmul(weights.asType(value.dtype), value)
  }

  /// Move the final three axes into the given order, upstream's `permute_final_dims`.
  ///
  /// `order` is expressed over the last three axes only: `(2, 0, 1)` sends the last
  /// axis to the front of the three.
  public static func permuteFinalDims(_ x: MLXArray, _ order: [Int]) -> MLXArray {
    let rank = x.ndim
    let leading = Array(0..<(rank - order.count))
    let base = rank - order.count
    return x.transposed(axes: leading + order.map { base + $0 })
  }

  /// SiLU / swish, upstream's `F.silu`.
  public static func silu(_ x: MLXArray) -> MLXArray {
    x * MLX.sigmoid(x)
  }
}
