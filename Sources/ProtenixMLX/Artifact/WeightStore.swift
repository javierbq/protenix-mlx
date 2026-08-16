import Foundation
import MLX

/// Builds graph layers from a loaded artifact, hiding whether it is int8 or dense.
///
/// Every layer in the port asks for its weights through this type, so the int8/dense
/// branch is decided once, here, from the manifest -- rather than at each of the
/// several hundred call sites that would otherwise each have to get it right.
public struct WeightStore {
  public let artifact: ProtenixArtifact
  private let specs: [String: TensorSpec]

  public init(artifact: ProtenixArtifact) {
    self.artifact = artifact
    self.specs = Dictionary(
      artifact.manifest.tensors.map { ($0.name, $0) },
      uniquingKeysWith: { current, _ in current }
    )
  }

  public var configuration: ProtenixModelConfiguration { artifact.configuration }

  /// Whether the matrix owned by module `path` was packed as int8.
  ///
  /// Takes the MODULE path (`...linear_q`), not the tensor name (`...linear_q.weight`)
  /// -- the quantized siblings hang off the module, so passing the tensor name looks
  /// up `...linear_q.weight.scales` and always misses.
  ///
  /// Decided per matrix, not per pack: an int8 pack still stores anything that is not
  /// a rank-2 `.weight` densely, so the manifest's quantization block alone does not
  /// settle it.
  private func isQuantized(_ path: String) -> Bool {
    artifact.manifest.quantization != nil && specs["\(path).scales"] != nil
  }

  /// The matrix owned by module `path`, with its bias when it has one.
  ///
  /// Protenix mixes `LinearNoBias` and `nn.Linear` throughout -- the Pairformer's
  /// `linear_q` carries a bias while the projection beside it does not -- so the bias
  /// is detected from the artifact rather than declared by each of several hundred
  /// call sites that could get it wrong.
  public func linear(_ path: String) throws -> AffineLinear {
    let weightName = "\(path).weight"
    // `.bias` is the layer's own additive term; `.biases` is the quantizer's
    // zero-point. Distinct names, and only the former is looked up here.
    let bias = specs["\(path).bias"] != nil ? try artifact.array("\(path).bias") : nil
    guard isQuantized(path) else {
      let dense = try artifact.array(weightName)
      guard dense.ndim == 2 else {
        throw ProtenixError.malformedQuantizedMatrix(
          name: weightName, reason: "expected a rank-2 matrix, found rank \(dense.ndim)"
        )
      }
      // A packed matrix reaching the dense path would be multiplied as though its
      // uint32 words were weights: no error, no crash, just silently wrong numbers.
      // Refuse instead of trusting that the branch above is right.
      guard dense.dtype != .uint32 else {
        throw ProtenixError.malformedQuantizedMatrix(
          name: weightName,
          reason: "matrix is int8-packed but has no scales; the artifact is incomplete"
        )
      }
      return AffineLinear(denseWeight: dense, linearBias: bias)
    }
    guard let quantization = artifact.manifest.quantization else {
      throw ProtenixError.missingQuantizationSpec(weightName)
    }
    guard let spec = specs[weightName], let logical = spec.logicalShape,
      let physical = spec.physicalShape
    else {
      throw ProtenixError.malformedQuantizedMatrix(
        name: weightName, reason: "manifest declares no logical/physical shape"
      )
    }
    guard logical.count == 2, physical.count == 2 else {
      throw ProtenixError.malformedQuantizedMatrix(
        name: weightName, reason: "logical/physical shapes must be rank 2"
      )
    }
    return AffineLinear(
      weight: try artifact.array(weightName),
      linearBias: bias,
      logicalInputWidth: logical[1],
      physicalInputWidth: physical[1],
      storage: .affineInt8(
        scales: try artifact.array("\(path).scales"),
        quantizationBiases: try artifact.array("\(path).biases"),
        groupSize: quantization.groupSize,
        bits: quantization.bits
      )
    )
  }

  public func embedding(_ path: String) throws -> AffineEmbedding {
    let weightName = "\(path).weight"
    guard isQuantized(path) else {
      let dense = try artifact.array(weightName)
      guard dense.dtype != .uint32 else {
        throw ProtenixError.malformedQuantizedMatrix(
          name: weightName,
          reason: "matrix is int8-packed but has no scales; the artifact is incomplete"
        )
      }
      return AffineEmbedding(denseWeight: dense)
    }
    guard let quantization = artifact.manifest.quantization,
      let spec = specs[weightName], let logical = spec.logicalShape, logical.count == 2
    else {
      throw ProtenixError.malformedQuantizedMatrix(
        name: weightName, reason: "manifest declares no logical shape"
      )
    }
    return AffineEmbedding(
      weight: try artifact.array(weightName),
      logicalOutputWidth: logical[1],
      storage: .affineInt8(
        scales: try artifact.array("\(path).scales"),
        quantizationBiases: try artifact.array("\(path).biases"),
        groupSize: quantization.groupSize,
        bits: quantization.bits
      )
    )
  }

  /// A LayerNorm's gain and optional shift. Always dense -- they are rank-1.
  public func layerNorm(_ path: String) throws -> LayerNormWeights {
    LayerNormWeights(
      weight: try? artifact.array("\(path).weight"),
      bias: try? artifact.array("\(path).bias")
    )
  }

  /// Any tensor the graph needs verbatim, such as the confidence head's bin edges.
  public func raw(_ name: String) throws -> MLXArray {
    try artifact.array(name)
  }

  /// Names present in the artifact under `prefix`, sorted. Used by the CLI's
  /// inspection path and by tests that assert a stack's depth from its weights.
  public func names(withPrefix prefix: String) -> [String] {
    artifact.manifest.tensors.map(\.name).filter { $0.hasPrefix(prefix) }.sorted()
  }
}

/// A LayerNorm's learned parameters. Both are optional because upstream's
/// `LayerNorm` can be configured without an affine transform.
///
/// Not `Sendable`: `MLXArray` is a reference into MLX's own graph and is not safe to
/// move across isolation domains, so weights stay on the thread that loaded them.
public struct LayerNormWeights {
  public let weight: MLXArray?
  public let bias: MLXArray?

  public init(weight: MLXArray?, bias: MLXArray?) {
    self.weight = weight
    self.bias = bias
  }
}
