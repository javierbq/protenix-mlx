import MLX

/// Algorithm 3: relative position encoding.
///
/// Splits into two halves because upstream does, and because they have very different
/// costs: `relativeFeatures` builds a `[N, N, 4·r_max + 2·s_max + 7]` one-hot block from
/// raw token indices and depends only on the input, while the projection is a single
/// matmul. A sampler can therefore build the features once and project per use.
public struct RelativePositionEncoding {
  public let rMax: Int
  public let sMax: Int
  let linear: AffineLinear

  public init(store: WeightStore, path: String, rMax: Int, sMax: Int) throws {
    self.rMax = rMax
    self.sMax = sMax
    linear = try store.linear("\(path).linear_no_bias")
  }

  /// Width of the one-hot feature block this encoder consumes.
  public var featureWidth: Int { 4 * rMax + 2 * sMax + 7 }

  /// - Parameter features: `[..., N, N, featureWidth]` from `relativeFeatures`.
  public func callAsFunction(_ features: MLXArray) -> MLXArray {
    linear(features)
  }

  /// Build the one-hot relative-position block from raw token indices.
  ///
  /// Every clip saturates to a dedicated "out of range" bucket rather than to the
  /// nearest in-range one: tokens in different chains get bucket `2·r_max + 1`, not
  /// bucket `2·r_max`, so "far apart" and "unrelated" stay distinguishable.
  ///
  /// - Parameters are each `[..., N]` integer arrays.
  public static func relativeFeatures(
    asymID: MLXArray,
    residueIndex: MLXArray,
    entityID: MLXArray,
    tokenIndex: MLXArray,
    symID: MLXArray,
    rMax: Int,
    sMax: Int
  ) -> MLXArray {
    let sameChain = equalOuter(asymID)
    let sameResidue = equalOuter(residueIndex)
    let sameEntity = equalOuter(entityID)

    let residueOffset =
      MLX.clip(
        differenceOuter(residueIndex) + rMax, min: MLXArray(0), max: MLXArray(2 * rMax))
      * sameChain + (1 - sameChain) * (2 * rMax + 1)
    let relativePosition = oneHot(residueOffset, classes: 2 * (rMax + 1))

    let sameChainAndResidue = sameChain * sameResidue
    let tokenOffset =
      MLX.clip(
        differenceOuter(tokenIndex) + rMax, min: MLXArray(0), max: MLXArray(2 * rMax))
      * sameChainAndResidue + (1 - sameChainAndResidue) * (2 * rMax + 1)
    let relativeToken = oneHot(tokenOffset, classes: 2 * (rMax + 1))

    let chainOffset =
      MLX.clip(
        differenceOuter(symID) + sMax, min: MLXArray(0), max: MLXArray(2 * sMax))
      * sameEntity + (1 - sameEntity) * (2 * sMax + 1)
    let relativeChain = oneHot(chainOffset, classes: 2 * (sMax + 1))

    return MLX.concatenated(
      [
        relativePosition, relativeToken,
        sameEntity.expandedDimensions(axis: -1).asType(.float32), relativeChain,
      ],
      axis: -1
    )
  }

  private static func equalOuter(_ x: MLXArray) -> MLXArray {
    (x.expandedDimensions(axis: -1) .== x.expandedDimensions(axis: -2))
      .asType(.int32)
  }

  private static func differenceOuter(_ x: MLXArray) -> MLXArray {
    x.expandedDimensions(axis: -1) - x.expandedDimensions(axis: -2)
  }

  private static func oneHot(_ indices: MLXArray, classes: Int) -> MLXArray {
    let identity = MLX.eye(classes, dtype: .float32)
    return identity[indices.asType(.int32)]
  }
}
