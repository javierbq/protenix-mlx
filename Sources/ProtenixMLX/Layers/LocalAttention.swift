import MLX

/// Blocked windowed attention over atoms, as used by `AtomTransformer` (Algorithm 7).
///
/// Atoms are far too numerous for dense attention, so each block of `nQueries` (32)
/// consecutive atoms attends to a `nKeys`-wide (128) window centred on it. The window
/// is built by padding the key axis and unfolding it, which means every query block sees
/// the same *shape* regardless of where it sits — including at the ends, where part of
/// the window falls outside the real atoms and must be masked to -inf rather than merely
/// zeroed. Zeroing would let padding compete in the softmax.
public enum LocalAttention {
  /// The window geometry for one atom count. All of it follows from `n`.
  public struct Trunking {
    public let atomCount: Int
    public let queryWindow: Int
    public let keyWindow: Int
    public let trunkCount: Int
    public let queryPadding: Int
    public let keyPadLeft: Int
    public let keyPadRight: Int

    public init(atomCount n: Int, queryWindow nQueries: Int, keyWindow nKeys: Int) {
      precondition(nKeys >= nQueries, "key window must be at least the query window")
      precondition(nQueries % 2 == 0 && nKeys % 2 == 0, "windows must be even")
      atomCount = n
      queryWindow = nQueries
      keyWindow = nKeys
      trunkCount = (n + nQueries - 1) / nQueries
      queryPadding = trunkCount * nQueries - n
      keyPadLeft = (nKeys - nQueries) / 2
      // Upstream writes this with halves: (n_trunks - 1/2)*n_queries + n_keys/2 - n + 1/2.
      // Doubled here so it stays exact in integer arithmetic.
      keyPadRight =
        (2 * trunkCount * nQueries - nQueries + nKeys - 2 * n + 1) / 2
    }

    /// Width of the padded key axis before unfolding.
    public var paddedKeyWidth: Int { atomCount + keyPadLeft + keyPadRight }
  }

  /// Split the query axis into `[..., trunks, nQueries, d]`, zero-padding the tail.
  public static func trunkQueries(_ x: MLXArray, _ trunking: Trunking) -> MLXArray {
    let padded = padAxis(
      x, axis: x.ndim - 2, before: 0, after: trunking.queryPadding, value: 0)
    let shape = padded.shape
    return padded.reshaped(
      Array(shape.dropLast(2)) + [trunking.trunkCount, trunking.queryWindow,
                                  shape[shape.count - 1]])
  }

  /// Slide a `nKeys`-wide window over the key axis: `[..., trunks, nKeys, d]`.
  public static func trunkKeys(_ x: MLXArray, _ trunking: Trunking) -> MLXArray {
    let padded = padAxis(
      x, axis: x.ndim - 2, before: trunking.keyPadLeft, after: trunking.keyPadRight,
      value: 0)
    // MLX has no `unfold`, so the windows are gathered explicitly. Each trunk t starts
    // at t * nQueries in the padded axis.
    let starts = MLXArray((0..<trunking.trunkCount).map { Int32($0 * trunking.queryWindow) })
    let offsets = MLXArray((0..<trunking.keyWindow).map { Int32($0) })
    let indices = starts.expandedDimensions(axis: 1) + offsets.expandedDimensions(axis: 0)
    return padded.take(indices.flattened(), axis: padded.ndim - 2)
      .reshaped(
        Array(padded.shape.dropLast(2))
          + [trunking.trunkCount, trunking.keyWindow, padded.shape[padded.ndim - 1]])
  }

  /// The -inf mask marking window positions that fall outside the real atoms.
  ///
  /// Shape `[trunks, nQueries, nKeys]`, broadcastable over batch and head axes.
  public static func windowBias(_ trunking: Trunking, dtype: DType = .float32)
    -> MLXArray
  {
    let n = trunking.atomCount
    // Absolute key index each (trunk, key-slot) refers to, in padded coordinates.
    let starts = MLXArray((0..<trunking.trunkCount).map { Int32($0 * trunking.queryWindow) })
    let offsets = MLXArray((0..<trunking.keyWindow).map { Int32($0) })
    let keyIndex =
      starts.expandedDimensions(axis: 1) + offsets.expandedDimensions(axis: 0)
      - MLXArray(Int32(trunking.keyPadLeft))  // [trunks, nKeys], real-atom coordinates
    let keyValid = MLX.logicalAnd(
      keyIndex .>= MLXArray(Int32(0)), keyIndex .< MLXArray(Int32(n)))

    // Query slots past the end of the real atoms are padding too.
    let queryStarts = MLXArray((0..<trunking.trunkCount).map { Int32($0 * trunking.queryWindow) })
    let queryOffsets = MLXArray((0..<trunking.queryWindow).map { Int32($0) })
    let queryIndex =
      queryStarts.expandedDimensions(axis: 1)
      + queryOffsets.expandedDimensions(axis: 0)  // [trunks, nQueries]
    let queryValid = queryIndex .< MLXArray(Int32(n))

    let valid = MLX.logicalAnd(
      keyValid.expandedDimensions(axis: 1),  // [trunks, 1, nKeys]
      queryValid.expandedDimensions(axis: 2)  // [trunks, nQueries, 1]
    )
    // -1e10, matching upstream's `inf` constant. Not Float.infinity: a fully-masked
    // row would then softmax to NaN instead of a uniform distribution.
    return MLX.where(valid, MLXArray(Float(0)), MLXArray(Float(-1e10))).asType(dtype)
  }

  /// Flatten `[..., trunks, nQueries, d]` back to `[..., n, d]`, dropping padding.
  public static func untrunk(_ x: MLXArray, _ trunking: Trunking) -> MLXArray {
    let shape = x.shape
    let flattened = x.reshaped(
      Array(shape.dropLast(3))
        + [trunking.trunkCount * trunking.queryWindow, shape[shape.count - 1]])
    guard trunking.queryPadding > 0 else { return flattened }
    return flattened[.ellipsis, 0..<trunking.atomCount, 0...]
  }

  /// Pad one axis with a constant.
  static func padAxis(
    _ x: MLXArray, axis: Int, before: Int, after: Int, value: Float
  ) -> MLXArray {
    guard before > 0 || after > 0 else { return x }
    var widths = Array(repeating: IntOrPair((0, 0)), count: x.ndim)
    widths[axis] = IntOrPair((before, after))
    return MLX.padded(x, widths: widths, value: MLXArray(value).asType(x.dtype))
  }
}
