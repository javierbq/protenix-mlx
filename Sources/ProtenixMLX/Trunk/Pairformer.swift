import MLX

/// Algorithm 17, lines 2-8: one Pairformer block.
///
/// Two things here are easy to get wrong by reading the algorithm rather than the code:
///
/// * **`triAttentionEnd` is a *starting*-node module.** Upstream constructs both
///   `tri_att_start` and `tri_att_end` as plain `TriangleAttention()`, whose `starting`
///   defaults to `true`; the "ending" behaviour comes from the block transposing `z`
///   around the call. Building it as an ending-node module here would transpose twice
///   and quietly compute the starting update again.
/// * **`attentionPairBias` is the no-`s` form** (`has_s=False`) even though the block
///   updates `s`. `s` is the *input* to the attention, not its conditioning, so there is
///   no AdaLN and no output gate. It also sets `create_offset_ln_z=True`, so its
///   `layernorm_z` carries a bias that the default configuration lacks -- handled
///   because `WeightStore.layerNorm` treats both terms as optional.
///
/// Dropout is absent throughout: it is row-wise and training-only, and at inference
/// upstream's `dropout_add_rowwise` reduces to a plain residual add.
public struct PairformerBlock {
  let triMulOut: TriangleMultiplication
  let triMulIn: TriangleMultiplication
  let triAttentionStart: TriangleAttention
  let triAttentionEnd: TriangleAttention
  let pairTransition: Transition
  /// Absent when the stack runs pair-only (`c_s == 0`), as the template stack does.
  let attentionPairBias: AttentionPairBias?
  let singleTransition: Transition?

  public init(store: WeightStore, path: String, headCount: Int, pairHeadCount: Int)
    throws
  {
    triMulOut = try TriangleMultiplication(
      store: store, path: "\(path).tri_mul_out", outgoing: true)
    triMulIn = try TriangleMultiplication(
      store: store, path: "\(path).tri_mul_in", outgoing: false)
    triAttentionStart = try TriangleAttention(
      store: store, path: "\(path).tri_att_start", starting: true,
      headCount: pairHeadCount)
    triAttentionEnd = try TriangleAttention(
      store: store, path: "\(path).tri_att_end", starting: true,
      headCount: pairHeadCount)
    pairTransition = try Transition(store: store, path: "\(path).pair_transition")

    let singlePath = "\(path).attention_pair_bias"
    if store.names(withPrefix: "\(singlePath).").isEmpty {
      attentionPairBias = nil
      singleTransition = nil
    } else {
      attentionPairBias = try AttentionPairBias(
        store: store, path: singlePath, headCount: headCount)
      singleTransition = try Transition(
        store: store, path: "\(path).single_transition")
    }
  }

  /// - Returns: the updated single and pair representations. `s` is returned unchanged
  ///   when this block has no single track.
  public func callAsFunction(_ s: MLXArray?, _ z: MLXArray) -> (MLXArray?, MLXArray) {
    var pair = z
    pair = pair + triMulOut(pair)
    pair = pair + triMulIn(pair)
    pair = pair + triAttentionStart(pair)

    // The ending-node update, expressed the way upstream does it: transpose, run the
    // starting-node module, transpose back.
    pair = pair.swappedAxes(-2, -3)
    pair = pair + triAttentionEnd(pair)
    pair = pair.swappedAxes(-2, -3)

    pair = pair + pairTransition(pair)

    guard let attentionPairBias, let singleTransition, let s else {
      return (s, pair)
    }
    var single = s + attentionPairBias(s, s: nil, z: pair)
    single = single + singleTransition(single)
    return (single, pair)
  }
}

/// Algorithm 17: the Pairformer trunk — 8 blocks in tiny, 48 in base and v2.
public struct PairformerStack {
  public let blocks: [PairformerBlock]

  public var blockCount: Int { blocks.count }

  public init(
    store: WeightStore, path: String, blockCount: Int, headCount: Int,
    pairHeadCount: Int
  ) throws {
    blocks = try (0..<blockCount).map { index in
      try PairformerBlock(
        store: store, path: "\(path).blocks.\(index)", headCount: headCount,
        pairHeadCount: pairHeadCount)
    }
  }

  public func callAsFunction(_ s: MLXArray?, _ z: MLXArray) -> (MLXArray?, MLXArray) {
    var single = s
    var pair = z
    for block in blocks {
      (single, pair) = block(single, pair)
      // Evaluated per block rather than once at the end: 48 blocks of lazily-built
      // graph would hold every intermediate pair tensor alive at once, which at
      // production token counts is the difference between fitting in memory and not.
      MLX.eval(pair)
      if let single { MLX.eval(single) }
    }
    return (single, pair)
  }
}
