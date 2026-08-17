import MLX

/// Algorithm 25: a transition block conditioned on the single embedding.
///
/// Differs from the plain `Transition` in three ways that all matter: the input norm is
/// adaptive rather than plain, there is no residual inside the block (the caller adds
/// it), and the output is gated by `sigmoid(linear_s(s))` — an adaLN-Zero gate whose
/// bias is initialized to -2.0 upstream, so it starts near-closed and opens with
/// training.
public struct ConditionedTransitionBlock {
  let adaptiveNorm: AdaptiveLayerNorm
  let linearA1: AffineLinear
  let linearA2: AffineLinear
  let linearB: AffineLinear
  let linearS: AffineLinear

  public init(store: WeightStore, path: String) throws {
    adaptiveNorm = try AdaptiveLayerNorm(store: store, path: "\(path).adaln")
    linearA1 = try store.linear("\(path).linear_nobias_a1")
    linearA2 = try store.linear("\(path).linear_nobias_a2")
    linearB = try store.linear("\(path).linear_nobias_b")
    linearS = try store.linear("\(path).linear_s")
  }

  public func callAsFunction(_ a: MLXArray, _ s: MLXArray) -> MLXArray {
    let normalized = adaptiveNorm(a, s)
    let hidden = TensorOps.silu(linearA1(normalized)) * linearA2(normalized)
    return MLX.sigmoid(linearS(s)) * linearB(hidden)
  }
}

/// Algorithm 23, lines 2-3: one diffusion transformer block.
///
/// Both sub-layers are residual, but the residual is added *outside* each sub-layer —
/// and the feed-forward reads the post-attention value, not the block input.
public struct DiffusionTransformerBlock {
  let attentionPairBias: AttentionPairBias
  let transition: ConditionedTransitionBlock

  public init(store: WeightStore, path: String, headCount: Int) throws {
    attentionPairBias = try AttentionPairBias(
      store: store, path: "\(path).attention_pair_bias", headCount: headCount)
    transition = try ConditionedTransitionBlock(
      store: store, path: "\(path).conditioned_transition_block")
  }

  public func callAsFunction(_ a: MLXArray, _ s: MLXArray, _ z: MLXArray) -> MLXArray {
    let attended = attentionPairBias(a, s: s, z: z) + a
    return transition(attended, s) + attended
  }
}

/// Algorithm 23: the diffusion transformer — 8 blocks in tiny/mini, 24 in base and v2.
///
/// `s` and `z` are conditioning and are never updated; only `a` flows through.
public struct DiffusionTransformer {
  public let blocks: [DiffusionTransformerBlock]

  public var blockCount: Int { blocks.count }

  public init(store: WeightStore, path: String, blockCount: Int, headCount: Int)
    throws
  {
    blocks = try (0..<blockCount).map { index in
      try DiffusionTransformerBlock(
        store: store, path: "\(path).blocks.\(index)", headCount: headCount)
    }
  }

  public func callAsFunction(_ a: MLXArray, _ s: MLXArray, _ z: MLXArray) -> MLXArray {
    var value = a
    for block in blocks {
      value = block(value, s, z)
      MLX.eval(value)
    }
    return value
  }
}
