import Foundation
import MLX

/// Algorithm 24: multi-head attention over the single representation, biased by pairs.
///
/// Used in two configurations that are different graphs, not different arguments:
///
/// * `hasS == true` (Pairformer, diffusion transformer) conditions the input norm on the
///   single embedding via `AdaptiveLayerNorm` and gates the output by
///   `sigmoid(linear_a_last(s))`.
/// * `hasS == false` (confidence head, template stack) uses a plain LayerNorm and has no
///   output gate at all -- `linear_a_last` does not exist in the checkpoint.
///
/// Which one applies is decided by the weights present, not by a caller-supplied flag,
/// so a variant that differs here cannot be mis-driven.
public struct AttentionPairBias {
  public let headCount: Int
  public let hasS: Bool

  let adaptiveNorm: AdaptiveLayerNorm?
  let plainNorm: LayerNormWeights?
  /// Present only in cross-attention mode (`AtomTransformer`), where keys and values
  /// come from a SECOND normalization rather than from the query's. Applied to the
  /// already-normalized `a`, not to the raw input — upstream rebinds `a` before
  /// computing `kv`, so the two norms compose rather than branch.
  let adaptiveNormKV: AdaptiveLayerNorm?
  let plainNormKV: LayerNormWeights?
  let normZ: LayerNormWeights
  let linearZ: AffineLinear
  let linearQ: AffineLinear
  let linearK: AffineLinear
  let linearV: AffineLinear
  let linearG: AffineLinear
  let linearO: AffineLinear
  let linearALast: AffineLinear?
  /// Per-head width. Upstream passes `c_a / n_heads` as `c_hidden`, and the 1/sqrt
  /// scaling uses that per-head width -- not the full embedding width.
  let headWidth: Int

  public init(store: WeightStore, path: String, headCount: Int) throws {
    self.headCount = headCount
    let adaptivePath = "\(path).layernorm_a"
    self.hasS = !store.names(withPrefix: "\(adaptivePath).linear_s.weight").isEmpty
    if hasS {
      adaptiveNorm = try AdaptiveLayerNorm(store: store, path: adaptivePath)
      plainNorm = nil
      linearALast = try store.linear("\(path).linear_a_last")
    } else {
      adaptiveNorm = nil
      plainNorm = try store.layerNorm(adaptivePath)
      linearALast = nil
    }

    // Cross-attention mode is likewise read from the weights: `layernorm_kv` exists
    // only when upstream was built with cross_attention_mode=True.
    let kvPath = "\(path).layernorm_kv"
    let hasKV = !store.names(withPrefix: "\(kvPath).").isEmpty
    if hasKV && hasS {
      adaptiveNormKV = try AdaptiveLayerNorm(store: store, path: kvPath)
      plainNormKV = nil
    } else if hasKV {
      adaptiveNormKV = nil
      plainNormKV = try store.layerNorm(kvPath)
    } else {
      adaptiveNormKV = nil
      plainNormKV = nil
    }
    normZ = try store.layerNorm("\(path).layernorm_z")
    linearZ = try store.linear("\(path).linear_nobias_z")
    linearQ = try store.linear("\(path).attention.linear_q")
    linearK = try store.linear("\(path).attention.linear_k")
    linearV = try store.linear("\(path).attention.linear_v")
    linearG = try store.linear("\(path).attention.linear_g")
    linearO = try store.linear("\(path).attention.linear_o")
    headWidth = linearQ.weight.shape[0] / headCount
  }

  /// Normalize the query input. `hasS` with no `s` supplied falls back to an
  /// unconditioned norm, but that is a degenerate case -- a conditioned block driven
  /// without its conditioning is a different model.
  private func normalizeQuery(_ a: MLXArray, _ s: MLXArray?) -> MLXArray {
    if let adaptiveNorm, let s { return adaptiveNorm(a, s) }
    if let plainNorm { return TensorOps.layerNorm(a, plainNorm) }
    return TensorOps.layerNorm(a, weight: nil, bias: nil)
  }

  /// Key/value input. In cross-attention mode this is a second normalization of the
  /// ALREADY-normalized query input; otherwise keys and values share the query's.
  private func normalizeKeyValue(_ normalizedQuery: MLXArray, _ s: MLXArray?)
    -> MLXArray
  {
    if let adaptiveNormKV, let s { return adaptiveNormKV(normalizedQuery, s) }
    if let plainNormKV { return TensorOps.layerNorm(normalizedQuery, plainNormKV) }
    return normalizedQuery
  }

  /// - Parameters:
  ///   - a: `[..., N, c_a]` single representation.
  ///   - s: `[..., N, c_s]` conditioning; required when `hasS`, ignored otherwise.
  ///   - z: `[..., N, N, c_z]` pair representation supplying the attention bias.
  public func callAsFunction(_ a: MLXArray, s: MLXArray?, z: MLXArray) -> MLXArray {
    let value = normalizeQuery(a, s)
    let keyValue = normalizeKeyValue(value, s)

    // Bias is [..., N, N, heads] projected from z, then moved to [..., heads, N, N].
    let bias = TensorOps.permuteFinalDims(
      linearZ(TensorOps.layerNorm(z, normZ)), [2, 0, 1])

    let query = TensorOps.splitHeads(linearQ(value), heads: headCount)
      / MLXArray(sqrt(Float(headWidth)))
    let key = TensorOps.splitHeads(linearK(keyValue), heads: headCount)
    let element = TensorOps.splitHeads(linearV(keyValue), heads: headCount)

    let attended = TensorOps.attention(
      query: query, key: key, value: element, bias: bias)
    // Gate is computed from the NORMALIZED input, per-head, before the heads are
    // merged -- gating the merged vector would apply the wrong slice to each head.
    let gate = MLX.sigmoid(
      TensorOps.splitHeads(linearG(value), heads: headCount))
    var output = linearO(TensorOps.mergeHeads(attended.asType(a.dtype) * gate))

    if let linearALast, let s {
      output = MLX.sigmoid(linearALast(s)) * output
    }
    return output
  }

  /// Algorithm 24 over a local window, as used by `AtomTransformer`.
  ///
  /// `z` arrives already trunked — `[..., trunks, nQueries, nKeys, c_z]` — because the
  /// atom-pair representation is only ever materialized in that blocked form; a dense
  /// `[N_atom, N_atom, c]` tensor would be prohibitive.
  ///
  /// `AtomTransformer` always runs in cross-attention mode, so keys and values here
  /// come from `layernorm_kv` applied on top of the query normalization.
  public func callAsFunction(
    local a: MLXArray, s: MLXArray?, trunkedZ: MLXArray,
    trunking: LocalAttention.Trunking
  ) -> MLXArray {
    let value = normalizeQuery(a, s)
    let keyValue = normalizeKeyValue(value, s)

    // [..., trunks, nQueries, nKeys, heads] -> [..., heads, trunks, nQueries, nKeys]
    let pairBias = TensorOps.permuteFinalDims(
      linearZ(TensorOps.layerNorm(trunkedZ, normZ)), [3, 0, 1, 2]
    ).asType(.float32)
    let windowBias = LocalAttention.windowBias(trunking)
    let bias = pairBias + windowBias

    let queries = TensorOps.splitHeads(linearQ(value), heads: headCount)
      / MLXArray(Float(headWidth).squareRoot())
    let keys = TensorOps.splitHeads(linearK(keyValue), heads: headCount)
    let values = TensorOps.splitHeads(linearV(keyValue), heads: headCount)

    // Head axis is already ahead of the atom axis, so trunking splits the atom axis
    // in place: [..., H, N, C] -> [..., H, trunks, window, C].
    let q = LocalAttention.trunkQueries(queries, trunking)
    let k = LocalAttention.trunkKeys(keys, trunking)
    let v = LocalAttention.trunkKeys(values, trunking)

    let attended = TensorOps.attention(query: q, key: k, value: v, bias: bias)
    let merged = LocalAttention.untrunk(attended, trunking)

    let gate = MLX.sigmoid(TensorOps.splitHeads(linearG(value), heads: headCount))
    var output = linearO(TensorOps.mergeHeads(merged.asType(a.dtype) * gate))
    if let linearALast, let s {
      output = MLX.sigmoid(linearALast(s)) * output
    }
    return output
  }
}

/// Algorithm 7: a local transformer over atoms, biased by the atom-pair representation.
///
/// Structurally a `DiffusionTransformer` whose attention is windowed. Upstream builds it
/// as exactly that — `DiffusionTransformer(cross_attention_mode=True)` — so the weights
/// live under `diffusion_transformer.blocks.*` and are loaded through the same path.
public struct AtomTransformer {
  public let blocks: [AtomTransformerBlock]
  public let queryWindow: Int
  public let keyWindow: Int

  public var blockCount: Int { blocks.count }

  public init(
    store: WeightStore, path: String, blockCount: Int, headCount: Int,
    queryWindow: Int = 32, keyWindow: Int = 128
  ) throws {
    self.queryWindow = queryWindow
    self.keyWindow = keyWindow
    blocks = try (0..<blockCount).map { index in
      try AtomTransformerBlock(
        store: store, path: "\(path).diffusion_transformer.blocks.\(index)",
        headCount: headCount)
    }
  }

  /// - Parameters:
  ///   - q: `[..., N_atom, c_atom]` atom representation being updated.
  ///   - c: `[..., N_atom, c_atom]` atom conditioning.
  ///   - p: `[..., trunks, nQueries, nKeys, c_atompair]` blocked atom-pair embedding.
  public func callAsFunction(_ q: MLXArray, _ c: MLXArray, _ p: MLXArray) -> MLXArray {
    let trunking = LocalAttention.Trunking(
      atomCount: q.shape[q.ndim - 2], queryWindow: queryWindow, keyWindow: keyWindow)
    var value = q
    for block in blocks {
      value = block(value, c, p, trunking)
      MLX.eval(value)
    }
    return value
  }
}

/// One block of `AtomTransformer`: windowed attention then a conditioned transition.
public struct AtomTransformerBlock {
  let attentionPairBias: AttentionPairBias
  let transition: ConditionedTransitionBlock

  public init(store: WeightStore, path: String, headCount: Int) throws {
    attentionPairBias = try AttentionPairBias(
      store: store, path: "\(path).attention_pair_bias", headCount: headCount)
    transition = try ConditionedTransitionBlock(
      store: store, path: "\(path).conditioned_transition_block")
  }

  public func callAsFunction(
    _ a: MLXArray, _ s: MLXArray, _ z: MLXArray,
    _ trunking: LocalAttention.Trunking
  ) -> MLXArray {
    let attended =
      attentionPairBias(local: a, s: s, trunkedZ: z, trunking: trunking) + a
    return transition(attended, s) + attended
  }
}
