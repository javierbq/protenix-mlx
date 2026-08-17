import MLX

/// Algorithm 2: the input feature embedder.
///
/// Produces the `s_inputs` single embedding the trunk and diffusion conditioning both
/// consume: a coordinate-free atom-attention pass folded to tokens, concatenated with
/// the per-token `restype` / `profile` / `deletion_mean` features.
///
/// ESM is not supported — those variants need ESM2-3B alongside and are out of scope —
/// so `linear_esm` is neither loaded nor applied.
public struct InputFeatureEmbedder {
  let encoder: AtomAttentionEncoder

  public init(store: WeightStore, path: String, blockCount: Int, headCount: Int)
    throws
  {
    encoder = try AtomAttentionEncoder(
      store: store, path: "\(path).atom_attention_encoder", blockCount: blockCount,
      headCount: headCount)
  }

  /// - Parameters:
  ///   - features: the atom-level features (coordinate-free — `r`/`s`/`z` unused).
  ///   - restype: `[..., N_token, 32]` one-hot residue type.
  ///   - profile: `[..., N_token, 32]` MSA profile.
  ///   - deletionMean: `[..., N_token, 1]` mean deletion count.
  /// - Returns: `[..., N_token, c_token + 65]` = `s_inputs`.
  public func callAsFunction(
    features: AtomAttentionEncoder.Features, restype: MLXArray, profile: MLXArray,
    deletionMean: MLXArray, tokenCount: Int
  ) -> MLXArray {
    let a = encoder(features, tokenCount: tokenCount).a
    return MLX.concatenated([a, restype, profile, deletionMean], axis: -1)
  }
}
