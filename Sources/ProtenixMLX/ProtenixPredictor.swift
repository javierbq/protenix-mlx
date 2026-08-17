import Foundation
import MLX

/// Folds a sequence: feature bundle + model artifact -> atom coordinates.
///
/// Assembles the trunk, diffusion module and confidence head from an artifact's own
/// `config.json`, so a single predictor serves every variant. No affinity head and no
/// templates.
public struct ProtenixPredictor {
  public let artifact: ProtenixArtifact
  let trunk: ProtenixTrunk
  let diffusion: DiffusionModule
  let sampler: DiffusionSampler
  /// Absent only for a pack that carries no confidence weights; every released one does.
  let confidence: ConfidenceHead?
  public let recyclingSteps: Int
  public let diffusionSteps: Int

  /// Whether this pack can report confidence. False makes `foldScored` refuse rather
  /// than return a structure with fabricated certainty attached.
  public var reportsConfidence: Bool { confidence != nil }

  /// A fold and what the model thinks of it.
  public struct Prediction {
    /// `[N_atom, 3]`.
    public let coordinates: MLXArray
    /// Absent when the pack carries no confidence head.
    public let scores: ConfidenceHead.Scores?
  }

  public init(artifact: ProtenixArtifact) throws {
    self.artifact = artifact
    let store = WeightStore(artifact: artifact)
    let config = store.configuration
    let model = config.model

    self.recyclingSteps = config.nCycle
    self.diffusionSteps = config.nDiffusionSteps

    trunk = try ProtenixTrunk(store: store, cycleCount: config.nCycle)
    diffusion = try DiffusionModule(
      store: store, path: "diffusion_module",
      sigmaData: Float(model.diffusionModule.sigmaData),
      atomEncoderBlocks: model.diffusionModule.atomEncoder.nBlocks,
      atomEncoderHeads: model.diffusionModule.atomEncoder.nHeads,
      transformerBlocks: model.diffusionModule.transformer.nBlocks,
      transformerHeads: model.diffusionModule.transformer.nHeads,
      atomDecoderBlocks: model.diffusionModule.atomDecoder.nBlocks,
      atomDecoderHeads: model.diffusionModule.atomDecoder.nHeads,
      rMax: model.relativePositionEncoding.rMax,
      sMax: model.relativePositionEncoding.sMax)
    // From the artifact, NOT from the struct defaults. These constants are not weights
    // and not derivable from them, so nothing in the parity suite covers them -- the
    // sampler draws its own noise and therefore has no PyTorch fixture. Running with
    // stepScaleEta 1.0 instead of upstream's 1.5 does not crash or produce garbage; it
    // produces a plausible structure whose bonds are systematically ~10% short.
    let diffusionConfig = config.sampleDiffusion
    sampler = DiffusionSampler(
      module: diffusion,
      stepScaleEta: diffusionConfig?.stepScaleEta ?? 1.5,
      gamma0: diffusionConfig?.gamma0 ?? 0.8,
      gammaMin: diffusionConfig?.gammaMin ?? 1.0,
      noiseScaleLambda: diffusionConfig?.noiseScaleLambda ?? 1.003)

    // Loaded only if the pack actually carries the weights. Every released pack does,
    // but a pack exported before the head was included would otherwise fail to load
    // entirely rather than simply folding without confidence.
    if store.names(withPrefix: "confidence_head.").isEmpty {
      confidence = nil
    } else {
      confidence = try ConfidenceHead(
        store: store, path: "confidence_head",
        configuration: model.confidenceHead,
        // The confidence head does not forward a head count to its own Pairformer, so
        // AttentionPairBias takes upstream's default of 16. TriangleAttention's count is
        // NOT a constant, though: under `hidden_scale_up` upstream recomputes it as
        // `c_z / c_hidden_pair_att`, so v2's 256-wide pair track needs 8 heads where
        // every other variant needs 4. Hardcoding 4 loaded v2's weights into a
        // 4-head layout and died at the first triangle attention with
        // "Shapes (1,N,4,N,N) and (1,1,8,N,N) cannot be broadcast" -- the trunk got this
        // right and only the head did not, so it reproduced on v2 alone.
        headCount: 16,
        pairHeadCount: model.confidenceHead.hiddenScaleUp
          ? model.confidenceHead.cZ / 32 : 4)
    }
  }

  /// Fold, returning `[N_atom, 3]` coordinates for one sample.
  ///
  /// - Parameters:
  ///   - recyclingSteps/diffusionSteps: override the artifact's defaults; nil uses them.
  ///   - progress: called after each trunk recycle and each diffusion step; return
  ///     false to cancel, which throws ``ProtenixError/cancelled``.
  public func fold(
    bundle: FeatureBundle, seed: UInt64 = 0, recyclingSteps: Int? = nil,
    diffusionSteps: Int? = nil, progress: FoldProgressHandler? = nil
  ) throws -> MLXArray {
    try predict(
      bundle: bundle, seed: seed, recyclingSteps: recyclingSteps,
      diffusionSteps: diffusionSteps, scoring: false, progress: progress
    ).coordinates
  }

  /// Fold and score: coordinates plus the confidence head's read on them.
  ///
  /// Costs a second Pairformer pass over the pair representation on top of the fold, so
  /// it is a separate entry point rather than something `fold` always does.
  public func foldScored(
    bundle: FeatureBundle, seed: UInt64 = 0, recyclingSteps: Int? = nil,
    diffusionSteps: Int? = nil, progress: FoldProgressHandler? = nil
  ) throws -> Prediction {
    guard reportsConfidence else {
      throw ProtenixError.missingTensor("confidence_head.plddt_weight")
    }
    return try predict(
      bundle: bundle, seed: seed, recyclingSteps: recyclingSteps,
      diffusionSteps: diffusionSteps, scoring: true, progress: progress)
  }

  private func predict(
    bundle: FeatureBundle, seed: UInt64, recyclingSteps: Int?, diffusionSteps: Int?,
    scoring: Bool, progress: FoldProgressHandler? = nil
  ) throws -> Prediction {
    let atomFeatures = try bundle.atomFeatures()
    let tokenCount = bundle.metadata.tokenCount
    let atomCount = bundle.metadata.atomCount

    let trunkOutput = try trunk(
      atomFeatures: atomFeatures,
      restype: bundle.batched(try bundle.tensor("restype")),
      profile: bundle.batched(try bundle.tensor("profile")),
      deletionMean: bundle.batched(try bundle.tensor("deletion_mean")),
      relativeFeatures: bundle.batched(try bundle.tensor("relp")),
      tokenBonds: bundle.batched(try bundle.tensor("token_bonds")),
      msaFeatures: bundle.batched(try bundle.tensor("msa_features")),
      tokenCount: tokenCount, progress: progress)
    MLX.eval(trunkOutput.s, trunkOutput.z, trunkOutput.sInputs)

    let context = DiffusionModule.Context(
      features: atomFeatures, relativeFeatures: bundle.batched(try bundle.tensor("relp")),
      sInputs: trunkOutput.sInputs, sTrunk: trunkOutput.s, zTrunk: trunkOutput.z,
      tokenCount: tokenCount)

    let sampled = try sampler(
      context: context, atomCount: atomCount, nSamples: 1,
      steps: diffusionSteps ?? self.diffusionSteps, seed: seed, progress: progress)
    MLX.eval(sampled)
    // [1, 1, N_atom, 3] -> [N_atom, 3].
    let coordinates = sampled.reshaped([atomCount, 3])

    guard scoring, let confidence else {
      return Prediction(coordinates: coordinates, scores: nil)
    }
    if let progress,
      !progress(FoldProgress(phase: .confidence, step: 0, total: 1))
    {
      throw ProtenixError.cancelled(phase: "confidence", step: 0)
    }
    // Scored on the coordinates just produced -- the head predicts the error in THIS
    // structure, so it cannot be run before the sampler or reused across seeds.
    let scores = confidence(
      coordinates: coordinates, sInputs: trunkOutput.sInputs, sTrunk: trunkOutput.s,
      zTrunk: trunkOutput.z,
      atomToToken: try bundle.tensor("atom_to_token_idx"),
      representativeAtoms: try bundle.tensor("distogram_rep_atom_mask"),
      atomToTokenAtom: try bundle.tensor("atom_to_tokatom_idx"),
      tokenCount: tokenCount)
    _ = progress?(FoldProgress(phase: .confidence, step: 1, total: 1))
    return Prediction(coordinates: coordinates, scores: scores)
  }
}
