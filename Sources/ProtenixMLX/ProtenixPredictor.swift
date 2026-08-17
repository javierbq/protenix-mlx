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
    sampler = DiffusionSampler(module: diffusion)

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
        // it takes upstream's defaults: 16 for AttentionPairBias, 4 for
        // TriangleAttention. These are NOT the trunk Pairformer's, which the config
        // does name.
        headCount: 16, pairHeadCount: 4)
    }
  }

  /// Fold, returning `[N_atom, 3]` coordinates for one sample.
  ///
  /// - Parameters:
  ///   - recyclingSteps/diffusionSteps: override the artifact's defaults; nil uses them.
  public func fold(
    bundle: FeatureBundle, seed: UInt64 = 0, recyclingSteps: Int? = nil,
    diffusionSteps: Int? = nil
  ) throws -> MLXArray {
    try predict(
      bundle: bundle, seed: seed, recyclingSteps: recyclingSteps,
      diffusionSteps: diffusionSteps, scoring: false
    ).coordinates
  }

  /// Fold and score: coordinates plus the confidence head's read on them.
  ///
  /// Costs a second Pairformer pass over the pair representation on top of the fold, so
  /// it is a separate entry point rather than something `fold` always does.
  public func foldScored(
    bundle: FeatureBundle, seed: UInt64 = 0, recyclingSteps: Int? = nil,
    diffusionSteps: Int? = nil
  ) throws -> Prediction {
    guard reportsConfidence else {
      throw ProtenixError.missingTensor("confidence_head.plddt_weight")
    }
    return try predict(
      bundle: bundle, seed: seed, recyclingSteps: recyclingSteps,
      diffusionSteps: diffusionSteps, scoring: true)
  }

  private func predict(
    bundle: FeatureBundle, seed: UInt64, recyclingSteps: Int?, diffusionSteps: Int?,
    scoring: Bool
  ) throws -> Prediction {
    let atomFeatures = try bundle.atomFeatures()
    let tokenCount = bundle.metadata.tokenCount
    let atomCount = bundle.metadata.atomCount

    let trunkOutput = trunk(
      atomFeatures: atomFeatures,
      restype: bundle.batched(try bundle.tensor("restype")),
      profile: bundle.batched(try bundle.tensor("profile")),
      deletionMean: bundle.batched(try bundle.tensor("deletion_mean")),
      relativeFeatures: bundle.batched(try bundle.tensor("relp")),
      tokenBonds: bundle.batched(try bundle.tensor("token_bonds")),
      msaFeatures: bundle.batched(try bundle.tensor("msa_features")),
      tokenCount: tokenCount)
    MLX.eval(trunkOutput.s, trunkOutput.z, trunkOutput.sInputs)

    let context = DiffusionModule.Context(
      features: atomFeatures, relativeFeatures: bundle.batched(try bundle.tensor("relp")),
      sInputs: trunkOutput.sInputs, sTrunk: trunkOutput.s, zTrunk: trunkOutput.z,
      tokenCount: tokenCount)

    let sampled = sampler(
      context: context, atomCount: atomCount, nSamples: 1,
      steps: diffusionSteps ?? self.diffusionSteps, seed: seed)
    MLX.eval(sampled)
    // [1, 1, N_atom, 3] -> [N_atom, 3].
    let coordinates = sampled.reshaped([atomCount, 3])

    guard scoring, let confidence else {
      return Prediction(coordinates: coordinates, scores: nil)
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
    return Prediction(coordinates: coordinates, scores: scores)
  }
}
