import Foundation
import MLX

/// Folds a sequence: feature bundle + model artifact -> atom coordinates.
///
/// Assembles the trunk and diffusion module from an artifact's own `config.json`, so a
/// single predictor serves every variant. Structure path only — no confidence, no
/// affinity.
public struct ProtenixPredictor {
  public let artifact: ProtenixArtifact
  let trunk: ProtenixTrunk
  let diffusion: DiffusionModule
  let sampler: DiffusionSampler
  public let recyclingSteps: Int
  public let diffusionSteps: Int

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
  }

  /// Fold, returning `[N_atom, 3]` coordinates for one sample.
  ///
  /// - Parameters:
  ///   - recyclingSteps/diffusionSteps: override the artifact's defaults; nil uses them.
  public func fold(
    bundle: FeatureBundle, seed: UInt64 = 0, recyclingSteps: Int? = nil,
    diffusionSteps: Int? = nil
  ) throws -> MLXArray {
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

    let coordinates = sampler(
      context: context, atomCount: atomCount, nSamples: 1,
      steps: diffusionSteps ?? self.diffusionSteps, seed: seed)
    MLX.eval(coordinates)
    // [1, 1, N_atom, 3] -> [N_atom, 3].
    return coordinates.reshaped([atomCount, 3])
  }
}
