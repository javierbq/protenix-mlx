import Foundation
import MLX
import MLXRandom

/// Algorithm 18: the EDM predictor-corrector sampler.
///
/// Runs the denoiser from high noise to zero, producing atom coordinates from pure
/// noise conditioned on the trunk. The per-step denoise is bitwise-verified against
/// PyTorch (`DiffusionModule`); the loop around it is not, because it draws fresh noise
/// and a random augmentation each step and no cross-framework RNG match is possible.
/// A `seed` is threaded so a run is reproducible within this framework.
public struct DiffusionSampler {
  public let module: DiffusionModule
  public let stepScaleEta: Float
  public let gamma0: Float
  public let gammaMin: Float
  public let noiseScaleLambda: Float

  public init(
    module: DiffusionModule, stepScaleEta: Float = 1.0, gamma0: Float = 0.0,
    gammaMin: Float = 1.0, noiseScaleLambda: Float = 1.003
  ) {
    self.module = module
    self.stepScaleEta = stepScaleEta
    self.gamma0 = gamma0
    self.gammaMin = gammaMin
    self.noiseScaleLambda = noiseScaleLambda
  }

  /// The inference noise schedule, `[nSteps + 1]`, descending to 0.
  ///
  /// `sigma_data * (s_max^(1/rho) + t*(s_min^(1/rho) - s_max^(1/rho)))^rho`, with the
  /// final level pinned to exactly 0 — the last step must land at zero noise.
  public static func noiseSchedule(
    steps: Int, sigmaData: Float = 16.0, sMax: Float = 160.0, sMin: Float = 4e-4,
    rho: Float = 7.0
  ) -> [Float] {
    let stepSize = 1.0 / Float(steps)
    let maxTerm = pow(sMax, 1 / rho)
    let minTerm = pow(sMin, 1 / rho)
    var schedule = (0...steps).map { index -> Float in
      let t = Float(index) * stepSize
      return sigmaData * pow(maxTerm + t * (minTerm - maxTerm), rho)
    }
    schedule[steps] = 0
    return schedule
  }

  /// Fold: sample `[..., nSamples, N_atom, 3]` coordinates from the trunk.
  ///
  /// - Parameters:
  ///   - context: trunk embeddings and atom features; its `pairConditioning` is filled
  ///     in here once and reused across every step.
  ///   - atomCount: N_atom, the number of atoms to place.
  ///   - steps: diffusion steps (5 for mini/tiny, 200 for base/v2).
  ///   - progress: called after each step; return false to cancel, which throws
  ///     ``ProtenixError/cancelled``. Between steps is the finest granularity that
  ///     exists — the MLX work of a step is already in flight by the time it starts.
  public func callAsFunction(
    context input: DiffusionModule.Context, atomCount: Int, nSamples: Int = 1,
    steps: Int, seed: UInt64 = 0, progress: FoldProgressHandler? = nil
  ) throws -> MLXArray {
    var key = MLXRandom.key(seed)
    let schedule = Self.noiseSchedule(steps: steps, sigmaData: module.sigmaData)

    // Hoist the noise-independent pair conditioning out of the loop.
    let context = DiffusionModule.Context(
      features: input.features, relativeFeatures: input.relativeFeatures,
      sInputs: input.sInputs, sTrunk: input.sTrunk, zTrunk: input.zTrunk,
      tokenCount: input.tokenCount,
      pairConditioning: module.prepareConditioning(
        zTrunk: input.zTrunk, relativeFeatures: input.relativeFeatures))

    let batch = Array(input.sInputs.shape.dropLast(2))
    let shape = batch + [nSamples, atomCount, 3]

    // x_0 = sigma_max * noise.
    let (initKey, firstStep) = split(&key)
    var x = MLXArray(schedule[0]) * MLXRandom.normal(shape, key: initKey)
    key = firstStep

    for index in 0..<steps {
      let sigmaLast = schedule[index]
      let sigmaNext = schedule[index + 1]

      x = centreRandomAugmentation(x, key: &key)

      // Predictor: inflate to t_hat, adding matched noise.
      let gamma = sigmaNext > gammaMin ? gamma0 : 0
      let tHat = sigmaLast * (gamma + 1)
      let deltaNoise = (tHat * tHat - sigmaLast * sigmaLast).squareRoot()
      let (noiseKey, nextKey) = split(&key)
      key = nextKey
      let xNoisy =
        x + noiseScaleLambda * deltaNoise * MLXRandom.normal(x.shape, key: noiseKey)

      // Corrector: one Euler step toward the denoised estimate.
      let level = MLXArray(Array(repeating: tHat, count: nSamples)).reshaped(
        batch + [nSamples])
      let denoised = module(xNoisy: xNoisy, noiseLevel: level, context: context)
      let delta = (xNoisy - denoised) / tHat
      let dt = sigmaNext - tHat
      x = xNoisy + stepScaleEta * dt * delta
      // Evaluated per step rather than at the end: the graph is what gets cancelled
      // between steps, and an unevaluated 200-step graph would defer all the work past
      // every cancellation check, making the handler a no-op that looks like one.
      MLX.eval(x)
      if let progress,
        !progress(FoldProgress(phase: .diffusion, step: index + 1, total: steps))
      {
        throw ProtenixError.cancelled(phase: "diffusion", step: index + 1)
      }
    }
    return x
  }

  /// Algorithm 19: centre on the origin, then apply one random rotation and small
  /// random translation. Centering is the load-bearing part — the network is trained
  /// on origin-centred inputs — and the rotation keeps the sampler from committing to
  /// one orientation early.
  func centreRandomAugmentation(_ x: MLXArray, key: inout MLXArray)
    -> MLXArray
  {
    let centred = x - x.mean(axis: -2, keepDims: true)
    let (rotationKey, translationKey) = split(&key)
    key = translationKey
    let rotation = randomRotation(
      batchShape: Array(x.shape.dropLast(2)), key: rotationKey)
    // [..., N, 3] @ [..., 3, 3]
    let rotated = MLX.matmul(centred, rotation.swappedAxes(-1, -2))
    let translation = MLXRandom.normal(
      Array(x.shape.dropLast(2)) + [1, 3], key: translationKey)
    return rotated + translation
  }

  /// A uniform random rotation matrix per batch element, via QR of a Gaussian matrix.
  private func randomRotation(batchShape: [Int], key: MLXArray) -> MLXArray {
    let gaussian = MLXRandom.normal(batchShape + [3, 3], key: key)
    // QR is CPU-only in MLX today; pinning the stream keeps the rest of the sampler on
    // the GPU. The matrices are 3x3, so the round-trip is negligible.
    let (q, r) = MLX.qr(gaussian.asType(.float32), stream: .cpu)
    // Fix the sign so det(Q) = +1 (a rotation, not a reflection): scale each column by
    // the sign of the corresponding R diagonal.
    let diagonalSign = MLX.sign(r.diagonal(axis1: -2, axis2: -1))
      .expandedDimensions(axis: -2)
    return q * diagonalSign
  }

  private func split(_ key: inout MLXArray) -> (MLXArray, MLXArray) {
    let parts = MLXRandom.split(key: key)
    return (parts.0, parts.1)
  }
}
