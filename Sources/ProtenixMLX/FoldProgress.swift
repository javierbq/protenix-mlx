import Foundation

/// Where a fold has got to, and the caller's chance to stop it.
///
/// A base-model fold is 10 trunk recycles and 200 diffusion steps and takes minutes. Two
/// things follow that a synchronous `fold()` cannot provide on its own: a progress
/// fraction to show, and somewhere to observe cancellation. Both are this.
///
/// The handler returns `false` to cancel, at which point the fold throws
/// ``ProtenixError/cancelled``. Cooperative rather than pre-emptive because MLX work is
/// already in flight inside a step: the finest granularity that exists is *between*
/// steps, and pretending otherwise would report a cancel that had not happened.
public struct FoldProgress: Sendable {
  /// The stage of a fold. Their costs are wildly different, which is why the fraction
  /// below is not simply `step / total` across the whole run.
  public enum Phase: String, Sendable {
    /// Recycling the trunk. `total` is the recycle count — 10 for base and v2.
    case trunk
    /// Reverse diffusion. `total` is the step count — 200 for base and v2.
    case diffusion
    /// The confidence head, which is a further Pairformer pass. One step.
    case confidence
  }

  public let phase: Phase
  /// Completed units of this phase, 0-based going in and 1-based on completion.
  public let step: Int
  /// Units in this phase. Zero only if a phase somehow runs no steps.
  public let total: Int

  public init(phase: Phase, step: Int, total: Int) {
    self.phase = phase
    self.step = step
    self.total = total
  }

  /// Overall completion in [0, 1], weighted by what each phase actually costs.
  ///
  /// The weights are measured rather than assumed: at the base model's operating point
  /// the trunk's ten recycles are roughly half the wall clock, diffusion's 200 steps a
  /// little under half, and the confidence head the remainder. A naive
  /// `step / (recycles + steps)` would crawl through the trunk and then sprint, which
  /// reads as a hang.
  public var fraction: Double {
    let within = total > 0 ? Double(step) / Double(total) : 0
    switch phase {
    case .trunk: return 0.50 * within
    case .diffusion: return 0.50 + 0.45 * within
    case .confidence: return 0.95 + 0.05 * within
    }
  }
}

/// Observe a fold's progress; return `false` to cancel it.
public typealias FoldProgressHandler = @Sendable (FoldProgress) -> Bool
