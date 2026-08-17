import Foundation

/// The architecture contract frozen into an artifact's `config.json`.
///
/// This file is not optional metadata. A Protenix checkpoint stores only
/// `{"model": state_dict, "model_version": str}` -- no dimensions, no block counts --
/// and the four released variants share tensor *names* while differing in depth and
/// width. Without this the runtime cannot know whether it is holding an 8-block or a
/// 48-block Pairformer, so `ProtenixArtifact` treats a missing config as fatal.
///
/// Decoded with explicit coding keys throughout rather than a snake-case strategy:
/// upstream mixes conventions (`N_cycle` beside `n_blocks` beside `c_z`), and a
/// key-mangling rule that silently misses one produces a default-valued dimension
/// instead of an error.
public struct ProtenixModelConfiguration: Codable, Sendable, Equatable {
  public static let supportedSchemaVersion = 1

  public let schemaVersion: Int
  public let modelName: String
  public let sourceRevision: String
  public let sourceCommit: String

  public let cS: Int
  public let cZ: Int
  public let cSInputs: Int
  public let cAtom: Int
  public let cAtompair: Int
  public let cToken: Int

  /// Upstream's default recycling count for this variant (4 for mini/tiny, 10 for
  /// base/v2). A default, not a limit -- callers may run fewer.
  public let nCycle: Int
  /// Upstream's default diffusion step count (5 for mini/tiny, 200 for base/v2).
  public let nDiffusionSteps: Int

  public let model: ModelSection
  /// The sampler's own constants. OPTIONAL so a pack exported before they were carried
  /// still loads -- but a pack that omits them gets upstream's values, not this
  /// runtime's guess, because the two silently differing is what this section is here to
  /// prevent. See ``SampleDiffusionSection``.
  public let sampleDiffusion: SampleDiffusionSection?
  /// The noise schedule's constants, likewise optional.
  public let inferenceNoiseScheduler: NoiseSchedulerSection?
  public let parameterCount: Int
  public let quantizedMatrixCount: Int
  public let graphRoots: [String]

  private enum CodingKeys: String, CodingKey {
    case schemaVersion = "schema_version"
    case modelName = "model_name"
    case sourceRevision = "source_revision"
    case sourceCommit = "source_commit"
    case cS = "c_s"
    case cZ = "c_z"
    case cSInputs = "c_s_inputs"
    case cAtom = "c_atom"
    case cAtompair = "c_atompair"
    case cToken = "c_token"
    case nCycle = "n_cycle"
    case nDiffusionSteps = "n_diffusion_steps"
    case model
    case sampleDiffusion = "sample_diffusion"
    case inferenceNoiseScheduler = "inference_noise_scheduler"
    case parameterCount = "parameter_count"
    case quantizedMatrixCount = "quantized_matrix_count"
    case graphRoots = "graph_roots"
  }

  /// The sampler constants from upstream's `sample_diffusion` config.
  ///
  /// These are NOT weights and NOT derivable from them, which is what makes them
  /// dangerous: the parity suite cannot catch a wrong one. Every learned module is
  /// checked against a recorded PyTorch fixture, but the sampler draws its own noise, so
  /// it has no fixture — and a wrong `stepScaleEta` does not crash or produce garbage. It
  /// produces a plausible-looking structure with systematically compressed bonds, which
  /// is exactly how this was found: CA-CA distances came out at 3.4 A against an ideal
  /// 3.8 after the port ran with the struct defaults (1.0 / 0.0) rather than upstream's.
  public struct SampleDiffusionSection: Codable, Sendable, Equatable {
    /// Churn: how much noise is re-injected before each Euler step. 0.8 upstream; at 0
    /// the sampler is deterministic-descent and under-converges.
    public let gamma0: Float
    /// Churn applies only above this noise level.
    public let gammaMin: Float
    /// Scales the re-injected noise.
    public let noiseScaleLambda: Float
    /// Scales the Euler step. 1.5 upstream; at 1.0 every step under-shoots.
    public let stepScaleEta: Float

    private enum CodingKeys: String, CodingKey {
      case gamma0
      case gammaMin = "gamma_min"
      case noiseScaleLambda = "noise_scale_lambda"
      case stepScaleEta = "step_scale_eta"
    }
  }

  /// The EDM noise schedule's constants, from upstream's `inference_noise_scheduler`.
  public struct NoiseSchedulerSection: Codable, Sendable, Equatable {
    public let rho: Float
    public let sMax: Float
    public let sMin: Float
    public let sigmaData: Float

    private enum CodingKeys: String, CodingKey {
      case rho
      case sMax = "s_max"
      case sMin = "s_min"
      case sigmaData = "sigma_data"
    }
  }

  /// Upstream's `configs.model` subtree, narrowed to what inference reads.
  ///
  /// The exported JSON also carries training-only knobs (`dropout`, `blocks_per_ckpt`,
  /// `use_fine_grained_checkpoint`). They are intentionally not decoded here: an
  /// inference runtime that read them could only misuse them.
  public struct ModelSection: Codable, Sendable, Equatable {
    public let pairformer: Stack
    public let msaModule: MSASection
    public let templateEmbedder: TemplateSection
    public let confidenceHead: ConfidenceSection
    public let diffusionModule: DiffusionSection
    public let distogramHead: DistogramSection
    public let inputEmbedder: InputEmbedderSection
    public let relativePositionEncoding: RelativePositionSection

    private enum CodingKeys: String, CodingKey {
      case pairformer
      case msaModule = "msa_module"
      case templateEmbedder = "template_embedder"
      case confidenceHead = "confidence_head"
      case diffusionModule = "diffusion_module"
      case distogramHead = "distogram_head"
      case inputEmbedder = "input_embedder"
      case relativePositionEncoding = "relative_position_encoding"
    }
  }

  public struct Stack: Codable, Sendable, Equatable {
    public let nBlocks: Int
    public let nHeads: Int
    public let cS: Int
    public let cZ: Int
    /// protenix-v2 widens every transition's hidden dimension with this flag rather
    /// than by changing a dimension, so it changes weight SHAPES and must be read.
    public let hiddenScaleUp: Bool

    private enum CodingKeys: String, CodingKey {
      case nBlocks = "n_blocks"
      case nHeads = "n_heads"
      case cS = "c_s"
      case cZ = "c_z"
      case hiddenScaleUp = "hidden_scale_up"
    }
  }

  public struct MSASection: Codable, Sendable, Equatable {
    public let nBlocks: Int
    public let cM: Int
    public let cZ: Int
    public let cSInputs: Int
    public let hiddenScaleUp: Bool
    public let msaMaxSize: Int

    private enum CodingKeys: String, CodingKey {
      case nBlocks = "n_blocks"
      case cM = "c_m"
      case cZ = "c_z"
      case cSInputs = "c_s_inputs"
      case hiddenScaleUp = "hidden_scale_up"
      case msaMaxSize = "msa_max_size"
    }
  }

  /// `nBlocks` is 0 for the v0.5.0 mini/tiny variants: they ship the template
  /// projections but no template stack, so templates are inert rather than absent.
  public struct TemplateSection: Codable, Sendable, Equatable {
    public let nBlocks: Int
    public let c: Int
    public let cZ: Int
    public let hiddenScaleUp: Bool

    public var isEnabled: Bool { nBlocks > 0 }

    private enum CodingKeys: String, CodingKey {
      case nBlocks = "n_blocks"
      case c
      case cZ = "c_z"
      case hiddenScaleUp = "hidden_scale_up"
    }
  }

  public struct ConfidenceSection: Codable, Sendable, Equatable {
    public let nBlocks: Int
    public let cS: Int
    public let cZ: Int
    public let cSInputs: Int
    public let maxAtomsPerToken: Int
    public let distanceBinStart: Double
    public let distanceBinEnd: Double
    public let distanceBinStep: Double
    public let hiddenScaleUp: Bool

    /// Number of distance bins, matching upstream's
    /// `arange(start, end, step)` -- the count the `linear_no_bias_d` input width
    /// was trained against.
    public var binCount: Int {
      max(0, Int(((distanceBinEnd - distanceBinStart) / distanceBinStep).rounded(.up)))
    }

    private enum CodingKeys: String, CodingKey {
      case nBlocks = "n_blocks"
      case cS = "c_s"
      case cZ = "c_z"
      case cSInputs = "c_s_inputs"
      case maxAtomsPerToken = "max_atoms_per_token"
      case distanceBinStart = "distance_bin_start"
      case distanceBinEnd = "distance_bin_end"
      case distanceBinStep = "distance_bin_step"
      case hiddenScaleUp = "hidden_scale_up"
    }
  }

  public struct DiffusionSection: Codable, Sendable, Equatable {
    public let cAtom: Int
    public let cAtompair: Int
    public let cToken: Int
    public let cS: Int
    public let cZ: Int
    public let cSInputs: Int
    public let sigmaData: Double
    public let atomEncoder: AtomStack
    public let atomDecoder: AtomStack
    public let transformer: AtomStack

    private enum CodingKeys: String, CodingKey {
      case cAtom = "c_atom"
      case cAtompair = "c_atompair"
      case cToken = "c_token"
      case cS = "c_s"
      case cZ = "c_z"
      case cSInputs = "c_s_inputs"
      case sigmaData = "sigma_data"
      case atomEncoder = "atom_encoder"
      case atomDecoder = "atom_decoder"
      case transformer
    }
  }

  public struct AtomStack: Codable, Sendable, Equatable {
    public let nBlocks: Int
    public let nHeads: Int

    private enum CodingKeys: String, CodingKey {
      case nBlocks = "n_blocks"
      case nHeads = "n_heads"
    }
  }

  public struct DistogramSection: Codable, Sendable, Equatable {
    public let cZ: Int
    public let noBins: Int

    private enum CodingKeys: String, CodingKey {
      case cZ = "c_z"
      case noBins = "no_bins"
    }
  }

  public struct InputEmbedderSection: Codable, Sendable, Equatable {
    public let cAtom: Int
    public let cAtompair: Int
    public let cToken: Int

    private enum CodingKeys: String, CodingKey {
      case cAtom = "c_atom"
      case cAtompair = "c_atompair"
      case cToken = "c_token"
    }
  }

  public struct RelativePositionSection: Codable, Sendable, Equatable {
    public let cZ: Int
    public let rMax: Int
    public let sMax: Int

    private enum CodingKeys: String, CodingKey {
      case cZ = "c_z"
      case rMax = "r_max"
      case sMax = "s_max"
    }
  }

  static func decode(from url: URL) throws -> ProtenixModelConfiguration {
    let data = try ArtifactIO.readData(url)
    let configuration: ProtenixModelConfiguration
    do {
      configuration = try JSONDecoder().decode(
        ProtenixModelConfiguration.self, from: data)
    } catch {
      throw ProtenixError.invalidJSON(
        file: url.lastPathComponent,
        reason: String(describing: error)
      )
    }
    guard configuration.schemaVersion == Self.supportedSchemaVersion else {
      throw ProtenixError.unsupportedSchema(
        found: configuration.schemaVersion,
        supported: Self.supportedSchemaVersion
      )
    }
    return configuration
  }
}
