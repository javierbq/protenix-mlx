import ArgumentParser
import Foundation
import MLX
import ProtenixMLX

@main
struct ProtenixMLXCommand: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "ProtenixMLXCLI",
    abstract: "Inspect and run Protenix MLX artifacts.",
    subcommands: [Inspect.self, Predict.self]
  )
}

/// Fold a sequence: model artifact + sequence (or feature bundle) -> a PDB structure.
struct Predict: ParsableCommand {
  static let configuration = CommandConfiguration(
    abstract: "Fold a sequence into coordinates and write a PDB.",
    discussion: """
      Give either --sequence, which featurizes here in Swift and needs nothing but the \
      model pack, or --features, a bundle from `protenix-mlx export-features`. The two \
      produce identical tensors for a canonical-20 protein; the bundle path exists for \
      inputs the Swift featurizer does not carry reference conformers for.
      """)

  @Option(name: .long, help: "An exported model artifact directory.")
  var model: String

  @Option(
    name: .long,
    help: "Protein sequence to fold. Separate the chains of a complex with \"/\".")
  var sequence: String?

  @Option(name: .long, help: "A feature bundle from `export-features`.")
  var features: String?

  @Option(name: .long, help: "Where to write the PDB.")
  var output: String

  @Option(name: .long, help: "Diffusion steps; defaults to the artifact's own.")
  var diffusionSteps: Int?

  @Option(name: .long, help: "RNG seed.")
  var seed: Int = 0

  @Flag(
    name: .long,
    help: "Run the confidence head and write per-atom pLDDT into the B-factor column.")
  var confidence: Bool = false

  func validate() throws {
    if (sequence == nil) == (features == nil) {
      throw ValidationError("give exactly one of --sequence or --features")
    }
  }

  func run() throws {
    let artifact = try ProtenixArtifact.load(
      from: URL(fileURLWithPath: model))
    let bundle =
      if let sequence {
        try Featurizer.bundle(sequence: sequence)
      } else {
        try FeatureBundle.load(from: URL(fileURLWithPath: features!))
      }
    let predictor = try ProtenixPredictor(artifact: artifact)

    print("model        \(artifact.configuration.modelName)")
    print("sequence     \(bundle.metadata.sequence)")
    print("tokens/atoms \(bundle.metadata.tokenCount) / \(bundle.metadata.atomCount)")
    print("recycling    \(predictor.recyclingSteps)")
    print(
      "diffusion    \(diffusionSteps ?? predictor.diffusionSteps) steps")

    var coordinates: MLXArray
    var scores: ConfidenceHead.Scores?
    if confidence {
      let prediction = try predictor.foldScored(
        bundle: bundle, seed: UInt64(seed), diffusionSteps: diffusionSteps)
      coordinates = prediction.coordinates
      scores = prediction.scores
    } else {
      coordinates = try predictor.fold(
        bundle: bundle, seed: UInt64(seed), diffusionSteps: diffusionSteps)
      scores = nil
    }

    if let scores {
      print(String(format: "mean pLDDT   %.1f", scores.meanPLDDT))
    }
    let pdb = StructureWriter.pdb(
      coordinates: coordinates, atoms: bundle.metadata.atoms,
      bFactors: scores?.plddt)
    try pdb.write(
      to: URL(fileURLWithPath: output), atomically: true, encoding: .utf8)
    print("\nwrote \(bundle.metadata.atomCount) atoms to \(output)")
  }
}

/// Load a pack, validate it against its own manifest, and report what it holds.
///
/// This is the runtime's half of the export contract: the exporter's `verify_pack.py`
/// proves the numbers survive packing, and this proves the pack is loadable and
/// self-consistent on the device that has to run it.
struct Inspect: ParsableCommand {
  static let configuration = CommandConfiguration(
    abstract: "Validate an artifact directory and summarize its contents."
  )

  @Option(name: .long, help: "Path to an exported artifact directory.")
  var model: String

  @Flag(name: .long, help: "List every tensor the manifest declares.")
  var listTensors = false

  func run() throws {
    let directory = URL(fileURLWithPath: model)
    let artifact = try ProtenixArtifact.load(from: directory)
    let store = WeightStore(artifact: artifact)
    let configuration = artifact.configuration
    let manifest = artifact.manifest

    print("model        \(configuration.modelName)")
    print("schema       \(manifest.schemaVersion)")
    print("source       \(manifest.sourceRevision) @ \(manifest.sourceCommit.prefix(12))")
    if let quantization = manifest.quantization {
      print(
        "packing      int8 (\(quantization.mode), group \(quantization.groupSize), "
          + "\(quantization.bits) bits), \(configuration.quantizedMatrixCount) matrices")
    } else {
      let widths = Set(manifest.tensors.map(\.dtype)).sorted()
      print("packing      dense (\(widths.joined(separator: ", ")))")
    }
    print("parameters   \(String(format: "%.2f", Double(configuration.parameterCount) / 1e6))M")
    print("arrays       \(manifest.tensors.count)")
    print("")
    print("widths       c_s=\(configuration.cS) c_z=\(configuration.cZ) "
      + "c_s_inputs=\(configuration.cSInputs) c_token=\(configuration.cToken) "
      + "c_atom=\(configuration.cAtom) c_atompair=\(configuration.cAtompair)")
    let model = configuration.model
    print("pairformer   \(model.pairformer.nBlocks) blocks, \(model.pairformer.nHeads) heads"
      + (model.pairformer.hiddenScaleUp ? ", hidden scale-up" : ""))
    print("msa          \(model.msaModule.nBlocks) blocks, c_m=\(model.msaModule.cM)")
    print("template     "
      + (model.templateEmbedder.isEnabled
        ? "\(model.templateEmbedder.nBlocks) blocks, c=\(model.templateEmbedder.c)"
        : "inert (0 blocks; projections present, no stack)"))
    print("diffusion    \(model.diffusionModule.transformer.nBlocks) transformer blocks, "
      + "encoder \(model.diffusionModule.atomEncoder.nBlocks), "
      + "decoder \(model.diffusionModule.atomDecoder.nBlocks)")
    print("confidence   \(model.confidenceHead.nBlocks) blocks, "
      + "\(model.confidenceHead.binCount) distance bins")
    print("distogram    \(model.distogramHead.noBins) bins")
    print("defaults     \(configuration.nCycle) recycles, "
      + "\(configuration.nDiffusionSteps) diffusion steps")

    // Prove the weights are not merely present but usable: build one matrix from
    // each major stack and run it. A pack whose scales are misaligned loads fine and
    // only fails here.
    print("")
    try probe(store: store, label: "pairformer block 0",
              path: "pairformer_stack.blocks.0.attention_pair_bias.attention.linear_q")
    try probe(store: store, label: "top-level s init", path: "linear_no_bias_sinit")
    try probe(store: store, label: "distogram head", path: "distogram_head.linear")
    try probe(store: store, label: "diffusion block 0",
              path: "diffusion_module.diffusion_transformer.blocks.0.attention_pair_bias"
                + ".attention.linear_q")
    try probe(store: store, label: "confidence block 0",
              path: "confidence_head.pairformer_stack.blocks.0.attention_pair_bias"
                + ".attention.linear_q")

    if listTensors {
      print("")
      for spec in manifest.tensors.sorted(by: { $0.name < $1.name }) {
        let padding = spec.logicalShape.map { " logical \($0)" } ?? ""
        print("  \(spec.name)  \(spec.shape) \(spec.dtype)\(padding)")
      }
    }
    print("\nOK: artifact is loadable and self-consistent")
  }

  private func probe(store: WeightStore, label: String, path: String) throws {
    guard store.names(withPrefix: "\(path).weight").isEmpty == false else {
      print("probe        \(label): absent from this variant")
      return
    }
    let linear = try store.linear(path)
    let input = MLXArray.zeros([1, linear.logicalInputWidth], dtype: .float16)
    let output = linear(input)
    output.eval()
    print("probe        \(label): \(linear.logicalInputWidth) -> \(output.shape[1]) ok")
  }
}
