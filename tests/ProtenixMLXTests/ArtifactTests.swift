import Foundation
import MLX
import Testing

@testable import ProtenixMLX

/// Builds artifact directories on disk so the loader is exercised through the same
/// path a downloaded pack takes, rather than through an in-memory shortcut.
private struct ArtifactBuilder {
  let directory: URL
  var tensors: [String: MLXArray] = [:]
  var specs: [TensorSpec] = []
  var quantization: QuantizationSpec?
  var modelName = "protenix_tiny_default_v0.5.0"

  init() throws {
    directory = FileManager.default.temporaryDirectory
      .appending(path: "protenix-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: directory, withIntermediateDirectories: true)
  }

  static func dtypeName(_ dtype: DType) -> String { ArtifactIO.dtypeName(dtype) }

  mutating func add(
    _ name: String,
    _ array: MLXArray,
    logicalShape: [Int]? = nil,
    physicalShape: [Int]? = nil
  ) {
    tensors[name] = array
    specs.append(
      TensorSpec(
        name: name,
        shape: array.shape,
        dtype: Self.dtypeName(array.dtype),
        shard: "model.safetensors",
        logicalShape: logicalShape,
        physicalShape: physicalShape
      )
    )
  }

  /// Quantize one matrix exactly as the Python exporter does, including the
  /// zero-padding of the contracted axis up to the group size.
  mutating func addQuantized(_ path: String, _ weight: MLXArray, groupSize: Int = 64) {
    let outputWidth = weight.shape[0]
    let logicalWidth = weight.shape[1]
    let physicalWidth = ((logicalWidth + groupSize - 1) / groupSize) * groupSize
    var padded = MLXArray.zeros([outputWidth, physicalWidth], dtype: .float16)
    padded[0..., 0..<logicalWidth] = weight.asType(.float16)
    let (packed, scales, biases) = MLX.quantized(
      padded, groupSize: groupSize, bits: 8, mode: .affine)
    add(
      "\(path).weight", packed,
      logicalShape: [outputWidth, logicalWidth],
      physicalShape: [outputWidth, physicalWidth])
    add("\(path).scales", scales)
    // `biases` is optional in mlx-swift because non-affine modes have no zero-point;
    // the affine mode used here always produces one.
    add("\(path).biases", biases!)
    quantization = QuantizationSpec(bits: 8, groupSize: groupSize, mode: "affine")
  }

  func write() throws -> URL {
    try MLX.save(arrays: tensors, url: directory.appending(path: "model.safetensors"))
    let manifest = ArtifactManifest(
      schemaVersion: 1,
      artifactKind: .model,
      modelName: modelName,
      sourceRevision: "main",
      sourceCommit: String(repeating: "a", count: 40),
      sourceCheckpointSha256: String(repeating: "b", count: 64),
      tensors: specs.sorted { $0.name < $1.name },
      quantization: quantization
    )
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    try encoder.encode(manifest)
      .write(to: directory.appending(path: "manifest.json"))
    try Self.configurationJSON(modelName: modelName)
      .write(to: directory.appending(path: "config.json"))
    return directory
  }

  /// A minimal config.json matching the tiny variant's real dimensions.
  static func configurationJSON(modelName: String) -> Data {
    let json = """
      {
        "schema_version": 1,
        "model_name": "\(modelName)",
        "source_revision": "main",
        "source_commit": "\(String(repeating: "a", count: 40))",
        "c_s": 384, "c_z": 128, "c_s_inputs": 449,
        "c_atom": 128, "c_atompair": 16, "c_token": 384,
        "n_cycle": 4, "n_diffusion_steps": 5,
        "parameter_count": 110649663, "quantized_matrix_count": 733,
        "graph_roots": ["pairformer_stack"],
        "model": {
          "pairformer": {"n_blocks": 8, "n_heads": 16, "c_s": 384, "c_z": 128,
                         "hidden_scale_up": false},
          "msa_module": {"n_blocks": 1, "c_m": 64, "c_z": 128, "c_s_inputs": 449,
                         "hidden_scale_up": false, "msa_max_size": 16384},
          "template_embedder": {"n_blocks": 0, "c": 64, "c_z": 128,
                                "hidden_scale_up": false},
          "confidence_head": {"n_blocks": 4, "c_s": 384, "c_z": 128,
                              "c_s_inputs": 449, "max_atoms_per_token": 24,
                              "distance_bin_start": 3.25, "distance_bin_end": 52.0,
                              "distance_bin_step": 1.25, "hidden_scale_up": false},
          "diffusion_module": {"c_atom": 128, "c_atompair": 16, "c_token": 768,
                               "c_s": 384, "c_z": 128, "c_s_inputs": 449,
                               "sigma_data": 16.0,
                               "atom_encoder": {"n_blocks": 1, "n_heads": 4},
                               "atom_decoder": {"n_blocks": 1, "n_heads": 4},
                               "transformer": {"n_blocks": 8, "n_heads": 16}},
          "distogram_head": {"c_z": 128, "no_bins": 64},
          "input_embedder": {"c_atom": 128, "c_atompair": 16, "c_token": 384},
          "relative_position_encoding": {"c_z": 128, "r_max": 32, "s_max": 2}
        }
      }
      """
    return Data(json.utf8)
  }
}

@Suite("Artifact loading")
struct ArtifactLoadingTests {

  @Test("loads a dense pack and its configuration")
  func loadsDensePack() throws {
    var builder = try ArtifactBuilder()
    builder.add("linear_no_bias_sinit.weight", MLXArray.zeros([384, 449], dtype: .float16))
    let artifact = try ProtenixArtifact.load(from: builder.write())
    #expect(artifact.isQuantized == false)
    #expect(artifact.configuration.cZ == 128)
    #expect(artifact.configuration.model.pairformer.nBlocks == 8)
  }

  @Test("a template stack of zero blocks reads as inert, not missing")
  func inertTemplateStack() throws {
    var builder = try ArtifactBuilder()
    builder.add("linear_no_bias_sinit.weight", MLXArray.zeros([384, 449], dtype: .float16))
    let artifact = try ProtenixArtifact.load(from: builder.write())
    #expect(artifact.configuration.model.templateEmbedder.isEnabled == false)
  }

  @Test("derives the confidence head's bin count from its bin edges")
  func confidenceBinCount() throws {
    var builder = try ArtifactBuilder()
    builder.add("linear_no_bias_sinit.weight", MLXArray.zeros([384, 449], dtype: .float16))
    let artifact = try ProtenixArtifact.load(from: builder.write())
    // arange(3.25, 52.0, 1.25) -> 39 bins, matching linear_no_bias_d's input width.
    #expect(artifact.configuration.model.confidenceHead.binCount == 39)
  }

  @Test("rejects a config that names a different model than the manifest")
  func rejectsInconsistentModelName() throws {
    var builder = try ArtifactBuilder()
    builder.add("a.weight", MLXArray.zeros([4, 4], dtype: .float16))
    builder.modelName = "protenix-v2"
    let directory = try builder.write()
    // Rewrite config.json to disagree with the manifest.
    try ArtifactBuilder.configurationJSON(modelName: "protenix_tiny_default_v0.5.0")
      .write(to: directory.appending(path: "config.json"))
    #expect(throws: ProtenixError.self) {
      try ProtenixArtifact.load(from: directory)
    }
  }

  @Test("a missing config.json is fatal, not tolerated")
  func requiresConfiguration() throws {
    var builder = try ArtifactBuilder()
    builder.add("a.weight", MLXArray.zeros([4, 4], dtype: .float16))
    let directory = try builder.write()
    try FileManager.default.removeItem(at: directory.appending(path: "config.json"))
    #expect(throws: ProtenixError.self) {
      try ProtenixArtifact.load(from: directory)
    }
  }

  @Test("detects a tensor the manifest does not declare")
  func detectsUndeclaredTensor() throws {
    var builder = try ArtifactBuilder()
    builder.add("a.weight", MLXArray.zeros([4, 4], dtype: .float16))
    let directory = try builder.write()
    // Add an array to the shard without declaring it -- the shape a mismatched
    // exporter/runtime pairing produces.
    var arrays = try MLX.loadArrays(url: directory.appending(path: "model.safetensors"))
    arrays["stowaway"] = MLXArray.zeros([2], dtype: .float16)
    try MLX.save(arrays: arrays, url: directory.appending(path: "model.safetensors"))
    #expect(throws: ProtenixError.self) {
      try ProtenixArtifact.load(from: directory)
    }
  }
}

@Suite("Weight store")
struct WeightStoreTests {

  private func store(_ builder: ArtifactBuilder) throws -> WeightStore {
    WeightStore(artifact: try ProtenixArtifact.load(from: try builder.write()))
  }

  @Test("int8 and dense packs of the same matrix agree")
  func int8MatchesDense() throws {
    // The regression that motivates this test: WeightStore once looked up
    // "<path>.weight.scales" instead of "<path>.scales", so every quantized matrix
    // fell through to the dense branch and its packed uint32 words were multiplied
    // as if they were weights -- no error, just wrong numbers.
    let weight = MLXRandom.normal([128, 384], key: MLXRandom.key(0)).asType(.float16)
    let input = MLXRandom.normal([2, 384], key: MLXRandom.key(1)).asType(.float16)

    var dense = try ArtifactBuilder()
    dense.add("probe.weight", weight)
    let denseOutput = try store(dense).linear("probe")(input)

    var quantized = try ArtifactBuilder()
    quantized.addQuantized("probe", weight)
    let quantizedStore = try store(quantized)
    let quantizedOutput = try quantizedStore.linear("probe")(input)

    #expect(quantizedStore.artifact.isQuantized)
    #expect(quantizedOutput.shape == denseOutput.shape)
    let difference = MLX.abs(quantizedOutput - denseOutput).max().item(Float.self)
    let scale = MLX.abs(denseOutput).max().item(Float.self)
    #expect(difference / scale < 0.05)
  }

  @Test("reports the logical input width, not the packed one")
  func reportsLogicalWidth() throws {
    // A bits=8 pack stores four values per uint32, so the stored matrix is a quarter
    // as wide. A runtime that read the stored width would build the wrong graph.
    var builder = try ArtifactBuilder()
    builder.addQuantized("probe", MLXArray.zeros([128, 384], dtype: .float16))
    let linear = try store(builder).linear("probe")
    #expect(linear.logicalInputWidth == 384)
    #expect(linear.weight.shape[1] == 96)
  }

  @Test("pads a matrix whose input width is not a multiple of the group size")
  func padsRaggedWidth() throws {
    let weight = MLXRandom.normal([32, 100], key: MLXRandom.key(2)).asType(.float16)
    let input = MLXRandom.normal([1, 100], key: MLXRandom.key(3)).asType(.float16)

    var dense = try ArtifactBuilder()
    dense.add("probe.weight", weight)
    let denseOutput = try store(dense).linear("probe")(input)

    var quantized = try ArtifactBuilder()
    quantized.addQuantized("probe", weight)
    let linear = try store(quantized).linear("probe")
    #expect(linear.logicalInputWidth == 100)
    #expect(linear.physicalInputWidth == 128)
    let output = linear(input)
    #expect(output.shape == [1, 32])
    let difference = MLX.abs(output - denseOutput).max().item(Float.self)
    let scale = MLX.abs(denseOutput).max().item(Float.self)
    #expect(difference / scale < 0.05)
  }

  @Test("attaches a bias when the artifact carries one")
  func attachesBias() throws {
    var builder = try ArtifactBuilder()
    builder.add("probe.weight", MLXArray.zeros([4, 8], dtype: .float16))
    builder.add("probe.bias", MLXArray.ones([4], dtype: .float16))
    let linear = try store(builder).linear("probe")
    let output = linear(MLXArray.zeros([1, 8], dtype: .float16))
    #expect(output.sum().item(Float.self) == 4.0)
  }

  @Test("does not mistake quantizer zero-points for a layer bias")
  func distinguishesBiasFromBiases() throws {
    // `.biases` is the quantizer's zero-point array; `.bias` is the layer's additive
    // term. Confusing them adds a [out, groups] array to a [out] one.
    var builder = try ArtifactBuilder()
    builder.addQuantized("probe", MLXArray.zeros([32, 128], dtype: .float16))
    let linear = try store(builder).linear("probe")
    #expect(linear.linearBias == nil)
  }

  @Test("refuses a packed matrix whose scales are missing")
  func refusesOrphanedPackedMatrix() throws {
    var builder = try ArtifactBuilder()
    builder.addQuantized("probe", MLXArray.zeros([32, 128], dtype: .float16))
    // Drop the scales but keep the quantization block: the artifact now claims a
    // dense read of a uint32 matrix, which must fail rather than produce numbers.
    builder.specs.removeAll { $0.name == "probe.scales" }
    builder.tensors.removeValue(forKey: "probe.scales")
    let storeWithoutScales = try store(builder)
    #expect(throws: ProtenixError.self) {
      try storeWithoutScales.linear("probe")
    }
  }

  @Test("names a weight the artifact lacks")
  func reportsMissingWeight() throws {
    var builder = try ArtifactBuilder()
    builder.add("probe.weight", MLXArray.zeros([4, 8], dtype: .float16))
    let weights = try store(builder)
    #expect(throws: ProtenixError.self) {
      try weights.linear("absent")
    }
  }
}
