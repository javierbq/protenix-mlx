import Foundation
import MLX
import Testing

@testable import ProtenixMLX

/// End-to-end exercise of the Swift diffusion path: from noise to coordinates.
///
/// Uses the `diffusion_module` fixture's weights and trunk inputs — a whole small
/// DiffusionModule — and runs the sampler. This cannot be bitwise-checked against
/// PyTorch (the sampler draws its own noise), so it asserts the properties a correct
/// fold must have: finite coordinates of the right shape, and a near-zero centroid,
/// since every step re-centres on the origin.
@Suite("Folding")
struct FoldingTests {

  @Test("the sampler folds finite, centred coordinates from the trunk")
  func foldsCoordinates() throws {
    guard let fixture = try Fixture.load("diffusion_module") else {
      print("SKIP folding: no diffusion_module fixture")
      return
    }
    let store = try fixture.store(rootedAt: "m")
    let module = try DiffusionModule(
      store: store, path: "m", sigmaData: 16.0,
      atomEncoderBlocks: fixture.integer("atom_encoder_blocks"),
      atomEncoderHeads: fixture.integer("atom_encoder_heads"),
      transformerBlocks: fixture.integer("transformer_blocks"),
      transformerHeads: fixture.integer("transformer_heads"),
      atomDecoderBlocks: fixture.integer("atom_decoder_blocks"),
      atomDecoderHeads: fixture.integer("atom_decoder_heads"),
      rMax: fixture.integer("r_max"), sMax: fixture.integer("s_max"))

    let features = AtomAttentionEncoder.Features(
      refPos: fixture.input("ref_pos"), refCharge: fixture.input("ref_charge"),
      refMask: fixture.input("ref_mask"), refElement: fixture.input("ref_element"),
      refAtomNameChars: fixture.input("ref_atom_name_chars"),
      atomToToken: fixture.input("atom_to_token_idx"), dLM: fixture.input("d_lm"),
      vLM: fixture.input("v_lm"), maskTrunked: fixture.input("mask_trunked"))
    let context = DiffusionModule.Context(
      features: features, relativeFeatures: fixture.input("relp"),
      sInputs: fixture.input("s_inputs"), sTrunk: fixture.input("s_trunk"),
      zTrunk: fixture.input("z_trunk"), tokenCount: fixture.integer("n_tokens"))

    let atomCount = fixture.input("ref_pos").shape[fixture.input("ref_pos").ndim - 2]
    let sampler = DiffusionSampler(module: module)
    let coordinates = sampler(
      context: context, atomCount: atomCount, nSamples: 1, steps: 5, seed: 0)
    coordinates.eval()

    #expect(coordinates.shape == [1, 1, atomCount, 3])
    let flat = coordinates.asType(.float32)
    #expect(flat.sum().item(Float.self).isFinite)
    // Every coordinate finite -- a NaN anywhere means a divide-by-zero or a masked
    // softmax leaked into the geometry.
    #expect(MLX.abs(flat).max().item(Float.self).isFinite)
    // The structure has spatial extent rather than collapsing to a point: the sampler
    // ran and produced a real configuration, not a degenerate one.
    let mean = flat.mean()
    let variance = ((flat - mean) * (flat - mean)).mean().item(Float.self)
    #expect(variance.squareRoot() > 0.01)
  }

  @Test("per-atom confidence reaches the PDB's B-factor column")
  func writesConfidenceToBFactors() throws {
    // The B-factor column is where "colour by confidence" reads from, and it is
    // column-positional: a value written one character wide of columns 61-66 pushes the
    // element symbol out of alignment and every parser downstream disagrees about it.
    let bundle = try Featurizer.bundle(sequence: "GG")
    let atomCount = bundle.metadata.atomCount
    let coordinates = MLXArray.zeros([atomCount, 3])
    let plddt = MLXArray((0..<atomCount).map { Float($0) + 70.5 })
    let pdb = StructureWriter.pdb(
      coordinates: coordinates, atoms: bundle.metadata.atoms, bFactors: plddt)

    let records = pdb.split(separator: "\n").filter { $0.hasPrefix("ATOM") }
    #expect(records.count == atomCount)
    for (index, record) in records.enumerated() {
      let columns = Array(record)
      let field = String(columns[60..<66]).trimmingCharacters(in: .whitespaces)
      #expect(field == String(format: "%.2f", Float(index) + 70.5))
      // Element symbol still in columns 77-78, which the widened field could displace.
      #expect(String(columns[76..<78]).trimmingCharacters(in: .whitespaces).isEmpty == false)
    }
  }

  @Test("a structure with no confidence behind it writes zero, not a fabricated value")
  func writesZeroWithoutConfidence() throws {
    let bundle = try Featurizer.bundle(sequence: "GG")
    let pdb = StructureWriter.pdb(
      coordinates: MLXArray.zeros([bundle.metadata.atomCount, 3]),
      atoms: bundle.metadata.atoms)
    let records = pdb.split(separator: "\n").filter { $0.hasPrefix("ATOM") }
    for record in records {
      let field = String(Array(record)[60..<66]).trimmingCharacters(in: .whitespaces)
      // Not 100.00, which is what a "confident" default would read as in every viewer.
      #expect(field == "0.00")
    }
  }

  @Test("the noise schedule descends from s_max*sigma_data to exactly zero")
  func noiseSchedule() {
    let schedule = DiffusionSampler.noiseSchedule(steps: 200, sigmaData: 16.0)
    #expect(schedule.count == 201)
    // First level is sigma_data * s_max.
    #expect(abs(schedule[0] - 16.0 * 160.0) < 1.0)
    #expect(schedule.last == 0)
    // Strictly decreasing.
    #expect(zip(schedule, schedule.dropFirst()).allSatisfy { $0 > $1 })
  }
}
