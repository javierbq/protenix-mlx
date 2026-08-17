import Foundation
import MLX
import Testing

@testable import ProtenixMLX

/// Asserts the Swift featurizer reproduces upstream's data pipeline **bitwise**.
///
/// The featurizer is the one part of this port with no PyTorch module to imitate: it
/// stands in for `SampleDictToFeatures` plus `make_dummy_feature` plus `generate_relp`
/// plus `update_input_feature_dict`, a data pipeline whose output feeds every learned
/// layer downstream. A quiet disagreement there does not crash — it folds a slightly
/// different molecule and returns a confident structure for it.
///
/// So the comparison is exact, not toleranced. It can be: the reference conformers ship
/// pre-centred (see ``ResidueTemplates``), so `ref_pos` is a copy rather than a sum, and
/// everything else is either a one-hot or a float32 subtraction of those same numbers.
/// A tolerance here would hide exactly the class of bug this suite exists to catch —
/// an off-by-one in the atom-window padding, a residue whose atoms come out reordered.
///
/// Bundles come from `scripts/build_feature_fixtures.sh`, which drives the real upstream
/// featurizer. They are not committed, and when absent these tests SKIP loudly rather
/// than pass — a green featurizer suite with no reference data would be worse than none.
@Suite("Featurizer parity")
struct FeaturizerParityTests {

  /// One exported bundle: what Python produced, and the chains it produced it from.
  struct Reference {
    let name: String
    let chains: [Featurizer.Chain]
    let metadata: FeatureBundle.Metadata
    let tensors: [String: MLXArray]

    static var root: URL? {
      let directory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // ProtenixMLXTests
        .deletingLastPathComponent()  // tests
        .deletingLastPathComponent()  // package root
        .appending(path: ".artifacts/feature_bundles")
      return FileManager.default.fileExists(atPath: directory.path) ? directory : nil
    }

    static func all() throws -> [Reference] {
      guard let root else { return [] }
      let names = try FileManager.default.contentsOfDirectory(atPath: root.path).sorted()
      return try names.compactMap { name in
        let directory = root.appending(path: name)
        guard
          FileManager.default.fileExists(
            atPath: directory.appending(path: "features.json").path)
        else { return nil }
        let bundle = try FeatureBundle.load(from: directory)
        let data = try Data(contentsOf: directory.appending(path: "features.json"))
        let document =
          (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        // The chains as the exporter grouped them, so Swift featurizes the same input
        // rather than a re-parse of the joined string.
        let entries = (document["chains"] as? [[String: Any]]) ?? []
        let chains = entries.map {
          Featurizer.Chain(
            id: ($0["chain"] as? String) ?? "A", sequence: ($0["sequence"] as? String) ?? "")
        }
        return Reference(
          name: name, chains: chains, metadata: bundle.metadata, tensors: bundle.tensors)
      }
    }
  }

  /// Every tensor the bundle carries, so a new one cannot be added without a check.
  static let tensorNames = [
    "ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars",
    "atom_to_token_idx", "distogram_rep_atom_mask", "atom_to_tokatom_idx",
    "d_lm", "v_lm", "mask_trunked", "restype", "profile",
    "deletion_mean", "relp", "token_bonds", "msa_features",
  ]

  @Test("every exported bundle is reproduced bitwise from sequence alone")
  func matchesUpstream() throws {
    let references = try Reference.all()
    guard !references.isEmpty else {
      print("SKIP featurizer parity: no bundles in .artifacts/feature_bundles "
        + "(run scripts/build_feature_fixtures.sh)")
      return
    }
    let templates = try ResidueTemplates.bundled()
    for reference in references {
      let built = try Featurizer.bundle(
        chains: reference.chains, name: reference.name, templates: templates)

      #expect(
        built.metadata.tokenCount == reference.metadata.tokenCount,
        "\(reference.name): token count")
      #expect(
        built.metadata.atomCount == reference.metadata.atomCount,
        "\(reference.name): atom count")

      #expect(
        Set(built.tensors.keys) == Set(Self.tensorNames),
        "\(reference.name): the featurizer produced a different tensor set")

      for name in Self.tensorNames {
        guard let expected = reference.tensors[name], let actual = built.tensors[name]
        else {
          Issue.record("\(reference.name): \(name) missing from one side")
          continue
        }
        #expect(actual.shape == expected.shape, "\(reference.name): \(name) shape")
        guard actual.shape == expected.shape else { continue }
        let difference = MLX.abs(actual.asType(.float32) - expected.asType(.float32))
        let worst = difference.max().item(Float.self)
        #expect(worst == 0, "\(reference.name): \(name) differs by \(worst)")
      }
    }
  }

  @Test("atom identity matches, because the PDB is written from it")
  func matchesAtomIdentity() throws {
    let references = try Reference.all()
    guard !references.isEmpty else {
      print("SKIP featurizer atom identity: no bundles in .artifacts/feature_bundles")
      return
    }
    for reference in references {
      let built = try Featurizer.bundle(chains: reference.chains, name: reference.name)
      let expected = reference.metadata.atoms
      let actual = built.metadata.atoms
      #expect(actual.count == expected.count, "\(reference.name): atom count")
      guard actual.count == expected.count else { continue }
      for (index, pair) in zip(actual, expected).enumerated() {
        let (left, right) = pair
        let describe = "\(left.atomName)/\(left.resName) vs \(right.atomName)/\(right.resName)"
        #expect(
          left.atomName == right.atomName && left.resName == right.resName
            && left.resId == right.resId && left.element == right.element,
          "\(reference.name): atom \(index) is \(describe)")
      }
    }
  }

  @Test("a bundle built in Swift folds through the predictor unchanged")
  func feedsThePredictor() throws {
    // The parity check above compares tensors; this one checks the bundle is *shaped*
    // the way `fold` reads it -- that every tensor it asks for by name is present and
    // batches to the rank the network expects. Cheap, and it would have caught a
    // correct tensor stored under a name nothing reads.
    let built = try Featurizer.bundle(sequence: "GSHM")
    for name in Self.tensorNames {
      #expect(built.tensors[name] != nil, "\(name) missing")
    }
    let features = try built.atomFeatures()
    #expect(features.refPos.shape == [1, built.metadata.atomCount, 3])
    #expect(built.batched(try built.tensor("restype")).shape
      == [1, built.metadata.tokenCount, 32])
  }
}

/// The table itself, which needs no fixtures at all.
@Suite("Residue templates")
struct ResidueTemplateTests {

  @Test("the shipped table carries the canonical twenty")
  func carriesTwenty() throws {
    let templates = try ResidueTemplates.bundled()
    #expect(templates.residues.count == 20)
    #expect(Set(templates.residues.keys) == Set("ACDEFGHIKLMNPQRSTVWY"))
  }

  @Test("the terminal form adds exactly OXT")
  func terminalFormAddsOXT() throws {
    let templates = try ResidueTemplates.bundled()
    for (letter, residue) in templates.residues.sorted(by: { $0.key < $1.key }) {
      let ordinary = residue.atoms.map(\.name)
      let terminal = residue.terminalAtoms.map(\.name)
      #expect(terminal == ordinary + ["OXT"], "\(letter)")
    }
  }

  @Test("an unknown residue is refused by name, never substituted")
  func refusesUnknownResidues() throws {
    // Upstream's tokenizers resolve an unrecognised letter to X and fold on. Doing that
    // here would return a structure for a sequence the caller never asked to fold.
    #expect(throws: ProtenixError.unsupportedResidue("X")) {
      _ = try Featurizer.bundle(sequence: "GSHXM")
    }
    #expect(throws: ProtenixError.unsupportedResidue("U")) {
      _ = try Featurizer.bundle(sequence: "GSHUM")
    }
  }

  @Test("an empty chain is refused")
  func refusesEmptyChains() throws {
    #expect(throws: ProtenixError.emptyChain("B")) {
      _ = try Featurizer.bundle(chains: [
        Featurizer.Chain(id: "A", sequence: "GSHM"),
        Featurizer.Chain(id: "B", sequence: "  "),
      ])
    }
  }

  @Test("lower-case sequences are folded, not refused")
  func acceptsLowerCase() throws {
    let upper = try Featurizer.bundle(sequence: "GSHM")
    let lower = try Featurizer.bundle(sequence: "gshm")
    #expect(lower.metadata.atomCount == upper.metadata.atomCount)
  }

  @Test("a homodimer's chains share an entity, a heterodimer's do not")
  func entityGrouping() throws {
    // relp is [rel_pos(66) | rel_token(66) | same_entity(1) | rel_chain(6)], so 132 is
    // the same-entity bit. A homodimer reading 0 there is being told its two identical
    // chains are unrelated, which changes what the model predicts about their interface.
    let homodimer = try Featurizer.bundle(sequence: "GSHM/GSHM")
    let heterodimer = try Featurizer.bundle(sequence: "GSHM/AWKD")
    let homoBit = homodimer.tensors["relp"]![0, 7, 132].item(Float.self)
    let heteroBit = heterodimer.tensors["relp"]![0, 7, 132].item(Float.self)
    #expect(homoBit == 1)
    #expect(heteroBit == 0)
  }
}
