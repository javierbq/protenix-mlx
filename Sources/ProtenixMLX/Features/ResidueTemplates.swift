import Foundation

/// The canonical-20 reference conformers, frozen out of the CCD at export time.
///
/// Everything upstream's featurizer looks up in a 624 MB `components.cif` for a *standard*
/// amino acid is a constant: the atom set, its order, elements, formal charges and the
/// reference conformer's coordinates. Only ligands and modified residues genuinely need a
/// chemical component dictionary at fold time, and this port folds neither. So the table
/// ships with the package — which is what lets a fold run with no Python, no rdkit and no
/// CCD on the device doing it.
///
/// Written by `protenix-mlx export-residue-templates`, which *derives* it by running
/// upstream's own featurizer and slicing per residue rather than reimplementing the
/// lookup, and which refuses to write a table whose conformers turn out to depend on
/// their neighbours.
///
/// Coordinates arrive already centred, exactly as the featurizer emitted them, so
/// building `ref_pos` is a copy rather than a sum — the reason Swift and Python agree on
/// it bit for bit instead of within a tolerance.
public struct ResidueTemplates: Sendable {
  /// One atom of a reference conformer.
  public struct Atom: Codable, Sendable {
    public let name: String
    public let element: String
    /// Index into the 128-wide `ref_element` one-hot: atomic number - 1.
    public let elementIndex: Int
    public let charge: Float
    /// Centred reference-conformer coordinates.
    public let pos: [Float]

    private enum CodingKeys: String, CodingKey {
      case name, element, charge, pos
      case elementIndex = "element_index"
    }
  }

  /// A canonical residue in both the forms a chain can contain.
  public struct Residue: Codable, Sendable {
    public let oneLetter: String
    /// The three-letter code, which is what reaches the written PDB.
    public let code: String
    /// Index into the 32-wide `restype` one-hot.
    public let restypeIndex: Int
    public let atoms: [Atom]
    /// The C-terminal form: the same conformer plus `OXT`, re-centred over all of it.
    ///
    /// Position changes a residue in exactly this one way. The N-terminus is not
    /// special — verified when the table is written, not assumed here.
    public let terminalAtoms: [Atom]

    private enum CodingKeys: String, CodingKey {
      case code, atoms
      case oneLetter = "one_letter"
      case restypeIndex = "restype_index"
      case terminalAtoms = "terminal_atoms"
    }
  }

  struct Document: Codable {
    let schemaVersion: Int
    let kind: String
    let restypeClasses: Int
    let elementClasses: Int
    let atomNameLength: Int
    let atomNameClasses: Int
    let upstreamCommit: String
    let residues: [Residue]

    private enum CodingKeys: String, CodingKey {
      case kind, residues
      case schemaVersion = "schema_version"
      case restypeClasses = "restype_classes"
      case elementClasses = "element_classes"
      case atomNameLength = "atom_name_length"
      case atomNameClasses = "atom_name_classes"
      case upstreamCommit = "upstream_commit"
    }
  }

  /// The schema this build understands. A table from the future is refused, not guessed
  /// at: every field here feeds a tensor the network reads positionally.
  public static let schemaVersion = 1

  /// `restype` is 32-wide: 20 amino acids + unknown, 4 RNA + unknown, 4 DNA + unknown, gap.
  public static let restypeClasses = 32
  /// `ref_element` is one-hot over atomic number, up to 128 (AF3 SI Table 5).
  public static let elementClasses = 128
  /// Atom names are padded to 4 characters, each encoded `ord(c) - 32` into 64 classes.
  public static let atomNameLength = 4
  public static let atomNameClasses = 64

  /// Residues by one-letter code.
  public let residues: [Character: Residue]
  /// The Protenix commit whose CCD these conformers came from.
  public let upstreamCommit: String

  init(document: Document) throws {
    guard document.schemaVersion == Self.schemaVersion else {
      throw ProtenixError.invalidJSON(
        file: "residue_templates.json",
        reason: """
          schema version \(document.schemaVersion) is not \(Self.schemaVersion); \
          this build cannot read that table
          """)
    }
    guard document.restypeClasses == Self.restypeClasses,
      document.elementClasses == Self.elementClasses,
      document.atomNameLength == Self.atomNameLength,
      document.atomNameClasses == Self.atomNameClasses
    else {
      throw ProtenixError.invalidJSON(
        file: "residue_templates.json",
        reason: "the table's encoding widths disagree with this build's")
    }
    var byLetter: [Character: Residue] = [:]
    for residue in document.residues {
      guard let letter = residue.oneLetter.first, residue.oneLetter.count == 1 else {
        throw ProtenixError.invalidJSON(
          file: "residue_templates.json",
          reason: "residue code \(residue.oneLetter.debugDescription) is not one letter")
      }
      byLetter[letter] = residue
    }
    residues = byLetter
    upstreamCommit = document.upstreamCommit
  }

  public static func load(from url: URL) throws -> ResidueTemplates {
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw ProtenixError.missingFile(url.path)
    }
    let data = try Data(contentsOf: url)
    let document: Document
    do {
      document = try JSONDecoder().decode(Document.self, from: data)
    } catch {
      throw ProtenixError.invalidJSON(
        file: url.lastPathComponent, reason: String(describing: error))
    }
    return try ResidueTemplates(document: document)
  }

  /// The table shipped inside this package.
  ///
  /// Loaded once and cached: it is ~90 KB of JSON, and a fold that re-parsed it per call
  /// would pay for it on every one.
  public static func bundled() throws -> ResidueTemplates {
    if let cached = cache.value { return cached }
    guard let url = Bundle.module.url(forResource: "residue_templates", withExtension: "json")
    else {
      throw ProtenixError.missingFile("residue_templates.json (in the package bundle)")
    }
    let loaded = try load(from: url)
    cache.value = loaded
    return loaded
  }

  private static let cache = Cache()

  private final class Cache: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: ResidueTemplates?
    var value: ResidueTemplates? {
      get { lock.withLock { stored } }
      set { lock.withLock { stored = newValue } }
    }
  }

  /// Look one residue up, naming the letter if it is not in the table.
  public func residue(for letter: Character) throws -> Residue {
    guard let residue = residues[letter] else {
      throw ProtenixError.unsupportedResidue(String(letter))
    }
    return residue
  }
}
