import Foundation

/// Artifact category encoded by the offline Python exporter.
public enum ArtifactKind: String, Codable, Sendable {
  case model
  case features
  case fixture
}

/// Global affine quantization settings for a model artifact.
///
/// Absent on a dense pack. Its presence is exactly what tells the runtime whether a
/// matrix should be read as three arrays (weight/scales/biases) or one.
public struct QuantizationSpec: Codable, Sendable, Equatable {
  public let bits: Int
  public let groupSize: Int
  public let mode: String
}

/// Manifest declaration for one SafeTensors array.
public struct TensorSpec: Codable, Sendable, Equatable {
  public let name: String
  public let shape: [Int]
  public let dtype: String
  public let shard: String
  /// Set only on affine-int8 matrices, whose input width is zero-padded up to the
  /// quantization group size. `logicalShape` is the width the graph actually uses.
  public let logicalShape: [Int]?
  public let physicalShape: [Int]?
}

/// Versioned contract generated alongside every model or feature bundle.
public struct ArtifactManifest: Codable, Sendable, Equatable {
  public static let supportedSchemaVersion = 1

  public let schemaVersion: Int
  public let artifactKind: ArtifactKind
  /// Upstream model name this pack was exported from, e.g.
  /// `protenix_tiny_default_v0.5.0`. Carried because a Protenix checkpoint has no
  /// self-describing architecture: the same tensor names appear at four different
  /// sizes, so the name is how a runtime knows which graph to build.
  public let modelName: String
  public let sourceRevision: String
  public let sourceCommit: String
  public let sourceCheckpointSha256: String?
  public let tensors: [TensorSpec]
  public let quantization: QuantizationSpec?

  static func decode(from url: URL) throws -> ArtifactManifest {
    let data = try ArtifactIO.readData(url)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    do {
      return try decoder.decode(ArtifactManifest.self, from: data)
    } catch {
      throw ProtenixError.invalidJSON(
        file: url.lastPathComponent,
        reason: String(describing: error)
      )
    }
  }
}
