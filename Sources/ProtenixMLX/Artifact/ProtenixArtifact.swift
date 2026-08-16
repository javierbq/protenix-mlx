import Foundation
import MLX

/// A validated model artifact and every array its manifest declares.
public struct ProtenixArtifact {
  public let manifest: ArtifactManifest
  /// Required, unlike boltz-mlx's optional configuration: Protenix architecture is
  /// unrecoverable from weights alone (see `ProtenixModelConfiguration`).
  public let configuration: ProtenixModelConfiguration
  public let arrays: [String: MLXArray]

  public var isQuantized: Bool { manifest.quantization != nil }

  public static func load(from directory: URL) throws -> ProtenixArtifact {
    let loaded = try ArtifactIO.load(from: directory, expectedKind: .model)
    let configurationURL = directory.appending(path: "config.json")
    guard FileManager.default.fileExists(atPath: configurationURL.path) else {
      throw ProtenixError.missingFile(configurationURL.path)
    }
    let configuration = try ProtenixModelConfiguration.decode(from: configurationURL)
    guard configuration.modelName == loaded.manifest.modelName else {
      throw ProtenixError.invalidJSON(
        file: "config.json",
        reason:
          "config names model \(configuration.modelName) but the manifest names "
          + "\(loaded.manifest.modelName); the artifact is inconsistent"
      )
    }
    return ProtenixArtifact(
      manifest: loaded.manifest,
      configuration: configuration,
      arrays: loaded.arrays
    )
  }

  /// Fetch one declared array, or say which weight is missing.
  public func array(_ name: String) throws -> MLXArray {
    guard let value = arrays[name] else { throw ProtenixError.missingTensor(name) }
    return value
  }
}

enum ArtifactIO {
  struct LoadedArtifact {
    let manifest: ArtifactManifest
    let arrays: [String: MLXArray]
  }

  static func readData(_ url: URL) throws -> Data {
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw ProtenixError.missingFile(url.path)
    }
    do {
      return try Data(contentsOf: url)
    } catch {
      throw ProtenixError.tensorLoadFailure(
        file: url.lastPathComponent,
        reason: String(describing: error)
      )
    }
  }

  /// Manifest spelling for an MLX dtype, matching the exporter's `_TORCH_DTYPE_NAMES`.
  static func dtypeName(_ dtype: DType) -> String {
    switch dtype {
    case .bfloat16: return "bfloat16"
    case .float16: return "float16"
    case .float32: return "float32"
    case .int8: return "int8"
    case .int16: return "int16"
    case .int32: return "int32"
    case .int64: return "int64"
    case .uint8: return "uint8"
    case .uint32: return "uint32"
    case .bool: return "bool"
    default: return String(describing: dtype)
    }
  }

  static func load(from directory: URL, expectedKind: ArtifactKind) throws
    -> LoadedArtifact
  {
    let manifest = try ArtifactManifest.decode(
      from: directory.appending(path: "manifest.json")
    )
    guard manifest.schemaVersion == ArtifactManifest.supportedSchemaVersion else {
      throw ProtenixError.unsupportedSchema(
        found: manifest.schemaVersion,
        supported: ArtifactManifest.supportedSchemaVersion
      )
    }
    guard manifest.artifactKind == expectedKind else {
      throw ProtenixError.wrongArtifactKind(
        expected: expectedKind.rawValue,
        found: manifest.artifactKind.rawValue
      )
    }

    var arrays: [String: MLXArray] = [:]
    let specsByShard = Dictionary(grouping: manifest.tensors, by: \.shard)
    for shard in specsByShard.keys.sorted() {
      let shardURL = directory.appending(path: shard)
      guard FileManager.default.fileExists(atPath: shardURL.path) else {
        throw ProtenixError.missingFile(shardURL.path)
      }
      let loaded: [String: MLXArray]
      do {
        loaded = try MLX.loadArrays(url: shardURL)
      } catch {
        throw ProtenixError.tensorLoadFailure(
          file: shard,
          reason: String(describing: error)
        )
      }
      // Checked as sets before anything is used: a truncated download or a pack
      // built by a different exporter shows up here as one clear error rather than
      // as a missing-weight crash somewhere deep in the graph.
      let expectedNames = Set(specsByShard[shard, default: []].map(\.name))
      let foundNames = Set(loaded.keys)
      guard expectedNames == foundNames else {
        throw ProtenixError.tensorNameMismatch(
          missing: expectedNames.subtracting(foundNames).sorted(),
          unexpected: foundNames.subtracting(expectedNames).sorted()
        )
      }
      arrays.merge(loaded) { current, _ in current }
    }

    for spec in manifest.tensors {
      guard let array = arrays[spec.name] else {
        throw ProtenixError.missingTensor(spec.name)
      }
      guard array.shape == spec.shape else {
        throw ProtenixError.tensorShapeMismatch(
          name: spec.name, declared: spec.shape, found: array.shape
        )
      }
      let found = dtypeName(array.dtype)
      guard found == spec.dtype else {
        throw ProtenixError.tensorDTypeMismatch(
          name: spec.name, declared: spec.dtype, found: found
        )
      }
    }

    return LoadedArtifact(manifest: manifest, arrays: arrays)
  }
}
