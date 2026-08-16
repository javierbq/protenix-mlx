import Foundation

/// Every way loading or running a Protenix artifact can fail.
///
/// Deliberately concrete: a pack is fetched over the network and unzipped on a user's
/// device, so "it didn't load" needs to distinguish a truncated download from a pack
/// built by a newer exporter from a genuine bug in this runtime.
public enum ProtenixError: Error, Equatable, Sendable {
  case missingFile(String)
  case invalidJSON(file: String, reason: String)
  case unsupportedSchema(found: Int, supported: Int)
  case wrongArtifactKind(expected: String, found: String)
  case tensorLoadFailure(file: String, reason: String)
  case tensorNameMismatch(missing: [String], unexpected: [String])
  case tensorShapeMismatch(name: String, declared: [Int], found: [Int])
  case tensorDTypeMismatch(name: String, declared: String, found: String)
  case missingTensor(String)
  case missingQuantizationSpec(String)
  case malformedQuantizedMatrix(name: String, reason: String)
  case unsupportedModel(String)
}

extension ProtenixError: LocalizedError {
  public var errorDescription: String? {
    switch self {
    case .missingFile(let path):
      return "artifact file is missing: \(path)"
    case .invalidJSON(let file, let reason):
      return "\(file) is not valid JSON: \(reason)"
    case .unsupportedSchema(let found, let supported):
      return
        "artifact uses schema version \(found) but this runtime supports \(supported); "
        + "the pack and the app are from different releases"
    case .wrongArtifactKind(let expected, let found):
      return "expected a \(expected) artifact but found a \(found) artifact"
    case .tensorLoadFailure(let file, let reason):
      return "could not read tensors from \(file): \(reason)"
    case .tensorNameMismatch(let missing, let unexpected):
      return
        "artifact contents do not match its manifest -- missing \(missing.count) "
        + "(\(missing.prefix(3).joined(separator: ", "))), unexpected "
        + "\(unexpected.count) (\(unexpected.prefix(3).joined(separator: ", ")))"
    case .tensorShapeMismatch(let name, let declared, let found):
      return "\(name) is declared \(declared) but stored as \(found)"
    case .tensorDTypeMismatch(let name, let declared, let found):
      return "\(name) is declared \(declared) but stored as \(found)"
    case .missingTensor(let name):
      return "the model needs a weight the artifact does not contain: \(name)"
    case .missingQuantizationSpec(let name):
      return "\(name) is packed as int8 but the manifest declares no quantization"
    case .malformedQuantizedMatrix(let name, let reason):
      return "quantized matrix \(name) is unusable: \(reason)"
    case .unsupportedModel(let name):
      return "this runtime does not implement the Protenix variant \(name)"
    }
  }
}
