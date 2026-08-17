import Foundation
import MLX

/// A feature bundle exported by `protenix-mlx export-features`.
///
/// Everything the trunk and diffusion module need for one sequence, precomputed in
/// Python and stored as a flat SafeTensors file plus a JSON sidecar carrying the atom
/// identities the structure writer needs.
public struct FeatureBundle {
  public struct Atom: Codable, Sendable {
    public let element: String
    public let atomName: String
    public let resName: String
    public let resId: Int
    public let chainId: String

    private enum CodingKeys: String, CodingKey {
      case element, resName = "res_name", resId = "res_id", chainId = "chain_id"
      case atomName = "atom_name"
    }
  }

  public struct Metadata: Codable, Sendable {
    public let schemaVersion: Int
    public let name: String
    public let sequence: String
    public let tokenCount: Int
    public let atomCount: Int
    public let atoms: [Atom]

    private enum CodingKeys: String, CodingKey {
      case schemaVersion = "schema_version"
      case name, sequence, atoms
      case tokenCount = "token_count"
      case atomCount = "atom_count"
    }
  }

  public let metadata: Metadata
  public let tensors: [String: MLXArray]

  public static func load(from directory: URL) throws -> FeatureBundle {
    let metadataURL = directory.appending(path: "features.json")
    guard FileManager.default.fileExists(atPath: metadataURL.path) else {
      throw ProtenixError.missingFile(metadataURL.path)
    }
    let data = try Data(contentsOf: metadataURL)
    let metadata: Metadata
    do {
      metadata = try JSONDecoder().decode(Metadata.self, from: data)
    } catch {
      throw ProtenixError.invalidJSON(
        file: "features.json", reason: String(describing: error))
    }
    let tensorURL = directory.appending(path: "features.safetensors")
    guard FileManager.default.fileExists(atPath: tensorURL.path) else {
      throw ProtenixError.missingFile(tensorURL.path)
    }
    let tensors = try MLX.loadArrays(url: tensorURL)
    return FeatureBundle(metadata: metadata, tensors: tensors)
  }

  public func tensor(_ name: String) throws -> MLXArray {
    guard let value = tensors[name] else { throw ProtenixError.missingTensor(name) }
    // Everything travels with a leading batch axis of 1, which the network broadcasts
    // over; the per-token/atom features are exported without it, so add it here.
    return value
  }

  /// The atom-attention features, batched.
  public func atomFeatures() throws -> AtomAttentionEncoder.Features {
    AtomAttentionEncoder.Features(
      refPos: try batched(tensor("ref_pos")),
      refCharge: try batched(tensor("ref_charge")),
      refMask: try batched(tensor("ref_mask")),
      refElement: try batched(tensor("ref_element")),
      refAtomNameChars: try batched(tensor("ref_atom_name_chars")),
      atomToToken: try tensor("atom_to_token_idx"),
      dLM: try batched(tensor("d_lm")),
      vLM: try batched(tensor("v_lm")),
      maskTrunked: try tensor("mask_trunked"))
  }

  /// Add a leading batch axis of 1.
  public func batched(_ x: MLXArray) -> MLXArray {
    x.expandedDimensions(axis: 0)
  }
}
