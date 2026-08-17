import Foundation
import MLX

/// Writes folded coordinates to a PDB file, using the atom identities the feature
/// bundle carried through from the featurizer.
public enum StructureWriter {
  /// Render `[N_atom, 3]` coordinates and the bundle's atom metadata as PDB text.
  public static func pdb(coordinates: MLXArray, atoms: [FeatureBundle.Atom]) -> String {
    precondition(
      coordinates.shape == [atoms.count, 3],
      "coordinate count must match the atom metadata")
    let values = coordinates.asType(.float32).asArray(Float.self)

    var lines: [String] = []
    for (index, atom) in atoms.enumerated() {
      let x = values[index * 3 + 0]
      let y = values[index * 3 + 1]
      let z = values[index * 3 + 2]
      lines.append(atomRecord(serial: index + 1, atom: atom, x: x, y: y, z: z))
    }
    lines.append("END")
    return lines.joined(separator: "\n") + "\n"
  }

  private static func atomRecord(
    serial: Int, atom: FeatureBundle.Atom, x: Float, y: Float, z: Float
  ) -> String {
    // Column-aligned to the PDB ATOM record spec. Atom names occupy columns 13-16 with
    // their own indentation rule: names shorter than four characters start in column 14.
    let name = atom.atomName.count >= 4
      ? String(atom.atomName.prefix(4))
      : " \(atom.atomName)"
    let serialField = String(serial).leftPadded(to: 5)
    let nameField = name.rightPadded(to: 4)
    let resField = atom.resName.rightPadded(to: 3)
    let chain = atom.chainId.isEmpty ? "A" : String(atom.chainId.prefix(1))
    let resSeq = String(atom.resId).leftPadded(to: 4)
    let xField = coordinateField(x)
    let yField = coordinateField(y)
    let zField = coordinateField(z)
    let element = atom.element.uppercased().leftPadded(to: 2)
    return
      "ATOM  \(serialField) \(nameField) \(resField) \(chain)\(resSeq)    "
      + "\(xField)\(yField)\(zField)  1.00  0.00          \(element)"
  }

  private static func coordinateField(_ value: Float) -> String {
    String(format: "%8.3f", value)
  }
}

extension String {
  fileprivate func leftPadded(to width: Int) -> String {
    count >= width ? self : String(repeating: " ", count: width - count) + self
  }
  fileprivate func rightPadded(to width: Int) -> String {
    count >= width ? self : self + String(repeating: " ", count: width - count)
  }
}
