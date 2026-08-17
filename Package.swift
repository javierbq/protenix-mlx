// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "ProtenixMLX",
  platforms: [
    .macOS(.v14),
    .iOS(.v17),
  ],
  products: [
    .library(name: "ProtenixMLX", targets: ["ProtenixMLX"]),
    .executable(name: "ProtenixMLXCLI", targets: ["ProtenixMLXCLI"]),
  ],
  dependencies: [
    // A range, not `exact:`. Two packages pinning the same dependency exactly is a
    // deadlock for anyone depending on both — RayMol consumes this alongside boltz-mlx
    // (itself pinned exactly at 0.31.6) and could never move mlx-swift while either
    // held. 0.31.6 is the floor because the quantized-matmul rename landed there.
    .package(
      url: "https://github.com/ml-explore/mlx-swift.git",
      "0.31.6"..<"0.32.0"
    ),
    .package(
      url: "https://github.com/apple/swift-argument-parser.git",
      exact: "1.8.2"
    ),
  ],
  targets: [
    .target(
      name: "ProtenixMLX",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "MLXLinalg", package: "mlx-swift"),
      ],
      // The canonical-20 reference conformers. ~90 KB of JSON that removes torch, rdkit
      // and a 624 MB components.cif from the machine doing the fold; see
      // Features/ResidueTemplates.swift.
      resources: [.process("Resources")]
    ),
    .executableTarget(
      name: "ProtenixMLXCLI",
      dependencies: [
        "ProtenixMLX",
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]
    ),
    .testTarget(
      name: "ProtenixMLXTests",
      dependencies: [
        "ProtenixMLX",
        .product(name: "MLX", package: "mlx-swift"),
      ],
      path: "tests/ProtenixMLXTests"
    ),
  ]
)
