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
    .package(
      url: "https://github.com/ml-explore/mlx-swift.git",
      exact: "0.31.6"
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
      ]
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
