// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MathSolver",
    products: [.library(name: "MathSolver", targets: ["MathSolver"])],
    targets: [
        .target(name: "MathSolver"),
        .testTarget(name: "MathSolverTests", dependencies: ["MathSolver"])
    ]
)
