// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "StudioCam",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "StudioCam", targets: ["StudioCam"])
    ],
    targets: [
        .executableTarget(
            name: "StudioCam",
            path: "Sources/StudioCam",
            exclude: ["Shaders/StudioCamShaders.metal"],
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("SwiftUI"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("CoreMediaIO"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("IOSurface"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("Vision"),
                .linkedFramework("VideoToolbox"),
                .linkedFramework("UniformTypeIdentifiers")
            ]
        )
    ]
)
