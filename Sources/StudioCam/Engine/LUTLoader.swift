import Foundation
import Metal
import simd

public struct Loaded3DLUT: Sendable {
    public let title: String
    public let url: URL
    public let dimension: UInt32
    public let domainMin: SIMD4<Float>
    public let domainMax: SIMD4<Float>
    public nonisolated(unsafe) let texture: MTLTexture
}

public enum LUTLoader {
    public static let supportedExtensions: Set<String> = ["cube"]

    public static func isLUTFile(_ url: URL) -> Bool {
        supportedExtensions.contains(url.pathExtension.lowercased())
    }

    /// Parses an industry-standard Adobe / DaVinci Resolve `.cube` 3D or 1D LUT file
    /// and uploads it into a hardware-interpolated `.rgba16Float` 3D Metal texture.
    public static func loadCubeLUT(from url: URL, device: MTLDevice) throws -> Loaded3DLUT {
        let content = try String(contentsOf: url, encoding: .utf8)
        var title = url.deletingPathExtension().lastPathComponent
        var size3D: Int = 0
        var size1D: Int = 0
        var domainMin = SIMD4<Float>(0, 0, 0, 0)
        var domainMax = SIMD4<Float>(1, 1, 1, 1)

        var floats: [Float] = []
        floats.reserveCapacity(33 * 33 * 33 * 3)

        content.enumerateLines { rawLine, _ in
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line.hasPrefix("#") { return }

            let upper = line.uppercased()
            if upper.hasPrefix("TITLE") {
                if let firstQuote = line.firstIndex(of: "\""),
                   let lastQuote = line.lastIndex(of: "\""),
                   firstQuote < lastQuote {
                    let inner = line[line.index(after: firstQuote)..<lastQuote]
                    if !inner.isEmpty { title = String(inner) }
                }
                return
            }
            if upper.hasPrefix("LUT_3D_SIZE") {
                let parts = line.split(whereSeparator: \.isWhitespace)
                if parts.count >= 2, let n = Int(parts[1]), n >= 2, n <= 129 {
                    size3D = n
                }
                return
            }
            if upper.hasPrefix("LUT_1D_SIZE") {
                let parts = line.split(whereSeparator: \.isWhitespace)
                if parts.count >= 2, let n = Int(parts[1]), n >= 2, n <= 65536 {
                    size1D = n
                }
                return
            }
            if upper.hasPrefix("DOMAIN_MIN") {
                let parts = line.split(whereSeparator: \.isWhitespace)
                if parts.count >= 4,
                   let r = Float(parts[1]), let g = Float(parts[2]), let b = Float(parts[3]) {
                    domainMin = SIMD4<Float>(r, g, b, 0)
                }
                return
            }
            if upper.hasPrefix("DOMAIN_MAX") {
                let parts = line.split(whereSeparator: \.isWhitespace)
                if parts.count >= 4,
                   let r = Float(parts[1]), let g = Float(parts[2]), let b = Float(parts[3]) {
                    domainMax = SIMD4<Float>(r, g, b, 1)
                }
                return
            }
            if upper.hasPrefix("LUT_") || upper.hasPrefix("DOMAIN_") {
                return
            }

            // Parse RGB triplet row
            let parts = line.split(whereSeparator: \.isWhitespace)
            if parts.count >= 3,
               let r = Float(parts[0]),
               let g = Float(parts[1]),
               let b = Float(parts[2]) {
                floats.append(r)
                floats.append(g)
                floats.append(b)
            }
        }

        if size3D > 0 {
            let expectedTriplets = size3D * size3D * size3D
            guard floats.count >= expectedTriplets * 3 else {
                throw NSError(
                    domain: "StudioCam.LUT",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Incomplete 3D LUT (\(floats.count / 3)/\(expectedTriplets) entries)"]
                )
            }
            let tex = try make3DTexture(device: device, dim: size3D, rgbTriplets: floats)
            return Loaded3DLUT(title: title, url: url, dimension: UInt32(size3D), domainMin: domainMin, domainMax: domainMax, texture: tex)
        } else if size1D > 0 {
            let expectedTriplets = size1D
            guard floats.count >= expectedTriplets * 3 else {
                throw NSError(domain: "StudioCam.LUT", code: 2, userInfo: [NSLocalizedDescriptionKey: "Incomplete 1D LUT"])
            }
            // Promote 1D LUT cleanly into a 33x33x33 3D texture so the GPU shader stays a single uniform codepath
            let dim = 33
            var promoted = [Float](repeating: 0, count: dim * dim * dim * 3)
            func sample1D(_ ch: Int, _ t: Float) -> Float {
                let pos = max(0, min(Float(size1D - 1), t * Float(size1D - 1)))
                let i0 = Int(floor(pos))
                let i1 = min(size1D - 1, i0 + 1)
                let frac = pos - Float(i0)
                return floats[i0 * 3 + ch] * (1 - frac) + floats[i1 * 3 + ch] * frac
            }
            var idx = 0
            for b in 0..<dim {
                let bf = Float(b) / Float(dim - 1)
                let bOut = sample1D(2, bf)
                for g in 0..<dim {
                    let gf = Float(g) / Float(dim - 1)
                    let gOut = sample1D(1, gf)
                    for r in 0..<dim {
                        let rf = Float(r) / Float(dim - 1)
                        promoted[idx] = sample1D(0, rf)
                        promoted[idx + 1] = gOut
                        promoted[idx + 2] = bOut
                        idx += 3
                    }
                }
            }
            let tex = try make3DTexture(device: device, dim: dim, rgbTriplets: promoted)
            return Loaded3DLUT(title: title, url: url, dimension: UInt32(dim), domainMin: domainMin, domainMax: domainMax, texture: tex)
        } else {
            throw NSError(domain: "StudioCam.LUT", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing LUT_3D_SIZE in \(url.lastPathComponent)"])
        }
    }

    private static func make3DTexture(device: MTLDevice, dim: Int, rgbTriplets: [Float]) throws -> MTLTexture {
        let desc = MTLTextureDescriptor()
        desc.textureType = .type3D
        desc.pixelFormat = .rgba16Float
        desc.width = dim
        desc.height = dim
        desc.depth = dim
        desc.mipmapLevelCount = 1
        desc.usage = [.shaderRead]

        guard let tex = device.makeTexture(descriptor: desc) else {
            throw NSError(domain: "StudioCam.LUT", code: 4, userInfo: [NSLocalizedDescriptionKey: "Could not allocate 3D LUT texture"])
        }

        let totalVoxels = dim * dim * dim
        var rgba16 = [UInt16](repeating: 0, count: totalVoxels * 4)
        let oneHalf = Float16(1.0).bitPattern
        for i in 0..<totalVoxels {
            rgba16[i * 4 + 0] = Float16(rgbTriplets[i * 3 + 0]).bitPattern
            rgba16[i * 4 + 1] = Float16(rgbTriplets[i * 3 + 1]).bitPattern
            rgba16[i * 4 + 2] = Float16(rgbTriplets[i * 3 + 2]).bitPattern
            rgba16[i * 4 + 3] = oneHalf
        }

        rgba16.withUnsafeBytes { rawPtr in
            tex.replace(
                region: MTLRegionMake3D(0, 0, 0, dim, dim, dim),
                mipmapLevel: 0,
                slice: 0,
                withBytes: rawPtr.baseAddress!,
                bytesPerRow: dim * 4 * MemoryLayout<UInt16>.size,
                bytesPerImage: dim * dim * 4 * MemoryLayout<UInt16>.size
            )
        }
        return tex
    }

    // MARK: - Built-in Studio .cube LUT Generator (33x33x33)

    public static func appSupportSubdirectory(_ name: String) -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        let dir = base.appendingPathComponent("StudioCam", isDirectory: true).appendingPathComponent(name, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    public static func ensureSampleLUTsExist(in directory: URL? = nil) -> [URL] {
        let targetDir = directory ?? appSupportSubdirectory("SampleLUTs")
        try? FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)

        let specs: [(filename: String, title: String, transform: (SIMD3<Float>) -> SIMD3<Float>)] = [
            ("Broadcast_Clean_Rec709.cube", "Broadcast Clean Rec709", { c in
                // Crisp executive contrast S-curve + highlight roll-off + natural skin-tone protection
                let luma = dot(c, SIMD3(0.2126, 0.7152, 0.0722))
                let curvedLuma = luma * luma * (3.0 - 2.0 * luma)
                let contrastMix: Float = 0.24
                var out = c * ((1.0 - contrastMix) + contrastMix * (curvedLuma / max(1e-4, luma)))
                // Subtle clean white balance polish
                out *= SIMD3(1.00, 1.005, 1.015)
                return simd_clamp(out, .zero, .one)
            }),
            ("Warm_Cinema_TealAmber.cube", "Warm Cinema Teal & Amber", { c in
                // Kodak 2383-inspired warm key tones + cool slate shadows with skin hue preservation
                let luma = dot(c, SIMD3(0.2126, 0.7152, 0.0722))
                let sCurve = luma * luma * (3.0 - 2.0 * luma)
                var out = simd_mix(c, c * (sCurve / max(1e-4, luma)), SIMD3<Float>(repeating: 0.32))
                let shadowWeight = (1.0 - luma) * (1.0 - luma)
                let midHighWeight = luma * (1.0 - 0.35 * luma)
                out += shadowWeight * SIMD3(-0.022, 0.008, 0.032)
                out += midHighWeight * SIMD3(0.032, 0.014, -0.020)
                return simd_clamp(out, .zero, .one)
            }),
            ("Crisp_Keynote_Air.cube", "Crisp Keynote Air", { c in
                // Lifted airy shadows, bright clean midtones, soft shoulder roll-off
                let lifted = pow(simd_clamp(c, .zero, .one), SIMD3<Float>(repeating: 0.90))
                let luma = dot(lifted, SIMD3(0.2126, 0.7152, 0.0722))
                let sat = simd_mix(SIMD3<Float>(repeating: luma), lifted, SIMD3<Float>(repeating: 1.08))
                return simd_clamp(SIMD3<Float>(repeating: 0.015) + sat * 0.975, .zero, .one)
            }),
            ("Moody_Studio_Slate.cube", "Moody Studio Slate", { c in
                // Controlled shadow desaturation, rich low-end density, protected warm skin tones
                let luma = dot(c, SIMD3(0.2126, 0.7152, 0.0722))
                let isWarmSkin = max(Float(0), min(Float(1), (c.x - c.z) * 2.2))
                let desatFactor: Float = 0.76 + (0.98 - 0.76) * isWarmSkin
                var out = simd_mix(SIMD3<Float>(repeating: luma), c, SIMD3<Float>(repeating: desatFactor))
                let sCurve = luma * luma * (3.0 - 2.0 * luma)
                out = simd_mix(out, out * (sCurve / max(1e-4, luma)), SIMD3<Float>(repeating: 0.36))
                out += (1.0 - luma) * SIMD3(-0.010, 0.004, 0.018)
                return simd_clamp(out, .zero, .one)
            })
        ]

        var urls: [URL] = []
        let dim = 33
        for spec in specs {
            let fileURL = targetDir.appendingPathComponent(spec.filename)
            urls.append(fileURL)
            if FileManager.default.fileExists(atPath: fileURL.path) { continue }

            var lines = [String]()
            lines.reserveCapacity(dim * dim * dim + 6)
            lines.append("# Generated by StudioCam 4K")
            lines.append("TITLE \"\(spec.title)\"")
            lines.append("LUT_3D_SIZE \(dim)")
            lines.append("DOMAIN_MIN 0.0 0.0 0.0")
            lines.append("DOMAIN_MAX 1.0 1.0 1.0")

            for b in 0..<dim {
                let bf = Float(b) / Float(dim - 1)
                for g in 0..<dim {
                    let gf = Float(g) / Float(dim - 1)
                    for r in 0..<dim {
                        let rf = Float(r) / Float(dim - 1)
                        let out = spec.transform(SIMD3(rf, gf, bf))
                        lines.append(String(format: "%.6f %.6f %.6f", out.x, out.y, out.z))
                    }
                }
            }
            let text = lines.joined(separator: "\n") + "\n"
            try? text.write(to: fileURL, atomically: true, encoding: .utf8)
        }
        return urls
    }
}
