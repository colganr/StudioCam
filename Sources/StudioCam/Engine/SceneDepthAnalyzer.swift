import Foundation
import CoreML
import Vision
import Metal
import CoreVideo
import QuartzCore

/// On-chip Apple Neural Engine (ANE) monocular scene-depth analyzer powered by Apple's
/// `DepthAnythingV2SmallF16P6` CoreML model (`18.6 MB`, `.cpuAndNeuralEngine`).
/// Analyzes BOTH real physical rooms (e.g. angled walls receding into a hallway) AND virtual
/// room backgrounds (static images or looping videos), caching a normalized `[0=near .. 1=far]`
/// `.r16Float` Metal texture in VRAM so ongoing per-frame cost is ~0.00% CPU/ANE.
public final class SceneDepthAnalyzer: @unchecked Sendable {
    private let device: MTLDevice
    private let workerQueue = DispatchQueue(label: "com.thetonegeek.studiocam.depth-ane", qos: .utility)
    private let lock = NSLock()

    private var vnRequest: VNCoreMLRequest?
    private var isModelLoading = false
    private var isInferring = false
    private var lastInferenceTime: CFTimeInterval = 0
    private var cachedSourceToken: String = ""

    private var _depthTexture: MTLTexture?
    private var _statusLabel: String = "Initializing ANE Depth…"

    public var depthTexture: MTLTexture? {
        lock.lock()
        defer { lock.unlock() }
        return _depthTexture
    }

    public var statusLabel: String {
        lock.lock()
        defer { lock.unlock() }
        return _statusLabel
    }

    public init(device: MTLDevice) {
        self.device = device
        prepareModelAsync()
    }

    /// Ensures Apple's 18.6 MB `DepthAnythingV2SmallF16P6` model is compiled in Application Support
    /// (downloading once from Apple's HuggingFace repo if absent on a fresh clone).
    public func prepareModelAsync(completion: ((Bool) -> Void)? = nil) {
        lock.lock()
        if vnRequest != nil {
            lock.unlock()
            completion?(true)
            return
        }
        if isModelLoading {
            lock.unlock()
            return
        }
        isModelLoading = true
        lock.unlock()

        workerQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                let modelsDir = LUTLoader.appSupportSubdirectory("Models")
                let compiledURL = modelsDir.appendingPathComponent("DepthAnythingV2SmallF16P6.mlmodelc")
                let pkgURL = modelsDir.appendingPathComponent("DepthAnythingV2SmallF16P6.mlpackage")

                if !FileManager.default.fileExists(atPath: compiledURL.path) {
                    if !FileManager.default.fileExists(atPath: pkgURL.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin").path) {
                        self.setStatus("Downloading ANE Depth Model (18 MB)…")
                        try Self.downloadAppleDepthPackage(to: pkgURL)
                    }
                    self.setStatus("Compiling ANE Depth Model…")
                    let tmpCompiled = try MLModel.compileModel(at: pkgURL)
                    try? FileManager.default.removeItem(at: compiledURL)
                    try FileManager.default.moveItem(at: tmpCompiled, to: compiledURL)
                }

                let cfg = MLModelConfiguration()
                cfg.computeUnits = .cpuAndNeuralEngine
                let mlModel = try MLModel(contentsOf: compiledURL, configuration: cfg)
                let vnModel = try VNCoreMLModel(for: mlModel)
                let req = VNCoreMLRequest(model: vnModel)
                req.imageCropAndScaleOption = .scaleFill

                self.lock.lock()
                self.vnRequest = req
                self.isModelLoading = false
                self._statusLabel = "ANE 3D Depth Ready"
                self.lock.unlock()
                completion?(true)
            } catch {
                self.lock.lock()
                self.isModelLoading = false
                self._statusLabel = "Fallback Perspective Depth"
                self.lock.unlock()
                completion?(false)
            }
        }
    }

    public func invalidateSceneCache() {
        lock.lock()
        cachedSourceToken = ""
        lastInferenceTime = 0
        lock.unlock()
    }

    /// Analyzes a static virtual background CGImage/MTLTexture once when loaded (0% runtime overhead!).
    public func analyzeStaticBackground(cgImage: CGImage, token: String) {
        workerQueue.async { [weak self] in
            guard let self = self else { return }
            guard let req = self.waitForRequestBriefly() else { return }
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            if (try? handler.perform([req])) != nil,
               let obs = req.results?.first as? VNPixelBufferObservation {
                self.uploadNormalizedDistanceMap(disparityBuffer: obs.pixelBuffer, foregroundMask: nil, token: token)
            }
        }
    }

    /// Lazily analyzes either a live camera room frame (excluding foreground person mask so room walls -> hallway
    /// span the full 0..1 depth curve!) OR a virtual room video frame (re-evaluating only on scene switch or every ~4s).
    public func maybeAnalyzeFrame(
        pixelBuffer: CVPixelBuffer,
        foregroundMaskToExclude: CVPixelBuffer?,
        sourceToken: String,
        refreshIntervalSeconds: Double = 4.0,
        forceImmediateSynchronous: Bool = false
    ) {
        let now = CACurrentMediaTime()
        lock.lock()
        let tokenChanged = (sourceToken != cachedSourceToken)
        let due = tokenChanged || (_depthTexture == nil) || ((now - lastInferenceTime) >= refreshIntervalSeconds)
        if !due || (isInferring && !forceImmediateSynchronous) {
            lock.unlock()
            return
        }
        guard let req = vnRequest else {
            lock.unlock()
            return
        }
        isInferring = true
        lastInferenceTime = now
        lock.unlock()

        let work = { [weak self] in
            guard let self = self else { return }
            defer {
                self.lock.lock()
                self.isInferring = false
                self.lock.unlock()
            }
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            if (try? handler.perform([req])) != nil,
               let obs = req.results?.first as? VNPixelBufferObservation {
                self.uploadNormalizedDistanceMap(
                    disparityBuffer: obs.pixelBuffer,
                    foregroundMask: foregroundMaskToExclude,
                    token: sourceToken
                )
            }
        }

        if forceImmediateSynchronous {
            work()
        } else {
            workerQueue.async(execute: work)
        }
    }

    // MARK: - Convert Disparity (High=Near, Low=Far) -> Normalized Distance [0=Near .. 1=Far Hallway]

    private func uploadNormalizedDistanceMap(
        disparityBuffer: CVPixelBuffer,
        foregroundMask: CVPixelBuffer?,
        token: String
    ) {
        guard CVPixelBufferGetPixelFormatType(disparityBuffer) == kCVPixelFormatType_OneComponent16Half else { return }

        CVPixelBufferLockBaseAddress(disparityBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(disparityBuffer, .readOnly) }

        if let fg = foregroundMask {
            CVPixelBufferLockBaseAddress(fg, .readOnly)
        }
        defer {
            if let fg = foregroundMask {
                CVPixelBufferUnlockBaseAddress(fg, .readOnly)
            }
        }

        let w = CVPixelBufferGetWidth(disparityBuffer)
        let h = CVPixelBufferGetHeight(disparityBuffer)
        let dispStride = CVPixelBufferGetBytesPerRow(disparityBuffer) / MemoryLayout<UInt16>.size
        guard let dispBase = CVPixelBufferGetBaseAddress(disparityBuffer)?.assumingMemoryBound(to: Float16.self) else { return }

        let maskW = foregroundMask.map { CVPixelBufferGetWidth($0) } ?? 0
        let maskH = foregroundMask.map { CVPixelBufferGetHeight($0) } ?? 0
        let maskBPR = foregroundMask.map { CVPixelBufferGetBytesPerRow($0) } ?? 0
        let maskBase = foregroundMask.flatMap { CVPixelBufferGetBaseAddress($0)?.assumingMemoryBound(to: UInt8.self) }

        // 1. Robust percentile/trimmed min-max over BACKGROUND pixels only (ignoring person silhouette)
        // so your closest room wall maps to 0.0 (gentle/sharp) and the deepest hallway maps to 1.0 (max optical bokeh)!
        var minDisp: Float = .greatestFiniteMagnitude
        var maxDisp: Float = -.greatestFiniteMagnitude
        var bgSampleCount = 0

        for y in stride(from: 2, to: h - 2, by: 2) {
            let row = dispBase + y * dispStride
            for x in stride(from: 2, to: w - 2, by: 2) {
                if let mPtr = maskBase, maskW > 0, maskH > 0 {
                    let mx = min(maskW - 1, (x * maskW) / w)
                    let my = min(maskH - 1, (y * maskH) / h)
                    if mPtr[my * maskBPR + mx] > 110 {
                        continue // skip foreground person when calibrating room depth range!
                    }
                }
                let v = Float(row[x])
                if v.isFinite {
                    if v < minDisp { minDisp = v }
                    if v > maxDisp { maxDisp = v }
                    bgSampleCount += 1
                }
            }
        }

        if bgSampleCount < 64 || !(maxDisp > minDisp + 1e-4) {
            minDisp = 0.05
            maxDisp = 0.95
        }
        let span = max(1e-4, maxDisp - minDisp)

        // 2. Write normalized distance [0.0 = nearest background surface, 1.0 = farthest hallway] into Float16 buffer
        var distF16 = [UInt16](repeating: 0, count: w * h)
        for y in 0..<h {
            let srcRow = dispBase + y * dispStride
            let dstOff = y * w
            for x in 0..<w {
                let disp = Float(srcRow[x])
                let normNear = max(0.0, min(1.0, (disp - minDisp) / span))
                let distFar = 1.0 - normNear // 0 = near, 1 = far distance!
                distF16[dstOff + x] = Float16(distFar).bitPattern
            }
        }

        lock.lock()
        var tex = _depthTexture
        if tex?.width != w || tex?.height != h {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: w, height: h, mipmapped: false)
            desc.usage = [.shaderRead]
            tex = device.makeTexture(descriptor: desc)
            _depthTexture = tex
        }
        cachedSourceToken = token
        _statusLabel = "ANE 3D Depth Active (\(w)×\(h))"
        lock.unlock()

        if let validTex = tex {
            distF16.withUnsafeBytes { rawPtr in
                validTex.replace(
                    region: MTLRegionMake2D(0, 0, w, h),
                    mipmapLevel: 0,
                    withBytes: rawPtr.baseAddress!,
                    bytesPerRow: w * MemoryLayout<UInt16>.size
                )
            }
        }
    }

    private func waitForRequestBriefly() -> VNCoreMLRequest? {
        for _ in 0..<50 {
            lock.lock()
            let r = vnRequest
            lock.unlock()
            if let r = r { return r }
            Thread.sleep(forTimeInterval: 0.1)
        }
        return nil
    }

    private func setStatus(_ s: String) {
        lock.lock()
        _statusLabel = s
        lock.unlock()
    }

    private static func downloadAppleDepthPackage(to pkgURL: URL) throws {
        let weightsDir = pkgURL.appendingPathComponent("Data/com.apple.CoreML/weights", isDirectory: true)
        try FileManager.default.createDirectory(at: weightsDir, withIntermediateDirectories: true)
        let base = "https://huggingface.co/apple/coreml-depth-anything-v2-small/resolve/main/DepthAnythingV2SmallF16P6.mlpackage"
        let files = [
            "Manifest.json",
            "Data/com.apple.CoreML/model.mlmodel",
            "Data/com.apple.CoreML/weights/weight.bin"
        ]
        for rel in files {
            let remote = URL(string: "\(base)/\(rel)")!
            let data = try Data(contentsOf: remote)
            try data.write(to: pkgURL.appendingPathComponent(rel), options: .atomic)
        }
    }
}
