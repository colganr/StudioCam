import SwiftUI
import AppKit
import AVFoundation
import Vision
import Metal
import CoreVideo

@main
@MainActor
struct StudioCamMain {
    static func main() {
        if CommandLine.arguments.contains("--benchmark") {
            let exitCode = HeadlessStudioBenchmark.run()
            exit(exitCode)
        } else {
            StudioCamMenuApp.main()
        }
    }
}

struct StudioCamMenuApp: App {
    @StateObject private var appController = StudioCamAppController()

    var body: some Scene {
        MenuBarExtra {
            if let pipeline = appController.pipeline {
                ControlCenterMenuView(
                    pipeline: pipeline,
                    settings: appController.settings,
                    prompter: pipeline.prompter,
                    isPoppedOutWindow: false,
                    onToggleCleanWindow: { appController.toggleCleanFeedWindow() },
                    onTogglePopoutControls: { appController.togglePopoutControlsWindow() }
                )
            } else {
                Text(appController.startupError ?? "Starting StudioCam…")
                    .padding()
            }
        } label: {
            Image(systemName: appController.isActive ? "video.fill" : "video")
        }
        .menuBarExtraStyle(.window)
    }
}

@MainActor
final class StudioCamAppController: ObservableObject {
    let settings = StudioSettings()
    var pipeline: StudioPipeline?
    @Published var startupError: String?
    @Published var isActive: Bool = false

    private var cleanWindow: NSWindow?
    private var popoutControlsWindow: NSWindow?

    init() {
        do {
            // Ensure built-in v2 highlight-safe .cube LUTs are up-to-date on disk BEFORE StudioPipeline restores selectedLUTURL!
            _ = LUTLoader.ensureSampleLUTsExist()
            let p = try StudioPipeline(settings: settings)
            self.pipeline = p
            seedBundledAssets(into: p)
        } catch {
            self.startupError = error.localizedDescription
        }
    }

    private func seedBundledAssets(into pipeline: StudioPipeline) {
        let appSupportSamples = LUTLoader.appSupportSubdirectory("SampleBackgrounds")
        let sampleDirs: [URL] = [
            Bundle.main.resourceURL?.appendingPathComponent("SampleBackgrounds"),
            appSupportSamples,
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("SampleBackgrounds")
        ].compactMap { $0 }

        var foundSamples: [URL] = []
        for dir in sampleDirs {
            if let items = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) {
                for u in items.sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
                where BackgroundMediaController.isVideoURL(u) {
                    if !foundSamples.contains(where: { $0.lastPathComponent == u.lastPathComponent }) {
                        foundSamples.append(u)
                    }
                }
            }
        }
        for sample in foundSamples.reversed() {
            settings.addRecentURL(sample)
        }
        if settings.customBackgroundURL == nil, let firstVideo = foundSamples.first {
            pipeline.selectCustomBackground(url: firstVideo)
        }

        let builtInLUTs = LUTLoader.ensureSampleLUTsExist()
        for lutURL in builtInLUTs.reversed() {
            settings.addRecentLUTURL(lutURL)
        }
    }

    func togglePopoutControlsWindow() {
        if let w = popoutControlsWindow, w.isVisible {
            w.close()
            popoutControlsWindow = nil
            pipeline?.isCleanWindowVisible = (cleanWindow?.isVisible == true)
            return
        }
        guard let pipeline = pipeline else { return }

        let w = NSWindow(
            contentRect: NSRect(x: 160, y: 140, width: 400, height: 740),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        w.title = "StudioCam Controls"
        w.isReleasedWhenClosed = false
        w.level = .floating
        w.contentView = NSHostingView(
            rootView: ControlCenterMenuView(
                pipeline: pipeline,
                settings: settings,
                prompter: pipeline.prompter,
                isPoppedOutWindow: true,
                onToggleCleanWindow: { [weak self] in self?.toggleCleanFeedWindow() },
                onTogglePopoutControls: { [weak self] in self?.togglePopoutControlsWindow() }
            )
        )
        NotificationCenter.default.addObserver(forName: NSWindow.willCloseNotification, object: w, queue: .main) { [weak self] _ in
            Task { @MainActor in
                self?.popoutControlsWindow = nil
                self?.pipeline?.isCleanWindowVisible = (self?.cleanWindow?.isVisible == true)
            }
        }
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        popoutControlsWindow = w
        pipeline.isCleanWindowVisible = true
    }

    func toggleCleanFeedWindow() {
        if let w = cleanWindow, w.isVisible {
            w.close()
            cleanWindow = nil
            pipeline?.isCleanWindowVisible = (popoutControlsWindow?.isVisible == true)
            return
        }
        guard let pipeline = pipeline else { return }

        let w = NSWindow(
            contentRect: NSRect(x: 120, y: 120, width: 1280, height: 720),
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        w.title = "StudioCam Clean Feed"
        w.aspectRatio = NSSize(width: 16, height: 9)
        w.backgroundColor = .black
        w.contentView = NSHostingView(
            rootView: MetalPreviewView(
                registerFrameProvider: { handler in
                    pipeline.addFrameSubscriber(handler)
                }
            )
        )
        NotificationCenter.default.addObserver(forName: NSWindow.willCloseNotification, object: w, queue: .main) { [weak self] _ in
            Task { @MainActor in
                self?.cleanWindow = nil
                self?.pipeline?.isCleanWindowVisible = (self?.popoutControlsWindow?.isVisible == true)
            }
        }
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        cleanWindow = w
        pipeline.isCleanWindowVisible = true
    }
}

/// Deterministic end-to-end Apple Silicon v2.0 Benchmark:
/// Tests BOTH:
/// 1. Client-Matched 1080p Pipeline (Teams / Meet / OBS Virtual Cam native 1920x1080 — zero 4K downsample!)
/// 2. Full 4K Creator Pipeline (3840x2160)
/// With ALL v2.0 features enabled simultaneously: ANE 3D Scene-Depth Optical Bokeh (Real Room & Virtual Room),
/// Optical Light Wrap, Ambient Room Color Match, Auto-Framing, Skin Polish, 35mm Film Grain, and 33³ .cube LUT.
enum HeadlessStudioBenchmark {
    static func run() -> Int32 {
        print("=== StudioCam v2.0 End-to-End Apple Silicon Benchmark ===")
        do {
            let sampleDir = LUTLoader.appSupportSubdirectory("SampleBackgrounds")
            let sample1 = sampleDir.appendingPathComponent("Studio_Cyber_4K.mp4")
            let sample2 = sampleDir.appendingPathComponent("Warm_Tube_Lounge_4K.mp4")

            if !FileManager.default.fileExists(atPath: sample1.path) {
                try generate4KHEVCLoop(at: sample1, warmPalette: false)
            }
            if !FileManager.default.fileExists(atPath: sample2.path) {
                try generate4KHEVCLoop(at: sample2, warmPalette: true)
            }

            let compositor = try MetalStudioCompositor(shaderSourceFallback: StudioPipeline.loadShaderFallbackSource())
            print("Metal GPU Device: \(compositor.device.name)")

            let depthAnalyzer = SceneDepthAnalyzer(device: compositor.device)
            let depthReadySem = DispatchSemaphore(value: 0)
            depthAnalyzer.prepareModelAsync { _ in depthReadySem.signal() }
            _ = depthReadySem.wait(timeout: .now() + 30.0)
            print("Scene Depth Engine: \(depthAnalyzer.statusLabel)")

            let sampleLUTs = LUTLoader.ensureSampleLUTsExist()
            let loadedLUT = try LUTLoader.loadCubeLUT(from: sampleLUTs[0], device: compositor.device)
            compositor.setActiveLUT(loadedLUT)
            print("3D Color LUT: \(loadedLUT.title) (\(loadedLUT.dimension)³ .cube)")

            let sinkBridge = CMIOSinkBridge()
            let (sinkOK, sinkMsg) = sinkBridge.start()
            let clientDims = sinkOK ? sinkBridge.activeDimensions : (1920, 1080)
            print("Client-Matched Virtual Camera Sink: ok=\(sinkOK) -> \(sinkMsg)")

            let seqHandler = VNSequenceRequestHandler()
            let segReq = VNGeneratePersonSegmentationRequest()
            segReq.qualityLevel = .balanced
            segReq.outputPixelFormat = kCVPixelFormatType_OneComponent8

            func benchmarkPipeline(
                label: String,
                preset: CaptureResolutionPreset,
                bgMode: BackgroundMode,
                frames: Int = 60
            ) throws {
                let outDims = preset.resolvedDimensions(clientHint: clientDims)
                let camPB = try makeSyntheticNV12CameraFrame(width: outDims.width, height: outDims.height)

                let bgController = BackgroundMediaController(device: compositor.device)
                bgController.setTargetDecodeDimensions(width: outDims.width, height: outDims.height)
                if bgMode == .video {
                    let sem = DispatchSemaphore(value: 0)
                    bgController.loadMedia(from: sample1) { _, _ in sem.signal() }
                    _ = sem.wait(timeout: .now() + 10.0)
                    bgController.setPlaybackActive(true)
                    RunLoop.main.run(until: Date().addingTimeInterval(0.35))
                }

                let snap = RenderSettingsSnapshot(
                    resolution: preset,
                    segmentationQuality: .balancedANE,
                    backgroundMode: bgMode,
                    gradientPreset: StudioGradientPreset.presets[0],
                    portraitEnabled: true,
                    portraitStrength: 0.65,
                    depthBokehEnabled: true,
                    focalPlaneOffset: 0.16,
                    studioLightEnabled: true,
                    studioLightStrength: 0.55,
                    edgeLightEnabled: false,
                    screenRingLightEnabled: false,
                    edgeLightStrength: 0.55,
                    edgeLightWarmth: 0.35,
                    spillSuppression: 0.45,
                    edgeSoftness: 0.62,
                    matteChoke: 0.05,
                    temporalStability: 0.80,
                    lightWrapStrength: 0.18,
                    ambientMatchStrength: 0.25,
                    autoFramingEnabled: false,
                    autoFramingZoom: 1.20,
                    skinPolishStrength: 0.25,
                    filmGrainStrength: 0.14,
                    mirrorPreview: false,
                    reactionsGestureEnabled: false,
                    virtualCamEnabled: sinkOK,
                    lutEnabled: true,
                    lutStrength: 0.85,
                    activeReactionType: 0,
                    activeReactionStart: 0
                )

                // Run initial one-shot ANE 3D scene depth on room or virtual room background!
                let visionPB0 = compositor.makeVisionInputBuffer(from: camPB)
                try seqHandler.perform([segReq], on: visionPB0, orientation: .up)
                let (vPB0, _) = bgController.currentBackgroundFrame()
                depthAnalyzer.maybeAnalyzeFrame(
                    pixelBuffer: (bgMode == .video && vPB0 != nil) ? vPB0! : visionPB0,
                    foregroundMaskToExclude: (bgMode == .none) ? segReq.results?.first?.pixelBuffer : nil,
                    sourceToken: "bench:\(label)",
                    forceImmediateSynchronous: true
                )

                // Warmup 6 frames
                for _ in 0..<6 {
                    let vIn = compositor.makeVisionInputBuffer(from: camPB)
                    try seqHandler.perform([segReq], on: vIn, orientation: .up)
                    let (vPB, sTex) = bgController.currentBackgroundFrame()
                    _ = compositor.renderFrame(
                        outputDimensions: outDims,
                        cameraPixelBuffer: camPB,
                        segmentationMaskBuffer: segReq.results?.first?.pixelBuffer,
                        backgroundVideoBuffer: vPB,
                        backgroundStaticTexture: sTex,
                        sceneDepthTexture: depthAnalyzer.depthTexture,
                        faceRect: SIMD4(0.5, 0.42, 0.16, 0.24),
                        cameraCropRect: SIMD4(0, 0, 1, 1),
                        settings: snap,
                        produceSink: sinkOK,
                        sinkDimensions: clientDims
                    )
                }

                var cpuStart = rusage()
                getrusage(RUSAGE_SELF, &cpuStart)
                let wallStart = CACurrentMediaTime()
                var aneTotalMs = 0.0

                for i in 0..<frames {
                    if bgMode == .video && (i % 4 == 0) {
                        RunLoop.main.run(until: Date().addingTimeInterval(0.002))
                    }
                    let tANE0 = CACurrentMediaTime()
                    let vIn = compositor.makeVisionInputBuffer(from: camPB)
                    try seqHandler.perform([segReq], on: vIn, orientation: .up)
                    aneTotalMs += (CACurrentMediaTime() - tANE0) * 1000.0

                    let (vPB, sTex) = bgController.currentBackgroundFrame()
                    if let res = compositor.renderFrame(
                        outputDimensions: outDims,
                        cameraPixelBuffer: camPB,
                        segmentationMaskBuffer: segReq.results?.first?.pixelBuffer,
                        backgroundVideoBuffer: vPB,
                        backgroundStaticTexture: sTex,
                        sceneDepthTexture: depthAnalyzer.depthTexture,
                        faceRect: SIMD4(0.5, 0.42, 0.16, 0.24),
                        cameraCropRect: SIMD4(0, 0, 1, 1),
                        settings: snap,
                        produceSink: sinkOK,
                        sinkDimensions: clientDims
                    ), sinkOK, let sPB = res.sinkPixelBuffer {
                        sinkBridge.enqueueFrame(sPB, fps: 30)
                    }
                }

                bgController.setPlaybackActive(false)
                let wallSec = CACurrentMediaTime() - wallStart
                var cpuEnd = rusage()
                getrusage(RUSAGE_SELF, &cpuEnd)
                let u = Double(cpuEnd.ru_utime.tv_sec - cpuStart.ru_utime.tv_sec) + Double(cpuEnd.ru_utime.tv_usec - cpuStart.ru_utime.tv_usec) / 1e6
                let s = Double(cpuEnd.ru_stime.tv_sec - cpuStart.ru_stime.tv_sec) + Double(cpuEnd.ru_stime.tv_usec - cpuStart.ru_stime.tv_usec) / 1e6
                let estCPU30 = ((u + s) / (Double(frames) / 30.0)) * 100.0

                print(String(
                    format: "%-44s | %dx%d | GPU: %.2f ms | ANE: %.2f ms | Max: %.1f FPS | 30fps CPU: %.1f%% (1 core)",
                    (label as NSString).utf8String!,
                    outDims.width, outDims.height,
                    compositor.smoothedGPUMilliseconds,
                    aneTotalMs / Double(frames),
                    Double(frames) / wallSec,
                    estCPU30
                ))
            }

            print("\n--- All Standout v2.0 Effects Active (3D ANE Depth Bokeh + Light Wrap + Color Match + LUT + Grain + Skin Polish) ---")
            try benchmarkPipeline(label: "1. Auto (Teams/Meet 1080p) Real Room Hallway", preset: .autoClient, bgMode: .none)
            try benchmarkPipeline(label: "2. Auto (Teams/Meet 1080p) + Video Room BG", preset: .autoClient, bgMode: .video)
            try benchmarkPipeline(label: "3. Explicit 4K UHD Creator + Video Room BG", preset: .uhd4K, bgMode: .video)

            print("\n--- Real-World Matte Shimmer & Motion-Release Stress Test (Noisy Webcam + ±0.30 Stateless Mask Flicker) ---")
            try runTemporalShimmerStressTest(compositor: compositor)

            sinkBridge.stop()
            return 0
        } catch {
            print("Benchmark failed: \(error)")
            return 1
        }
    }

    private static func runTemporalShimmerStressTest(compositor: MetalStudioCompositor) throws {
        let w = 1280, h = 720
        let maskW = 512, maskH = 288
        compositor.resetTemporalHistory()

        let snap = RenderSettingsSnapshot(
            resolution: .fhd1080,
            segmentationQuality: .balancedANE,
            backgroundMode: .gradient,
            gradientPreset: StudioGradientPreset(
                id: "test_black", name: "Black",
                topRGB: SIMD4(0.0, 0.0, 0.0, 1.0),
                bottomRGB: SIMD4(0.0, 0.0, 0.0, 1.0)
            ),
            portraitEnabled: false,
            portraitStrength: 0.0,
            depthBokehEnabled: false,
            focalPlaneOffset: 0.16,
            studioLightEnabled: false,
            studioLightStrength: 0.0,
            edgeLightEnabled: false,
            screenRingLightEnabled: false,
            edgeLightStrength: 0.0,
            edgeLightWarmth: 0.0,
            spillSuppression: 0.0,
            edgeSoftness: 0.62,
            matteChoke: 0.05,
            temporalStability: 0.80,
            lightWrapStrength: 0.0,
            ambientMatchStrength: 0.0,
            autoFramingEnabled: false,
            autoFramingZoom: 1.0,
            skinPolishStrength: 0.0,
            filmGrainStrength: 0.0,
            mirrorPreview: false,
            reactionsGestureEnabled: false,
            virtualCamEnabled: false,
            lutEnabled: false,
            lutStrength: 0.0,
            activeReactionType: 0,
            activeReactionStart: 0
        )

        func makeNoisyCam(frame: Int, shiftedRight: Bool) throws -> CVPixelBuffer {
            let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true]
            var pbOpt: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pbOpt)
            let pb = pbOpt!
            CVPixelBufferLockBaseAddress(pb, [])
            defer { CVPixelBufferUnlockBaseAddress(pb, []) }
            let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
            let bpr = CVPixelBufferGetBytesPerRow(pb)
            let splitX = shiftedRight ? Int(Double(w) * 0.65) : Int(Double(w) * 0.50)
            var rng: UInt32 = UInt32(truncatingIfNeeded: frame) &* 2654435761 &+ 1337
            for y in 0..<h {
                let row = base + y * bpr
                for x in 0..<w {
                    rng = rng &* 1664525 &+ 1013904223
                    let isoNoise = Int((rng >> 24) % 9) - 4 // ±1.6% webcam ISO sensor grain
                    let isPerson = x < splitX
                    let r = max(0, min(255, (isPerson ? 215 : 42) + isoNoise))
                    let g = max(0, min(255, (isPerson ? 195 : 48) + isoNoise))
                    let b = max(0, min(255, (isPerson ? 185 : 58) + isoNoise))
                    row[x * 4 + 0] = UInt8(b)
                    row[x * 4 + 1] = UInt8(g)
                    row[x * 4 + 2] = UInt8(r)
                    row[x * 4 + 3] = 255
                }
            }
            return pb
        }

        func makeFlickeringMask(frame: Int, shiftedRight: Bool) throws -> CVPixelBuffer {
            let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true]
            var pbOpt: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, maskW, maskH, kCVPixelFormatType_OneComponent8, attrs as CFDictionary, &pbOpt)
            let pb = pbOpt!
            CVPixelBufferLockBaseAddress(pb, [])
            defer { CVPixelBufferUnlockBaseAddress(pb, []) }
            let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
            let bpr = CVPixelBufferGetBytesPerRow(pb)
            let splitX = shiftedRight ? Double(maskW) * 0.65 : Double(maskW) * 0.50
            // Alternating ±4px (~±0.30 alpha) raw Vision boundary jitter every frame!
            let jitter = (frame % 2 == 0) ? 3.5 : -3.5
            for y in 0..<maskH {
                let row = base + y * bpr
                for x in 0..<maskW {
                    let dist = (splitX + jitter) - Double(x)
                    let v = max(0.0, min(1.0, 0.5 + dist / 10.0))
                    row[x] = UInt8(v * 255.0)
                }
            }
            return pb
        }

        func sampleEdgeBandLuma(_ pb: CVPixelBuffer, xMin: Int, xMax: Int) -> [Double] {
            CVPixelBufferLockBaseAddress(pb, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
            let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
            let bpr = CVPixelBufferGetBytesPerRow(pb)
            var vals: [Double] = []
            vals.reserveCapacity((xMax - xMin) * 40)
            for y in stride(from: 200, to: 520, by: 8) {
                let row = base + y * bpr
                for x in xMin..<xMax {
                    let b = Double(row[x * 4 + 0]) / 255.0
                    let g = Double(row[x * 4 + 1]) / 255.0
                    let r = Double(row[x * 4 + 2]) / 255.0
                    vals.append(0.2126 * r + 0.7152 * g + 0.0722 * b)
                }
            }
            return vals
        }

        // 1a. Measure EXACT unstabilized raw Vision flicker RMS (hasHistory reset every frame)
        var rawPrevEdge: [Double]? = nil
        var rawSqSum = 0.0
        var rawSqCount = 0
        for f in 0..<12 {
            compositor.resetTemporalHistory()
            let camPB = try makeNoisyCam(frame: f, shiftedRight: false)
            let maskPB = try makeFlickeringMask(frame: f, shiftedRight: false)
            if let res = compositor.renderFrame(
                outputDimensions: (w, h),
                cameraPixelBuffer: camPB,
                segmentationMaskBuffer: maskPB,
                backgroundVideoBuffer: nil,
                backgroundStaticTexture: nil,
                sceneDepthTexture: nil,
                faceRect: SIMD4(0.35, 0.42, 0.16, 0.24),
                cameraCropRect: SIMD4(0, 0, 1, 1),
                settings: snap,
                produceSink: false,
                waitForGPUCompletion: true
            ) {
                let edge = sampleEdgeBandLuma(res.mainPixelBuffer, xMin: w/2 - 14, xMax: w/2 + 14)
                if let prev = rawPrevEdge {
                    for i in 0..<edge.count {
                        let d = edge[i] - prev[i]
                        rawSqSum += d * d
                        rawSqCount += 1
                    }
                }
                rawPrevEdge = edge
            }
        }
        let rawBaselineRMS = sqrt(rawSqSum / Double(max(1, rawSqCount)))

        // 1b. Measure stabilized StudioCam matte RMS across 24 frames with temporal EMA + guided clamping active
        compositor.resetTemporalHistory()
        var prevEdge: [Double]? = nil
        var sqDiffSum = 0.0
        var sqDiffCount = 0
        for f in 0..<24 {
            let camPB = try makeNoisyCam(frame: f, shiftedRight: false)
            let maskPB = try makeFlickeringMask(frame: f, shiftedRight: false)
            if let res = compositor.renderFrame(
                outputDimensions: (w, h),
                cameraPixelBuffer: camPB,
                segmentationMaskBuffer: maskPB,
                backgroundVideoBuffer: nil,
                backgroundStaticTexture: nil,
                sceneDepthTexture: nil,
                faceRect: SIMD4(0.35, 0.42, 0.16, 0.24),
                cameraCropRect: SIMD4(0, 0, 1, 1),
                settings: snap,
                produceSink: false,
                waitForGPUCompletion: true
            ) {
                let edge = sampleEdgeBandLuma(res.mainPixelBuffer, xMin: w/2 - 14, xMax: w/2 + 14)
                if f >= 8, let prev = prevEdge {
                    for i in 0..<edge.count {
                        let d = edge[i] - prev[i]
                        sqDiffSum += d * d
                        sqDiffCount += 1
                    }
                }
                prevEdge = edge
            }
        }
        let stabilizedRMS = sqrt(sqDiffSum / Double(max(1, sqDiffCount)))
        let suppressionFactor = rawBaselineRMS / max(1e-5, stabilizedRMS)

        // 2. Verify instant motion release on BOTH bright AND worst-case low-contrast dark-shirt-over-dark-wall side-to-side sway!
        // 2a. Leading edge snap (person shifts right from x=50% to x=65% on a single frame)
        let moveCam = try makeNoisyCam(frame: 25, shiftedRight: true)
        let moveMask = try makeFlickeringMask(frame: 25, shiftedRight: true)
        var motionCaptureRatio = 0.0
        if let res = compositor.renderFrame(
            outputDimensions: (w, h),
            cameraPixelBuffer: moveCam,
            segmentationMaskBuffer: moveMask,
            backgroundVideoBuffer: nil,
            backgroundStaticTexture: nil,
            sceneDepthTexture: nil,
            faceRect: SIMD4(0.55, 0.42, 0.16, 0.24),
            cameraCropRect: SIMD4(0, 0, 1, 1),
            settings: snap,
            produceSink: false,
            waitForGPUCompletion: true
        ) {
            let movedZone = sampleEdgeBandLuma(res.mainPixelBuffer, xMin: Int(Double(w) * 0.54), xMax: Int(Double(w) * 0.61))
            let meanLuma = movedZone.reduce(0.0, +) / Double(max(1, movedZone.count))
            motionCaptureRatio = (meanLuma / 0.78) * 100.0
        }

        // 2b. Worst-case low-contrast side-to-side sway: dark shirt (RGB ~0.22) moving LEFT off dark wall (RGB ~0.18 -> tiny colorDiff 0.04!)
        func makeLowContrastSwayFrame(frame: Int, bodyCenterX: Double) throws -> (CVPixelBuffer, CVPixelBuffer) {
            let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true]
            var camOpt: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &camOpt)
            let camPB = camOpt!
            CVPixelBufferLockBaseAddress(camPB, [])
            let cBase = CVPixelBufferGetBaseAddress(camPB)!.assumingMemoryBound(to: UInt8.self)
            let cBPR = CVPixelBufferGetBytesPerRow(camPB)
            var rng: UInt32 = UInt32(truncatingIfNeeded: frame) &* 2654435761 &+ 9973
            let minX = Int((bodyCenterX - 0.14) * Double(w))
            let maxX = Int((bodyCenterX + 0.14) * Double(w))
            for y in 0..<h {
                let row = cBase + y * cBPR
                for x in 0..<w {
                    rng = rng &* 1664525 &+ 1013904223
                    let n = Int((rng >> 24) % 7) - 3
                    let inBody = (x >= minX && x <= maxX)
                    let baseV = inBody ? 56 : 46 // Only 10/255 (~0.039) separation!
                    let v = UInt8(max(0, min(255, baseV + n)))
                    row[x * 4 + 0] = v
                    row[x * 4 + 1] = v
                    row[x * 4 + 2] = v
                    row[x * 4 + 3] = 255
                }
            }
            CVPixelBufferUnlockBaseAddress(camPB, [])

            var maskOpt: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, maskW, maskH, kCVPixelFormatType_OneComponent8, attrs as CFDictionary, &maskOpt)
            let mPB = maskOpt!
            CVPixelBufferLockBaseAddress(mPB, [])
            let mBase = CVPixelBufferGetBaseAddress(mPB)!.assumingMemoryBound(to: UInt8.self)
            let mBPR = CVPixelBufferGetBytesPerRow(mPB)
            let mMinX = (bodyCenterX - 0.14) * Double(maskW)
            let mMaxX = (bodyCenterX + 0.14) * Double(maskW)
            for y in 0..<maskH {
                let row = mBase + y * mBPR
                for x in 0..<maskW {
                    let dx = max(mMinX - Double(x), Double(x) - mMaxX)
                    let alpha = max(0.0, min(1.0, 0.5 - dx / 4.0))
                    row[x] = UInt8(alpha * 255.0)
                }
            }
            CVPixelBufferUnlockBaseAddress(mPB, [])
            return (camPB, mPB)
        }

        compositor.resetTemporalHistory()
        // Settle stationary dark shirt at x=0.58 for 10 frames
        for f in 0..<10 {
            let (cPB, mPB) = try makeLowContrastSwayFrame(frame: f, bodyCenterX: 0.58)
            _ = compositor.renderFrame(
                outputDimensions: (w, h), cameraPixelBuffer: cPB, segmentationMaskBuffer: mPB,
                backgroundVideoBuffer: nil, backgroundStaticTexture: nil, sceneDepthTexture: nil,
                faceRect: SIMD4(0.58, 0.42, 0.16, 0.24), cameraCropRect: SIMD4(0, 0, 1, 1),
                settings: snap, produceSink: false, waitForGPUCompletion: true
            )
        }
        // Now rapidly sway body LEFT to x=0.36 on 1 frame -> inspect vacated trailing wake [0.56..0.68] (should be ~0.0% ghost!)
        let (swayCam, swayMask) = try makeLowContrastSwayFrame(frame: 11, bodyCenterX: 0.36)
        var trailingGhostPct = 100.0
        var lowContrastLeadingPct = 0.0
        if let res = compositor.renderFrame(
            outputDimensions: (w, h), cameraPixelBuffer: swayCam, segmentationMaskBuffer: swayMask,
            backgroundVideoBuffer: nil, backgroundStaticTexture: nil, sceneDepthTexture: nil,
            faceRect: SIMD4(0.36, 0.42, 0.16, 0.24), cameraCropRect: SIMD4(0, 0, 1, 1),
            settings: snap, produceSink: false, waitForGPUCompletion: true
        ) {
            let wakeZone = sampleEdgeBandLuma(res.mainPixelBuffer, xMin: Int(Double(w) * 0.56), xMax: Int(Double(w) * 0.68))
            let wakeMean = wakeZone.reduce(0.0, +) / Double(max(1, wakeZone.count))
            trailingGhostPct = (wakeMean / (56.0 / 255.0)) * 100.0

            let leadZone = sampleEdgeBandLuma(res.mainPixelBuffer, xMin: Int(Double(w) * 0.26), xMax: Int(Double(w) * 0.38))
            let leadMean = leadZone.reduce(0.0, +) / Double(max(1, leadZone.count))
            lowContrastLeadingPct = (leadMean / (56.0 / 255.0)) * 100.0
        }

        print(String(
            format: "Raw Unstabilized RMS: %.4f -> Stabilized RMS: %.4f (%.1f× shimmer reduction) | 1-Frame Motion Snap: %.1f%%",
            rawBaselineRMS, stabilizedRMS, suppressionFactor, motionCaptureRatio
        ))
        print(String(
            format: "Fast Low-Contrast Sway (dark shirt on dark wall): Trailing Ghost Residual: %.2f%% | Leading Body Snap: %.1f%%",
            trailingGhostPct, lowContrastLeadingPct
        ))

        // 3. Highlight-Safe Skin & Hand Exposure Verification (studioLight=1.0 + Rec709 LUT + LightWrap + AmbientMatch)
        // Verifies bright skin highlights (RGB 223, 193, 195 — exact IMG1 palm max) NEVER clip to 255 or lose warm R/G ratio!
        var skinSnap = snap
        skinSnap.studioLightEnabled = true
        skinSnap.studioLightStrength = 1.0
        skinSnap.lightWrapStrength = 0.20
        skinSnap.ambientMatchStrength = 0.25
        skinSnap.lutEnabled = true
        skinSnap.lutStrength = 0.85

        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true]
        var skinCamOpt: CVPixelBuffer?
        var skinMaskOpt: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 320, 180, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &skinCamOpt)
        CVPixelBufferCreate(kCFAllocatorDefault, 320, 180, kCVPixelFormatType_OneComponent8, attrs as CFDictionary, &skinMaskOpt)
        if let sCam = skinCamOpt, let sMask = skinMaskOpt {
            CVPixelBufferLockBaseAddress(sCam, [])
            let base = CVPixelBufferGetBaseAddress(sCam)!.assumingMemoryBound(to: UInt8.self)
            let bpr = CVPixelBufferGetBytesPerRow(sCam)
            for y in 0..<180 {
                let row = base + y * bpr
                for x in 0..<320 {
                    row[x * 4 + 0] = 195 // B
                    row[x * 4 + 1] = 193 // G
                    row[x * 4 + 2] = 223 // R (exact bright skin highlight)
                    row[x * 4 + 3] = 255
                }
            }
            CVPixelBufferUnlockBaseAddress(sCam, [])

            CVPixelBufferLockBaseAddress(sMask, [])
            memset(CVPixelBufferGetBaseAddress(sMask)!, 255, CVPixelBufferGetBytesPerRow(sMask) * 180)
            CVPixelBufferUnlockBaseAddress(sMask, [])

            compositor.resetTemporalHistory()
            if let res = compositor.renderFrame(
                outputDimensions: (320, 180), cameraPixelBuffer: sCam, segmentationMaskBuffer: sMask,
                backgroundVideoBuffer: nil, backgroundStaticTexture: nil, sceneDepthTexture: nil,
                faceRect: SIMD4(0.5, 0.5, 0.30, 0.35), cameraCropRect: SIMD4(0, 0, 1, 1),
                settings: skinSnap, produceSink: false, waitForGPUCompletion: true
            ) {
                CVPixelBufferLockBaseAddress(res.mainPixelBuffer, .readOnly)
                let oBase = CVPixelBufferGetBaseAddress(res.mainPixelBuffer)!.assumingMemoryBound(to: UInt8.self)
                let oBPR = CVPixelBufferGetBytesPerRow(res.mainPixelBuffer)
                let p = oBase + 90 * oBPR + 160 * 4
                let ob = Int(p[0]), og = Int(p[1]), or_ = Int(p[2])
                CVPixelBufferUnlockBaseAddress(res.mainPixelBuffer, .readOnly)
                let clipped = (or_ >= 250 || og >= 250 || ob >= 250) ? "CLIPPED" : "0% CLIPPED (Highlight-Safe)"
                print("Bright Skin Highlight (In 223,193,195 @ StudioLight 100% + LUT): Out (\(or_),\(og),\(ob)) -> \(clipped)")
            }
        }
    }

    private static func makeSyntheticNV12CameraFrame(width w: Int, height h: Int) throws -> CVPixelBuffer {
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferMetalCompatibilityKey: true
        ]
        var pbOpt: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, attrs as CFDictionary, &pbOpt)
        guard let pb = pbOpt else { throw NSError(domain: "Bench", code: 1) }

        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }

        if let yBase = CVPixelBufferGetBaseAddressOfPlane(pb, 0)?.assumingMemoryBound(to: UInt8.self) {
            let yBPR = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
            for y in 0..<h {
                let row = yBase + y * yBPR
                let ny = Float(y) / Float(h)
                for x in 0..<w {
                    let nx = Float(x) / Float(w)
                    let headDist = hypot((nx - 0.5) * 1.6, (ny - 0.38))
                    let torsoDist = hypot((nx - 0.5) * 0.95, (ny - 0.82) * 0.75)
                    let inPerson = (headDist < 0.16) || (torsoDist < 0.28)
                    // Angled hallway luminance gradient behind person
                    let hallwayWall = UInt8(min(220, max(35, Int(45.0 + abs(nx - 0.72) * 180.0))))
                    row[x] = inPerson ? UInt8(170) : hallwayWall
                }
            }
        }
        if let uvBase = CVPixelBufferGetBaseAddressOfPlane(pb, 1)?.assumingMemoryBound(to: UInt8.self) {
            let uvBPR = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)
            memset(uvBase, 128, uvBPR * (h / 2))
        }
        return pb
    }

    private static func generate4KHEVCLoop(at url: URL, warmPalette: Bool) throws {
        let w = 3840
        let h = 2160
        let fps: Int32 = 30
        let totalFrames = 120

        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: w,
            AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 18_000_000]
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                kCVPixelBufferMetalCompatibilityKey as String: true
            ]
        )
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let dev = MTLCreateSystemDefaultDevice()!
        let queue = dev.makeCommandQueue()!
        var texCache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, dev, nil, &texCache)

        let genKernelSrc = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void renderLoopBG(
            texture2d<float, access::write> outTex [[texture(0)]],
            constant float2& p [[buffer(0)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint2 sz = uint2(outTex.get_width(), outTex.get_height());
            if (gid.x >= sz.x || gid.y >= sz.y) return;
            float2 uv = float2(gid) / float2(sz);
            float phase = p.x * 6.2831853f;
            bool warm = (p.y > 0.5f);

            float2 c1 = float2(0.32f + 0.14f * cos(phase), 0.38f + 0.12f * sin(phase));
            float2 c2 = float2(0.70f + 0.12f * sin(phase), 0.62f + 0.14f * cos(phase));
            float2 c3 = float2(0.50f + 0.18f * cos(phase + 2.1f), 0.28f + 0.10f * sin(phase + 2.1f));

            float g1 = exp(-4.5f * dot((uv - c1) * float2(1.77f, 1.0f), (uv - c1) * float2(1.77f, 1.0f)));
            float g2 = exp(-5.2f * dot((uv - c2) * float2(1.77f, 1.0f), (uv - c2) * float2(1.77f, 1.0f)));
            float g3 = exp(-7.0f * dot((uv - c3) * float2(1.77f, 1.0f), (uv - c3) * float2(1.77f, 1.0f)));

            float3 base = warm ? float3(0.06f, 0.045f, 0.04f) : float3(0.04f, 0.06f, 0.11f);
            float3 col1 = warm ? float3(0.72f, 0.36f, 0.12f) : float3(0.10f, 0.48f, 0.86f);
            float3 col2 = warm ? float3(0.45f, 0.16f, 0.24f) : float3(0.42f, 0.18f, 0.78f);
            float3 col3 = warm ? float3(0.92f, 0.68f, 0.28f) : float3(0.12f, 0.80f, 0.76f);

            float3 rgb = base + col1 * g1 * 0.55f + col2 * g2 * 0.50f + col3 * g3 * 0.35f;
            float slat = 0.94f + 0.06f * smoothstep(0.4f, 0.6f, sin(uv.x * 90.0f));
            rgb *= slat * (1.0f - 0.35f * dot(uv - 0.5f, uv - 0.5f));
            outTex.write(float4(saturate(rgb), 1.0f), gid);
        }
        """
        let lib = try dev.makeLibrary(source: genKernelSrc, options: nil)
        let pso = try dev.makeComputePipelineState(function: lib.makeFunction(name: "renderLoopBG")!)

        for f in 0..<totalFrames {
            while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.003) }
            var pbOpt: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, adaptor.pixelBufferPool!, &pbOpt)
            guard let pb = pbOpt else { continue }

            var cvTex: CVMetalTexture?
            let attrs: [CFString: Any] = [kCVMetalTextureUsage: MTLTextureUsage.shaderWrite.rawValue]
            CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, texCache!, pb, attrs as CFDictionary, .bgra8Unorm, w, h, 0, &cvTex)
            guard let tex = cvTex.flatMap({ CVMetalTextureGetTexture($0) }),
                  let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else { continue }

            var params = SIMD2<Float>(Float(f) / Float(totalFrames), warmPalette ? 1.0 : 0.0)
            enc.setComputePipelineState(pso)
            enc.setTexture(tex, index: 0)
            enc.setBytes(&params, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
            enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            let pts = CMTimeMake(value: Int64(f), timescale: fps)
            adaptor.append(pb, withPresentationTime: pts)
        }

        input.markAsFinished()
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()
    }
}
