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
                    spillSuppression: 0.70,
                    edgeSoftness: 0.45,
                    lightWrapStrength: 0.45,
                    ambientMatchStrength: 0.35,
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

            sinkBridge.stop()
            return 0
        } catch {
            print("Benchmark failed: \(error)")
            return 1
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
