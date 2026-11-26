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
            let exitCode = Headless4KBenchmark.run()
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
                    onToggleCleanWindow: { appController.toggleCleanFeedWindow() }
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

    init() {
        do {
            let p = try StudioPipeline(settings: settings)
            self.pipeline = p
            seedBundled4KSampleVideosIfPresent(into: p)
        } catch {
            self.startupError = error.localizedDescription
        }
    }

    private func seedBundled4KSampleVideosIfPresent(into pipeline: StudioPipeline) {
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
        // Default to the first 4K sample video if the user hasn't picked a custom file yet
        if settings.customBackgroundURL == nil, let firstVideo = foundSamples.first {
            pipeline.selectCustomBackground(url: firstVideo)
        }

        let builtInLUTs = LUTLoader.ensureSampleLUTsExist()
        for lutURL in builtInLUTs.reversed() {
            settings.addRecentLUTURL(lutURL)
        }
    }

    func toggleCleanFeedWindow() {
        if let w = cleanWindow, w.isVisible {
            w.close()
            cleanWindow = nil
            pipeline?.isCleanWindowVisible = false
            return
        }
        guard let pipeline = pipeline else { return }

        let w = NSWindow(
            contentRect: NSRect(x: 120, y: 120, width: 1280, height: 720),
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        w.title = "StudioCam 4K Clean Feed"
        w.aspectRatio = NSSize(width: 16, height: 9)
        w.backgroundColor = .black
        w.contentView = NSHostingView(
            rootView: MetalPreviewView(
                device: pipeline.compositor.device,
                commandQueue: pipeline.compositor.commandQueue,
                registerFrameProvider: { handler in
                    pipeline.onFrameRendered = handler
                }
            )
        )
        NotificationCenter.default.addObserver(forName: NSWindow.willCloseNotification, object: w, queue: .main) { [weak self] _ in
            Task { @MainActor in
                self?.pipeline?.isCleanWindowVisible = false
                self?.cleanWindow = nil
            }
        }
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        cleanWindow = w
        pipeline.isCleanWindowVisible = true
    }
}

/// Deterministic end-to-end 4K hardware benchmark:
/// 1. Generates two real 3840x2160 4K HEVC sample looping background videos on Apple Media Engine (if not yet generated).
/// 2. Decodes 4K HEVC via `BackgroundMediaController` (`AVPlayerLooper` + `IOSurface` NV12).
/// 3. Runs `VNGeneratePersonSegmentationRequest` (ANE) + `MetalStudioCompositor` (4K Guided Filter + Temporal EMA +
///    Spill Suppression + Studio Light + Edge Light + Portrait Bokeh + 4K Video Composite + 1080p Virtual Cam Sink scale).
/// 4. Verifies `CMIOSinkBridge` zero-cert Virtual Camera connection and prints CPU/GPU/ANE latency metrics.
enum Headless4KBenchmark {
    static func run() -> Int32 {
        print("=== StudioCam 4K End-to-End Apple Silicon Benchmark ===")
        do {
            let sampleDir = LUTLoader.appSupportSubdirectory("SampleBackgrounds")
            try FileManager.default.createDirectory(at: sampleDir, withIntermediateDirectories: true)

            let sample1 = sampleDir.appendingPathComponent("Studio_Cyber_4K.mp4")
            let sample2 = sampleDir.appendingPathComponent("Warm_Tube_Lounge_4K.mp4")

            if !FileManager.default.fileExists(atPath: sample1.path) {
                print("Generating 3840x2160 4K HEVC sample video 1: \(sample1.lastPathComponent)...")
                try generate4KHEVCLoop(at: sample1, warmPalette: false)
            }
            if !FileManager.default.fileExists(atPath: sample2.path) {
                print("Generating 3840x2160 4K HEVC sample video 2: \(sample2.lastPathComponent)...")
                try generate4KHEVCLoop(at: sample2, warmPalette: true)
            }

            let compositor = try MetalStudioCompositor(shaderSourceFallback: StudioPipeline.loadShaderFallbackSource())
            print("Metal GPU Device: \(compositor.device.name)")

            let bgController = BackgroundMediaController(device: compositor.device)
            let loadSem = DispatchSemaphore(value: 0)
            var loadedSummary = ""
            bgController.loadMedia(from: sample1) { _, summary in
                loadedSummary = summary
                loadSem.signal()
            }
            _ = loadSem.wait(timeout: .now() + 10.0)
            bgController.setPlaybackActive(true)
            // Pump main runloop briefly so AVQueuePlayer prerolls Media Engine hardware decoder
            RunLoop.main.run(until: Date().addingTimeInterval(0.45))
            print("Loaded 4K Video Background: \(loadedSummary)")

            let sampleLUTs = LUTLoader.ensureSampleLUTsExist()
            let loadedLUT = try LUTLoader.loadCubeLUT(from: sampleLUTs[0], device: compositor.device)
            compositor.setActiveLUT(loadedLUT)
            print("Loaded 3D Color LUT: \(loadedLUT.title) (\(loadedLUT.dimension)×\(loadedLUT.dimension)×\(loadedLUT.dimension) .cube)")

            // Create synthetic 3840x2160 NV12 ('420v') IOSurface camera buffer with realistic silhouette
            let camPB = try makeSynthetic4KNV12CameraFrame()
            let seqHandler = VNSequenceRequestHandler()
            let segReq = VNGeneratePersonSegmentationRequest()
            segReq.qualityLevel = .balanced
            segReq.outputPixelFormat = kCVPixelFormatType_OneComponent8

            // Also create a 3840x2160 static GPU texture for head-to-head comparison with 4K30 video
            let staticDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 3840, height: 2160, mipmapped: false)
            staticDesc.usage = [.shaderRead]
            staticDesc.storageMode = .private
            let staticTex = compositor.device.makeTexture(descriptor: staticDesc)!

            func makeDailyDriverSnap(bgMode: BackgroundMode, lutEnabled: Bool) -> RenderSettingsSnapshot {
                RenderSettingsSnapshot(
                    resolution: .uhd4K,
                    segmentationQuality: .balancedANE,
                    backgroundMode: bgMode,
                    gradientPreset: StudioGradientPreset.presets[0],
                    portraitEnabled: true,
                    portraitStrength: 0.55,
                    studioLightEnabled: true,
                    studioLightStrength: 0.55,
                    edgeLightEnabled: false,
                    screenRingLightEnabled: false,
                    edgeLightStrength: 0.55,
                    edgeLightWarmth: 0.35,
                    spillSuppression: 0.70,
                    edgeSoftness: 0.45,
                    mirrorPreview: false,
                    reactionsGestureEnabled: false,
                    virtualCamEnabled: true,
                    lutEnabled: lutEnabled,
                    lutStrength: 0.85,
                    activeReactionType: 0,
                    activeReactionStart: 0
                )
            }

            // Warmup ANE & Metal pipelines (6 frames)
            let warmSnap = makeDailyDriverSnap(bgMode: .video, lutEnabled: true)
            for _ in 0..<6 {
                let visionPB = compositor.makeVisionInputBuffer(from: camPB)
                try seqHandler.perform([segReq], on: visionPB, orientation: .up)
                let (vPB, _) = bgController.currentBackgroundFrame()
                _ = compositor.renderFrame(
                    cameraPixelBuffer: camPB,
                    segmentationMaskBuffer: segReq.results?.first?.pixelBuffer,
                    backgroundVideoBuffer: vPB,
                    backgroundStaticTexture: staticTex,
                    faceRect: SIMD4(0.5, 0.42, 0.16, 0.24),
                    settings: warmSnap,
                    produceSink: true
                )
            }

            let sinkBridge = CMIOSinkBridge()
            let (sinkOK, sinkMsg) = sinkBridge.start()
            print("Zero-Cert Virtual Camera Sink Test: ok=\(sinkOK) (\(sinkMsg))")
            print("\n--- Daily-Driver Preset Benchmark (Portrait ON + Studio Light ON + Background ON @ 3840x2160) ---")

            func runProfile(_ name: String, snap: RenderSettingsSnapshot, frames: Int = 60) throws {
                var cpuStart = rusage()
                getrusage(RUSAGE_SELF, &cpuStart)
                let wallStart = CACurrentMediaTime()
                var aneTotalMs = 0.0

                for i in 0..<frames {
                    if snap.backgroundMode == .video && (i % 4 == 0) {
                        RunLoop.main.run(until: Date().addingTimeInterval(0.002))
                    }
                    let tANE0 = CACurrentMediaTime()
                    let visionPB = compositor.makeVisionInputBuffer(from: camPB)
                    try seqHandler.perform([segReq], on: visionPB, orientation: .up)
                    aneTotalMs += (CACurrentMediaTime() - tANE0) * 1000.0

                    let (vPB, _) = bgController.currentBackgroundFrame()
                    if let res = compositor.renderFrame(
                        cameraPixelBuffer: camPB,
                        segmentationMaskBuffer: segReq.results?.first?.pixelBuffer,
                        backgroundVideoBuffer: vPB,
                        backgroundStaticTexture: staticTex,
                        faceRect: SIMD4(0.5, 0.42, 0.16, 0.24),
                        settings: snap,
                        produceSink: sinkOK,
                        sinkDimensions: sinkBridge.activeDimensions
                    ), sinkOK, let sPB = res.sinkPixelBuffer {
                        sinkBridge.enqueueFrame(sPB, fps: 30)
                    }
                }

                let wallSec = CACurrentMediaTime() - wallStart
                var cpuEnd = rusage()
                getrusage(RUSAGE_SELF, &cpuEnd)
                let userSec = Double(cpuEnd.ru_utime.tv_sec - cpuStart.ru_utime.tv_sec)
                    + Double(cpuEnd.ru_utime.tv_usec - cpuStart.ru_utime.tv_usec) / 1e6
                let sysSec = Double(cpuEnd.ru_stime.tv_sec - cpuStart.ru_stime.tv_sec)
                    + Double(cpuEnd.ru_stime.tv_usec - cpuStart.ru_stime.tv_usec) / 1e6
                let estRealtime30CPU = ((userSec + sysSec) / (Double(frames) / 30.0)) * 100.0

                print(String(
                    format: "%-38s | GPU: %.2f ms | ANE: %.2f ms | Max: %.1f FPS | 4K30 CPU: %.2f%% (1 core)",
                    (name as NSString).utf8String!,
                    compositor.smoothedGPUMilliseconds,
                    aneTotalMs / Double(frames),
                    Double(frames) / wallSec,
                    estRealtime30CPU
                ))
            }

            try runProfile("A. Static 4K Image (No LUT)", snap: makeDailyDriverSnap(bgMode: .image, lutEnabled: false))
            try runProfile("B. Static 4K Image + 33³ .cube LUT", snap: makeDailyDriverSnap(bgMode: .image, lutEnabled: true))
            try runProfile("C. 4K30 HEVC Video (No LUT)", snap: makeDailyDriverSnap(bgMode: .video, lutEnabled: false))
            try runProfile("D. 4K30 HEVC Video + 33³ .cube LUT", snap: makeDailyDriverSnap(bgMode: .video, lutEnabled: true))

            sinkBridge.stop()
            bgController.setPlaybackActive(false)
            return 0
        } catch {
            print("Benchmark failed: \(error)")
            return 1
        }
    }

    private static func makeSynthetic4KNV12CameraFrame() throws -> CVPixelBuffer {
        let w = 3840
        let h = 2160
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
                    row[x] = inPerson ? UInt8(165) : UInt8(60)
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
        let totalFrames = 120 // 4.0s seamless cosine loop

        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: w,
            AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 18_000_000
            ]
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
            // Subtle architectural studio slats
            float slat = 0.94f + 0.06f * smoothstep(0.4f, 0.6f, sin(uv.x * 90.0f));
            rgb *= slat * (1.0f - 0.35f * dot(uv - 0.5f, uv - 0.5f));
            outTex.write(float4(saturate(rgb), 1.0f), gid);
        }
        """
        let lib = try dev.makeLibrary(source: genKernelSrc, options: nil)
        let pso = try dev.makeComputePipelineState(function: lib.makeFunction(name: "renderLoopBG")!)

        for f in 0..<totalFrames {
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.003)
            }
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
