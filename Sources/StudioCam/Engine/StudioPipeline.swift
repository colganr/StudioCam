import Foundation
import AVFoundation
import Vision
import Metal
import CoreVideo
import CoreMedia
import AppKit
import simd

public struct CameraDeviceOption: Identifiable, Equatable {
    public let id: String
    public let name: String
}

public final class StudioPipeline: NSObject, ObservableObject, @unchecked Sendable {
    public let settings: StudioSettings

    @Published public private(set) var cameras: [CameraDeviceOption] = []
    @Published public private(set) var isPipelineRunning: Bool = false
    @Published public private(set) var isRecording: Bool = false
    @Published public private(set) var virtualCamStatus: String = "Off"
    @Published public private(set) var activeMediaLabel: String = "Ice Blue Gradient"
    @Published public private(set) var activeLUTLabel: String = "Off"
    @Published public private(set) var fpsTelemetry: Double = 0
    @Published public private(set) var gpuMsTelemetry: Double = 0
    @Published public private(set) var aneMsTelemetry: Double = 0
    @Published public private(set) var resolutionTelemetry: String = "Idle (0% CPU/GPU)"

    // Consumer tracking for Zero-Idle Power Management
    @MainActor public var isPopoverVisible: Bool = false { didSet { updatePowerState() } }
    @MainActor public var isCleanWindowVisible: Bool = false { didSet { updatePowerState() } }

    public let compositor: MetalStudioCompositor
    public let bgMedia: BackgroundMediaController
    public let sinkBridge = CMIOSinkBridge()
    private let ringLight: ScreenRingLightController

    // Preview callback (fires on render queue for zero-CPU MTKView redraws)
    private let frameCallbackLock = NSLock()
    private var _onFrameRendered: ((MTLTexture) -> Void)?
    public var onFrameRendered: ((MTLTexture) -> Void)? {
        get { frameCallbackLock.lock(); defer { frameCallbackLock.unlock() }; return _onFrameRendered }
        set { frameCallbackLock.lock(); _onFrameRendered = newValue; frameCallbackLock.unlock() }
    }

    private let captureQueue = DispatchQueue(label: "com.thetonegeek.studiocam.capture", qos: .userInteractive)
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private var activeDeviceInput: AVCaptureDeviceInput?
    private var configuredPresetResolution: CaptureResolutionPreset?

    // Vision ANE state (isolated to captureQueue)
    private let visionSequenceHandler = VNSequenceRequestHandler()
    private let segRequest = VNGeneratePersonSegmentationRequest()
    private let faceRequest = VNDetectFaceRectanglesRequest()
    private let handRequest = VNDetectHumanHandPoseRequest()
    private var cachedMaskBuffer: CVPixelBuffer?
    private var trackedFaceRect = SIMD4<Float>(0.5, 0.42, 0.16, 0.24)
    private var frameIndex: UInt64 = 0
    private var lastReactionCooldownTime: TimeInterval = 0

    // FPS & ANE timing state
    private var lastFrameTimestamp: CFTimeInterval = 0
    private var smoothedFPS: Double = 0
    private var smoothedANEMs: Double = 0
    private var lastUIStatsPush: CFTimeInterval = 0

    // Optional 4K HEVC hardware recorder
    private let recorderLock = NSLock()
    private var assetWriter: AVAssetWriter?
    private var writerInput: AVAssetWriterInput?
    private var writerAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var recordingStartPTS: CMTime?

    @MainActor
    public init(settings: StudioSettings) throws {
        self.settings = settings
        let shaderSrc = Self.loadShaderFallbackSource()
        self.compositor = try MetalStudioCompositor(shaderSourceFallback: shaderSrc)
        self.bgMedia = BackgroundMediaController(device: compositor.device)
        self.ringLight = ScreenRingLightController()
        super.init()

        segRequest.qualityLevel = .balanced
        segRequest.outputPixelFormat = kCVPixelFormatType_OneComponent8
        handRequest.maximumHandCount = 2

        refreshCameras()
        restoreCustomBackgroundIfNeeded()
        observeCameraHotplugNotifications()
    }

    @MainActor
    private func observeCameraHotplugNotifications() {
        let nc = NotificationCenter.default
        nc.addObserver(forName: .AVCaptureDeviceWasConnected, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in
                self?.refreshCameras()
            }
        }
        nc.addObserver(forName: .AVCaptureDeviceWasDisconnected, object: nil, queue: .main) { [weak self] note in
            let removedID = (note.object as? AVCaptureDevice)?.uniqueID
            Task { @MainActor in
                guard let self = self else { return }
                let wasActive = (removedID != nil && removedID == self.settings.selectedCameraUniqueID)
                self.refreshCameras()
                if wasActive {
                    // Seamless auto-failover (e.g. USB webcam unplugged -> built-in MacBook FaceTime HD Camera)
                    self.reconfigureCaptureInput()
                }
            }
        }
    }

    public static func loadShaderFallbackSource() -> String {
        let candidates: [URL?] = [
            Bundle.main.url(forResource: "StudioCamShaders", withExtension: "metal"),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Sources/StudioCam/Shaders/StudioCamShaders.metal"),
            URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
                .appendingPathComponent("../../../Sources/StudioCam/Shaders/StudioCamShaders.metal")
        ]
        for urlOpt in candidates {
            if let u = urlOpt, let s = try? String(contentsOf: u, encoding: .utf8) {
                return s
            }
        }
        return ""
    }

    @MainActor
    public func refreshCameras() {
        var types: [AVCaptureDevice.DeviceType] = [.external, .builtInWideAngleCamera, .continuityCamera]
        if #available(macOS 14.0, *) {
            types.append(.deskViewCamera)
        }
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: types,
            mediaType: .video,
            position: .unspecified
        )
        // Exclude virtual cameras so we never create a feedback loop
        let physical = discovery.devices.filter { dev in
            let n = dev.localizedName.lowercased()
            return !n.contains("virtual") && !n.contains("studiocam")
        }
        self.cameras = physical.map { CameraDeviceOption(id: $0.uniqueID, name: $0.localizedName) }

        if settings.selectedCameraUniqueID.isEmpty || !cameras.contains(where: { $0.id == settings.selectedCameraUniqueID }) {
            // Prefer Elgato Facecam 4K automatically if attached
            if let facecam = cameras.first(where: { $0.name.localizedCaseInsensitiveContains("Facecam") }) {
                settings.selectedCameraUniqueID = facecam.id
            } else if let first = cameras.first {
                settings.selectedCameraUniqueID = first.id
            }
        }
    }

    @MainActor
    public func selectCamera(uniqueID: String) {
        guard settings.selectedCameraUniqueID != uniqueID else { return }
        settings.selectedCameraUniqueID = uniqueID
        reconfigureCaptureInput()
    }

    @MainActor
    public func applyResolutionChange() {
        reconfigureCaptureInput()
    }

    @MainActor
    public func selectGradientPreset(_ preset: StudioGradientPreset) {
        settings.selectedGradientID = preset.id
        settings.backgroundMode = .gradient
        activeMediaLabel = "\(preset.name) Gradient"
        updatePowerState()
    }

    @MainActor
    public func selectCustomBackground(url: URL) {
        settings.customBackgroundURL = url
        settings.addRecentURL(url)
        let isVid = BackgroundMediaController.isVideoURL(url)
        settings.backgroundMode = isVid ? .video : .image
        bgMedia.loadMedia(from: url) { [weak self] ok, summary in
            Task { @MainActor in
                guard let self = self else { return }
                self.activeMediaLabel = ok ? summary : "Failed: \(summary)"
                self.bgMedia.setPlaybackActive(self.isPipelineRunning && self.settings.backgroundMode == .video)
            }
        }
    }

    @MainActor
    public func selectLUT(url: URL) {
        do {
            let loaded = try LUTLoader.loadCubeLUT(from: url, device: compositor.device)
            compositor.setActiveLUT(loaded)
            settings.selectedLUTURL = url
            settings.addRecentLUTURL(url)
            settings.lutEnabled = true
            activeLUTLabel = "\(loaded.title) (\(loaded.dimension)³)"
        } catch {
            activeLUTLabel = "LUT Error: \(error.localizedDescription)"
        }
    }

    @MainActor
    public func toggleLUT() {
        if settings.lutEnabled {
            settings.lutEnabled = false
        } else if let existing = settings.selectedLUTURL, FileManager.default.fileExists(atPath: existing.path) {
            selectLUT(url: existing)
        } else if let firstRecent = settings.recentLUTURLs.first {
            selectLUT(url: firstRecent)
        }
    }

    @MainActor
    public func toggleVirtualCamera(_ enabled: Bool) {
        settings.virtualCamEnabled = enabled
        if enabled {
            let (ok, msg) = sinkBridge.start()
            virtualCamStatus = ok ? "Live → \(msg)" : msg
            if !ok {
                settings.virtualCamEnabled = false
            }
        } else {
            sinkBridge.stop()
            virtualCamStatus = "Off"
        }
        updatePowerState()
    }

    @MainActor
    public func syncEdgeLightOverlay() {
        ringLight.update(
            enabled: isPipelineRunning && settings.edgeLightEnabled && settings.screenRingLightEnabled,
            intensity: settings.edgeLightStrength,
            warmth: settings.edgeLightWarmth
        )
    }

    /// Starts hardware capture, Media Engine decode, ANE segmentation, and Metal ONLY when >= 1 output is active.
    /// Drops to 0.0% CPU / 0.0% GPU when all consumers close.
    @MainActor
    public func updatePowerState() {
        let shouldRun = isPopoverVisible || isCleanWindowVisible || settings.virtualCamEnabled || isRecording
        if shouldRun && !isPipelineRunning {
            startPipeline()
        } else if !shouldRun && isPipelineRunning {
            stopPipeline()
        } else {
            bgMedia.setPlaybackActive(shouldRun && settings.backgroundMode == .video)
            syncEdgeLightOverlay()
        }
    }

    @MainActor
    private func startPipeline() {
        isPipelineRunning = true
        bgMedia.setPlaybackActive(settings.backgroundMode == .video)
        syncEdgeLightOverlay()

        let camID = settings.selectedCameraUniqueID
        let res = settings.resolution
        captureQueue.async { [weak self] in
            guard let self = self else { return }
            self.configureSessionLocked(cameraID: camID, resolution: res)
            if !self.captureSession.isRunning {
                self.captureSession.startRunning()
            }
        }
    }

    @MainActor
    private func stopPipeline() {
        isPipelineRunning = false
        bgMedia.setPlaybackActive(false)
        ringLight.hide()
        fpsTelemetry = 0
        gpuMsTelemetry = 0
        aneMsTelemetry = 0
        resolutionTelemetry = "Idle (0% CPU/GPU)"

        captureQueue.async { [weak self] in
            guard let self = self else { return }
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
            self.compositor.resetTemporalHistory()
            self.cachedMaskBuffer = nil
        }
    }

    @MainActor
    private func reconfigureCaptureInput() {
        guard isPipelineRunning else { return }
        let camID = settings.selectedCameraUniqueID
        let res = settings.resolution
        captureQueue.async { [weak self] in
            self?.configureSessionLocked(cameraID: camID, resolution: res)
        }
    }

    private func configureSessionLocked(cameraID: String, resolution: CaptureResolutionPreset) {
        if activeDeviceInput?.device.uniqueID == cameraID && configuredPresetResolution == resolution && !captureSession.inputs.isEmpty {
            return
        }

        captureSession.beginConfiguration()
        defer { captureSession.commitConfiguration() }

        for input in captureSession.inputs {
            captureSession.removeInput(input)
        }

        guard let device = AVCaptureDevice(uniqueID: cameraID) ?? AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device) else {
            return
        }

        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
            activeDeviceInput = input
        }

        if captureSession.outputs.isEmpty {
            // Request hardware-native biplanar NV12 ('420v') for zero conversion from Facecam 4K / FaceTime HD
            videoDataOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
                kCVPixelBufferMetalCompatibilityKey as String: true
            ]
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.setSampleBufferDelegate(self, queue: captureQueue)
            if captureSession.canAddOutput(videoDataOutput) {
                captureSession.addOutput(videoDataOutput)
            }
        }

        // Pick best matching native format (preferring '420v' at requested 3840x2160 or 1920x1080 @ 30-60fps)
        selectBestDeviceFormat(device: device, targetWidth: Int32(resolution.width), targetHeight: Int32(resolution.height))
        configuredPresetResolution = resolution
        compositor.resetTemporalHistory()
    }

    private func selectBestDeviceFormat(device: AVCaptureDevice, targetWidth: Int32, targetHeight: Int32) {
        // Universal camera format scorer across built-in MacBook FaceTime HD (1080p/720p + square Center Stage),
        // Continuity Camera (iPhone), Apple Studio Display, external 4K/1080p/720p UVC webcams, and HDMI capture cards:
        // 1. Exclude portrait/square sensor crops when landscape formats exist.
        // 2. Prefer >= 24 fps real-time video modes over low-fps 5MP/12MP still-photo modes.
        // 3. Cap at requested resolution (4K or 1080p), falling back gracefully to camera's max native resolution
        //    (e.g. 1920x1080 on MacBook FaceTime HD Camera while still compositing 4K backgrounds!).
        let allVideoFormats = device.formats.filter { fmt in
            CMFormatDescriptionGetMediaType(fmt.formatDescription) == kCMMediaType_Video
        }
        guard !allVideoFormats.isEmpty else { return }

        let landscapeRealTime = allVideoFormats.filter { fmt in
            let d = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
            let maxFPS = fmt.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            return d.width > d.height && d.width <= targetWidth && d.height <= targetHeight && maxFPS >= 24.0
        }

        let candidates = !landscapeRealTime.isEmpty ? landscapeRealTime : allVideoFormats.filter { fmt in
            let d = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
            return d.width <= targetWidth && d.height <= targetHeight
        }
        let pool = !candidates.isEmpty ? candidates : allVideoFormats

        let sorted = pool.sorted { a, b in
            let da = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let db = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            // Prefer 16:9-like widescreen aspect over 4:3 when resolution tier is comparable
            let aspectA = abs((Double(da.width) / Double(max(1, da.height))) - (16.0 / 9.0)) < 0.12
            let aspectB = abs((Double(db.width) / Double(max(1, db.height))) - (16.0 / 9.0)) < 0.12
            if da.width != db.width { return da.width > db.width }
            if aspectA != aspectB { return aspectA }
            let subA = CMFormatDescriptionGetMediaSubType(a.formatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            let subB = CMFormatDescriptionGetMediaSubType(b.formatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            if subA != subB { return subA }
            let maxFPSA = a.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            let maxFPSB = b.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            return maxFPSA > maxFPSB
        }

        guard let best = sorted.first else { return }
        do {
            try device.lockForConfiguration()
            device.activeFormat = best
            // Safely pick a valid AVFrameRateRange and use its exact CMTime bounds when discrete
            // (prevents NSInvalidArgumentException on built-in MacBook FaceTime HD / appleh13camerad!)
            if let range30 = best.videoSupportedFrameRateRanges.first(where: { $0.minFrameRate <= 30.01 && $0.maxFrameRate >= 29.97 }) {
                if abs(range30.maxFrameRate - range30.minFrameRate) < 0.1 {
                    device.activeVideoMinFrameDuration = range30.minFrameDuration
                    device.activeVideoMaxFrameDuration = range30.maxFrameDuration
                } else {
                    let dur30 = CMTimeMake(value: 1, timescale: 30)
                    let clamped = CMTimeMaximum(range30.minFrameDuration, CMTimeMinimum(dur30, range30.maxFrameDuration))
                    device.activeVideoMinFrameDuration = clamped
                    device.activeVideoMaxFrameDuration = clamped
                }
            } else if let bestRange = best.videoSupportedFrameRateRanges.max(by: { $0.maxFrameRate < $1.maxFrameRate }) {
                device.activeVideoMinFrameDuration = bestRange.minFrameDuration
                device.activeVideoMaxFrameDuration = bestRange.minFrameDuration
            }
            device.unlockForConfiguration()
        } catch {}
    }

    @MainActor
    private func restoreCustomBackgroundIfNeeded() {
        if let url = settings.customBackgroundURL, FileManager.default.fileExists(atPath: url.path) {
            bgMedia.loadMedia(from: url) { [weak self] ok, summary in
                Task { @MainActor in
                    if ok { self?.activeMediaLabel = summary }
                }
            }
        }
        if let lutURL = settings.selectedLUTURL, FileManager.default.fileExists(atPath: lutURL.path) {
            if let loaded = try? LUTLoader.loadCubeLUT(from: lutURL, device: compositor.device) {
                compositor.setActiveLUT(loaded)
                activeLUTLabel = "\(loaded.title) (\(loaded.dimension)³)"
            }
        }
    }

    // MARK: - 4K HEVC Hardware Recording

    @MainActor
    public func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    @MainActor
    private func startRecording() {
        let outDir = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Movies")
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let outURL = outDir.appendingPathComponent("StudioCam_\(stamp).mov")

        let w = settings.resolution.width
        let h = settings.resolution.height
        guard let writer = try? AVAssetWriter(outputURL: outURL, fileType: .mov) else { return }

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: w,
            AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: w >= 3840 ? 28_000_000 : 10_000_000
            ]
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        input.expectsMediaDataInRealTime = true
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h
            ]
        )
        guard writer.canAdd(input) else { return }
        writer.add(input)
        writer.startWriting()

        recorderLock.lock()
        self.assetWriter = writer
        self.writerInput = input
        self.writerAdaptor = adaptor
        self.recordingStartPTS = nil
        recorderLock.unlock()

        isRecording = true
        updatePowerState()
    }

    @MainActor
    private func stopRecording() {
        recorderLock.lock()
        let writer = self.assetWriter
        let input = self.writerInput
        self.assetWriter = nil
        self.writerInput = nil
        self.writerAdaptor = nil
        self.recordingStartPTS = nil
        recorderLock.unlock()

        isRecording = false
        updatePowerState()

        input?.markAsFinished()
        let savedURL = writer?.outputURL
        writer?.finishWriting {
            if let url = savedURL {
                DispatchQueue.main.async {
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                }
            }
        }
    }
}

extension StudioPipeline: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let camPB = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let snap = settings.threadSafeSnapshot()
        frameIndex &+= 1

        // 1. Run ANE Person Segmentation (plus low-rate 5 Hz Face / Hand tracking only when those features are active)
        let needsSegmentation = (snap.backgroundMode != .none) || snap.portraitEnabled || snap.studioLightEnabled || snap.edgeLightEnabled
        if needsSegmentation {
            let desiredQuality: VNGeneratePersonSegmentationRequest.QualityLevel =
                (snap.segmentationQuality == .accurateMax) ? .accurate : .balanced
            if segRequest.qualityLevel != desiredQuality {
                segRequest.qualityLevel = desiredQuality
            }

            var requests: [VNRequest] = [segRequest]
            if snap.studioLightEnabled && (frameIndex % 6 == 0) {
                requests.append(faceRequest)
            }
            if snap.reactionsGestureEnabled && (frameIndex % 10 == 0) {
                requests.append(handRequest)
            }

            let t0 = CACurrentMediaTime()
            let visionPB = compositor.makeVisionInputBuffer(from: camPB)
            try? visionSequenceHandler.perform(requests, on: visionPB, orientation: .up)
            let aneMs = (CACurrentMediaTime() - t0) * 1000.0
            smoothedANEMs = (smoothedANEMs == 0) ? aneMs : (smoothedANEMs * 0.88 + aneMs * 0.12)

            if let maskObs = segRequest.results?.first {
                cachedMaskBuffer = maskObs.pixelBuffer
            }
            if snap.studioLightEnabled, let face = faceRequest.results?.first {
                let b = face.boundingBox
                // Vision normalized origin is bottom-left; convert to top-left UV
                let cx = Float(b.midX)
                let cy = Float(1.0 - b.midY)
                let rx = Float(max(0.12, b.width * 0.85))
                let ry = Float(max(0.18, b.height * 1.15))
                trackedFaceRect = trackedFaceRect * 0.70 + SIMD4<Float>(cx, cy, rx, ry) * 0.30
            }
            if snap.reactionsGestureEnabled {
                evaluateHandReactionGesture()
            }
        }

        // 2. Pull latest Media Engine 4K video background frame (or static GPU texture)
        let (bgVideoPB, bgStaticTex) = bgMedia.currentBackgroundFrame()

        // 3. Single-pass Metal Guided Filter + 4K Studio Composite
        let sinkActive = snap.virtualCamEnabled
        let sinkDims = sinkActive ? sinkBridge.activeDimensions : (1920, 1080)
        guard let rendered = compositor.renderFrame(
            cameraPixelBuffer: camPB,
            segmentationMaskBuffer: cachedMaskBuffer,
            backgroundVideoBuffer: bgVideoPB,
            backgroundStaticTexture: bgStaticTex,
            faceRect: trackedFaceRect,
            settings: snap,
            produceSink: sinkActive,
            sinkDimensions: sinkDims
        ) else { return }

        // 4. Deliver to active outputs (Preview / Clean Window, Virtual Cam Sink, Recorder)
        onFrameRendered?(rendered.mainTexture)

        if sinkActive, let sinkPB = rendered.sinkPixelBuffer {
            sinkBridge.enqueueFrame(sinkPB, fps: 30)
        }

        recorderLock.lock()
        if let writer = assetWriter, let input = writerInput, let adaptor = writerAdaptor, writer.status == .writing {
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            if recordingStartPTS == nil {
                recordingStartPTS = pts
                writer.startSession(atSourceTime: pts)
            }
            if input.isReadyForMoreMediaData {
                adaptor.append(rendered.mainPixelBuffer, withPresentationTime: pts)
            }
        }
        recorderLock.unlock()

        // 5. Throttled 2 Hz Telemetry Update to MainActor UI (zero main-thread spam)
        let now = CACurrentMediaTime()
        if lastFrameTimestamp > 0 {
            let dt = now - lastFrameTimestamp
            if dt > 0.001 && dt < 1.0 {
                let instFPS = 1.0 / dt
                smoothedFPS = (smoothedFPS == 0) ? instFPS : (smoothedFPS * 0.85 + instFPS * 0.15)
            }
        }
        lastFrameTimestamp = now

        if now - lastUIStatsPush > 0.5 {
            lastUIStatsPush = now
            let fpsVal = smoothedFPS
            let gpuVal = compositor.smoothedGPUMilliseconds
            let aneVal = smoothedANEMs
            let camW = CVPixelBufferGetWidth(camPB)
            let camH = CVPixelBufferGetHeight(camPB)
            let outW = snap.resolution.width
            let outH = snap.resolution.height
            Task { @MainActor [weak self] in
                guard let self = self, self.isPipelineRunning else { return }
                self.fpsTelemetry = fpsVal
                self.gpuMsTelemetry = gpuVal
                self.aneMsTelemetry = aneVal
                self.resolutionTelemetry = "In \(camW)×\(camH) → Out \(outW)×\(outH)"
            }
        }
    }

    nonisolated private func evaluateHandReactionGesture() {
        let now = CACurrentMediaTime()
        guard now - lastReactionCooldownTime > 3.5,
              let hand = handRequest.results?.first else { return }
        guard let thumbTip = try? hand.recognizedPoint(.thumbTip),
              let thumbIP = try? hand.recognizedPoint(.thumbIP),
              let indexTip = try? hand.recognizedPoint(.indexTip),
              let indexMCP = try? hand.recognizedPoint(.indexMCP),
              let middleTip = try? hand.recognizedPoint(.middleTip),
              let middleMCP = try? hand.recognizedPoint(.middleMCP),
              thumbTip.confidence > 0.6, indexTip.confidence > 0.6 else { return }

        // Victory / Peace sign -> Confetti
        if indexTip.location.y > indexMCP.location.y + 0.08 && middleTip.location.y > middleMCP.location.y + 0.08 {
            lastReactionCooldownTime = now
            Task { @MainActor [weak self] in self?.settings.triggerReaction(3) }
        } else if thumbTip.location.y > thumbIP.location.y + 0.05 && indexTip.location.y < indexMCP.location.y + 0.02 {
            // Thumbs up -> Ring burst
            lastReactionCooldownTime = now
            Task { @MainActor [weak self] in self?.settings.triggerReaction(1) }
        }
    }
}
