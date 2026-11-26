import Foundation
import AVFoundation
import Vision
import Metal
import CoreVideo
import CoreMedia
import QuartzCore
import AppKit
import simd

public struct CameraDeviceOption: Identifiable, Equatable {
    public let id: String
    public let name: String
}

public final class StudioPipeline: NSObject, ObservableObject, @unchecked Sendable {
    public let settings: StudioSettings
    @MainActor public let prompter: LensPrompterController

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

    @MainActor public var isPopoverVisible: Bool = false { didSet { updatePowerState() } }
    @MainActor public var isCleanWindowVisible: Bool = false { didSet { updatePowerState() } }

    public let compositor: MetalStudioCompositor
    public let bgMedia: BackgroundMediaController
    public let depthAnalyzer: SceneDepthAnalyzer
    public let sinkBridge = CMIOSinkBridge()
    private let ringLight: ScreenRingLightController

    private let frameCallbackLock = NSLock()
    private var frameSubscribers: [UUID: (CVPixelBuffer) -> Void] = [:]

    public func addFrameSubscriber(_ handler: @escaping (CVPixelBuffer) -> Void) -> () -> Void {
        let id = UUID()
        frameCallbackLock.lock()
        frameSubscribers[id] = handler
        frameCallbackLock.unlock()
        return { [weak self] in
            guard let self = self else { return }
            self.frameCallbackLock.lock()
            self.frameSubscribers.removeValue(forKey: id)
            self.frameCallbackLock.unlock()
        }
    }

    private func broadcastCompletedFrame(_ pixelBuffer: CVPixelBuffer) {
        frameCallbackLock.lock()
        let handlers = Array(frameSubscribers.values)
        frameCallbackLock.unlock()
        for h in handlers {
            h(pixelBuffer)
        }
    }

    private let captureQueue = DispatchQueue(label: "com.thetonegeek.studiocam.capture", qos: .userInteractive)
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private var activeDeviceInput: AVCaptureDeviceInput?
    private var configuredDimensions: (Int, Int) = (0, 0)
    private var configuredSensorHeadroom: Bool = false

    // Vision ANE state (isolated to captureQueue)
    private let visionSequenceHandler = VNSequenceRequestHandler()
    private let segRequest = VNGeneratePersonSegmentationRequest()
    private let faceRequest = VNDetectFaceRectanglesRequest()
    private let handRequest = VNDetectHumanHandPoseRequest()
    private var cachedMaskBuffer: CVPixelBuffer?
    private var trackedFaceRect = SIMD4<Float>(0.5, 0.42, 0.16, 0.24)
    private var frameIndex: UInt64 = 0
    private var lastReactionCooldownTime: TimeInterval = 0

    // Dead-zone tripod spring damper for Lossless Cinema Auto-Framing ("Center Stage")
    private var tripodAnchorCenter = SIMD2<Float>(0.5, 0.44)
    private var smoothedCropCenter = SIMD2<Float>(0.5, 0.5)
    private var smoothedCropScale: Float = 1.0

    // FPS & ANE timing
    private var lastFrameTimestamp: CFTimeInterval = 0
    private var smoothedFPS: Double = 0
    private var smoothedANEMs: Double = 0
    private var lastUIStatsPush: CFTimeInterval = 0

    // Hardware HEVC recorder
    private let recorderLock = NSLock()
    private var assetWriter: AVAssetWriter?
    private var writerInput: AVAssetWriterInput?
    private var writerAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var recordingStartPTS: CMTime?

    @MainActor
    public init(settings: StudioSettings) throws {
        self.settings = settings
        self.prompter = LensPrompterController()
        let shaderSrc = Self.loadShaderFallbackSource()
        self.compositor = try MetalStudioCompositor(shaderSourceFallback: shaderSrc)
        self.bgMedia = BackgroundMediaController(device: compositor.device)
        self.depthAnalyzer = SceneDepthAnalyzer(device: compositor.device)
        self.ringLight = ScreenRingLightController()
        super.init()

        segRequest.qualityLevel = .balanced
        segRequest.outputPixelFormat = kCVPixelFormatType_OneComponent8
        handRequest.maximumHandCount = 2

        // Analyze depth immediately whenever a static virtual room background image loads!
        bgMedia.onStaticImageLoaded = { [weak self] cgImage, token in
            self?.depthAnalyzer.analyzeStaticBackground(cgImage: cgImage, token: "img:\(token)")
        }

        // Dynamically adapt pipeline resolution if downstream app (Teams/Meet/Zoom/OBS) changes active CMIO format!
        sinkBridge.onActiveFormatChanged = { [weak self] dims in
            Task { @MainActor in
                guard let self = self else { return }
                self.virtualCamStatus = "Live (\(dims.0)×\(dims.1))"
                if self.settings.resolution == .autoClient {
                    self.reconfigureCaptureInput()
                }
            }
        }

        refreshCameras()
        restoreCustomBackgroundIfNeeded()
        observeCameraHotplugNotifications()
    }

    @MainActor
    private func observeCameraHotplugNotifications() {
        let nc = NotificationCenter.default
        nc.addObserver(forName: .AVCaptureDeviceWasConnected, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.refreshCameras() }
        }
        nc.addObserver(forName: .AVCaptureDeviceWasDisconnected, object: nil, queue: .main) { [weak self] note in
            let removedID = (note.object as? AVCaptureDevice)?.uniqueID
            Task { @MainActor in
                guard let self = self else { return }
                let wasActive = (removedID != nil && removedID == self.settings.selectedCameraUniqueID)
                self.refreshCameras()
                if wasActive { self.reconfigureCaptureInput() }
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

    /// Computes exact target pipeline dimensions without wasted 4K->1080p downsampling.
    public nonisolated func resolvePipelineDimensions(for snap: RenderSettingsSnapshot) -> (width: Int, height: Int) {
        let clientHint: (Int, Int)? = snap.virtualCamEnabled ? sinkBridge.activeDimensions : (1920, 1080)
        return snap.resolution.resolvedDimensions(clientHint: clientHint)
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
        let physical = discovery.devices.filter { dev in
            let n = dev.localizedName.lowercased()
            return !n.contains("virtual") && !n.contains("studiocam")
        }
        self.cameras = physical.map { CameraDeviceOption(id: $0.uniqueID, name: $0.localizedName) }

        if settings.selectedCameraUniqueID.isEmpty || !cameras.contains(where: { $0.id == settings.selectedCameraUniqueID }) {
            if let facecam = cameras.first(where: { $0.name.localizedCaseInsensitiveContains("Facecam") }) {
                settings.selectedCameraUniqueID = facecam.id
            } else if let first = cameras.first {
                settings.selectedCameraUniqueID = first.id
            }
        }
    }

    @MainActor
    public func applyStudioProfile(_ profile: StudioProfilePreset) {
        settings.activeProfileID = profile.id
        settings.resolution = profile.resolution
        settings.portraitEnabled = profile.portraitEnabled
        settings.portraitStrength = profile.portraitStrength
        settings.depthBokehEnabled = profile.depthBokehEnabled
        settings.focalPlaneOffset = profile.focalPlaneOffset
        settings.studioLightEnabled = profile.studioLightEnabled
        settings.studioLightStrength = profile.studioLightStrength
        settings.edgeLightEnabled = profile.edgeLightEnabled
        settings.lightWrapStrength = profile.lightWrapStrength
        settings.ambientMatchStrength = profile.ambientMatchStrength
        settings.autoFramingEnabled = profile.autoFramingEnabled
        settings.skinPolishStrength = profile.skinPolishStrength
        settings.filmGrainStrength = profile.filmGrainStrength

        if profile.backgroundMode == .none {
            settings.backgroundMode = .none
            activeMediaLabel = "Real Room + Hallway Depth"
            depthAnalyzer.invalidateSceneCache()
        } else if profile.backgroundMode == .video {
            if let custom = settings.customBackgroundURL {
                selectCustomBackground(url: custom)
            } else if let firstVid = settings.recentBackgroundURLs.first(where: { BackgroundMediaController.isVideoURL($0) }) {
                selectCustomBackground(url: firstVid)
            }
        }

        if profile.lutEnabled, let hint = profile.lutFilenameHint,
           let match = settings.recentLUTURLs.first(where: { $0.lastPathComponent == hint }) {
            selectLUT(url: match)
        } else {
            settings.lutEnabled = profile.lutEnabled
        }

        syncEdgeLightOverlay()
        reconfigureCaptureInput()
    }

    @MainActor
    public func selectCamera(uniqueID: String) {
        guard settings.selectedCameraUniqueID != uniqueID else { return }
        settings.selectedCameraUniqueID = uniqueID
        depthAnalyzer.invalidateSceneCache()
        reconfigureCaptureInput()
    }

    @MainActor
    public func applyResolutionChange() {
        let dims = resolvePipelineDimensions(for: settings.threadSafeSnapshot())
        bgMedia.setTargetDecodeDimensions(width: dims.width, height: dims.height)
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
        let dims = resolvePipelineDimensions(for: settings.threadSafeSnapshot())
        bgMedia.setTargetDecodeDimensions(width: dims.width, height: dims.height)
        depthAnalyzer.invalidateSceneCache()

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
            } else if settings.resolution == .autoClient {
                reconfigureCaptureInput()
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
        let snap = settings.threadSafeSnapshot()
        let dims = resolvePipelineDimensions(for: snap)
        bgMedia.setTargetDecodeDimensions(width: dims.width, height: dims.height)
        bgMedia.setPlaybackActive(settings.backgroundMode == .video)
        syncEdgeLightOverlay()

        let camID = settings.selectedCameraUniqueID
        let wantSensorHeadroom = settings.autoFramingEnabled
        captureQueue.async { [weak self] in
            guard let self = self else { return }
            self.configureSessionLocked(cameraID: camID, targetDims: dims, allowSensorHeadroom: wantSensorHeadroom)
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
    public func reconfigureCaptureInput() {
        guard isPipelineRunning else { return }
        let camID = settings.selectedCameraUniqueID
        let dims = resolvePipelineDimensions(for: settings.threadSafeSnapshot())
        bgMedia.setTargetDecodeDimensions(width: dims.width, height: dims.height)
        let wantSensorHeadroom = settings.autoFramingEnabled
        captureQueue.async { [weak self] in
            self?.configureSessionLocked(cameraID: camID, targetDims: dims, allowSensorHeadroom: wantSensorHeadroom)
        }
    }

    private func configureSessionLocked(cameraID: String, targetDims: (width: Int, height: Int), allowSensorHeadroom: Bool) {
        if activeDeviceInput?.device.uniqueID == cameraID
            && configuredDimensions == targetDims
            && configuredSensorHeadroom == allowSensorHeadroom
            && !captureSession.inputs.isEmpty {
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

        // Stick directly to the downstream client's requested resolution (e.g. 1920x1080 for Teams/Meet)
        // unless Lossless Auto-Framing is explicitly toggled ON (where 4K sensor headroom keeps digital pan/zoom lossless!).
        let sensorCapW = Int32(allowSensorHeadroom ? max(targetDims.width, 3840) : targetDims.width)
        let sensorCapH = Int32(allowSensorHeadroom ? max(targetDims.height, 2160) : targetDims.height)
        selectBestDeviceFormat(device: device, targetWidth: sensorCapW, targetHeight: sensorCapH)

        configuredDimensions = targetDims
        configuredSensorHeadroom = allowSensorHeadroom
        compositor.resetTemporalHistory()
    }

    private func selectBestDeviceFormat(device: AVCaptureDevice, targetWidth: Int32, targetHeight: Int32) {
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
        let dims = resolvePipelineDimensions(for: settings.threadSafeSnapshot())
        bgMedia.setTargetDecodeDimensions(width: dims.width, height: dims.height)
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

    // MARK: - Hardware HEVC Recording

    @MainActor
    public func toggleRecording() {
        if isRecording { stopRecording() } else { startRecording() }
    }

    @MainActor
    private func startRecording() {
        let outDir = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Movies")
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let outURL = outDir.appendingPathComponent("StudioCam_\(stamp).mov")

        let dims = resolvePipelineDimensions(for: settings.threadSafeSnapshot())
        let w = dims.width
        let h = dims.height
        guard let writer = try? AVAssetWriter(outputURL: outURL, fileType: .mov) else { return }

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: w,
            AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: w >= 3840 ? 28_000_000 : 12_000_000
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
        let outDims = resolvePipelineDimensions(for: snap)
        frameIndex &+= 1

        // 1. ANE Person Segmentation + Face Tracking (for Studio Light, Skin Polish, and Auto-Framing)
        let needsSegmentation = (snap.backgroundMode != .none) || snap.portraitEnabled || snap.studioLightEnabled || snap.edgeLightEnabled
        let needsFace = snap.studioLightEnabled || snap.autoFramingEnabled || snap.skinPolishStrength > 0.01
        let visionPB = compositor.makeVisionInputBuffer(from: camPB, quality: snap.segmentationQuality)

        if needsSegmentation || needsFace {
            let desiredQuality: VNGeneratePersonSegmentationRequest.QualityLevel
            switch snap.segmentationQuality {
            case .fastVideoCall: desiredQuality = .fast
            case .balancedANE: desiredQuality = .balanced
            case .accurateMax: desiredQuality = .accurate
            }
            if segRequest.qualityLevel != desiredQuality {
                segRequest.qualityLevel = desiredQuality
            }

            var requests: [VNRequest] = []
            if needsSegmentation { requests.append(segRequest) }
            // Run face tracking at 1.5 Hz (every 20 frames) instead of 6 Hz to avoid Vision CPU spikes!
            if needsFace && (frameIndex % 20 == 1) { requests.append(faceRequest) }
            if snap.reactionsGestureEnabled && (frameIndex % 15 == 1) { requests.append(handRequest) }

            if !requests.isEmpty {
                let t0 = CACurrentMediaTime()
                try? visionSequenceHandler.perform(requests, on: visionPB, orientation: .up)
                let aneMs = (CACurrentMediaTime() - t0) * 1000.0
                if needsSegmentation {
                    smoothedANEMs = (smoothedANEMs == 0) ? aneMs : (smoothedANEMs * 0.88 + aneMs * 0.12)
                }
            }

            if needsSegmentation, let maskObs = segRequest.results?.first {
                cachedMaskBuffer = maskObs.pixelBuffer
            }
            if needsFace, let face = faceRequest.results?.first {
                let b = face.boundingBox
                let cx = Float(b.midX)
                let cy = Float(1.0 - b.midY)
                let rx = Float(max(0.11, b.width * 0.85))
                let ry = Float(max(0.16, b.height * 1.15))
                trackedFaceRect = trackedFaceRect * 0.72 + SIMD4<Float>(cx, cy, rx, ry) * 0.28
            }
            if snap.reactionsGestureEnabled {
                evaluateHandReactionGesture()
            }
        }

        // 2. Dead-Zone Tripod Damper for Lossless Cinema Auto-Framing
        let camCropRect = computeAutoFramingCropRect(snap: snap)

        // 3. Pull Virtual Background Frame & Update Lazy On-Chip ANE 3D Scene-Depth Map
        let (bgVideoPB, bgStaticTex) = bgMedia.currentBackgroundFrame()
        if snap.portraitEnabled && snap.depthBokehEnabled {
            if snap.backgroundMode == .none {
                // Analyze real physical room & angled hallway (excluding foreground person mask!)
                depthAnalyzer.maybeAnalyzeFrame(
                    pixelBuffer: visionPB,
                    foregroundMaskToExclude: cachedMaskBuffer,
                    sourceToken: "liveCam:\(CVPixelBufferGetWidth(camPB))x\(CVPixelBufferGetHeight(camPB))",
                    refreshIntervalSeconds: 4.5
                )
            } else if snap.backgroundMode == .video, let vPB = bgVideoPB {
                // Analyze virtual background video of a real room!
                depthAnalyzer.maybeAnalyzeFrame(
                    pixelBuffer: vPB,
                    foregroundMaskToExclude: nil,
                    sourceToken: "vid:\(bgMedia.currentSourceToken)",
                    refreshIntervalSeconds: 4.0
                )
            }
        }

        // 4. Single-pass Metal Composite at exact client-requested resolution
        let sinkActive = snap.virtualCamEnabled
        let sinkDims = sinkActive ? sinkBridge.activeDimensions : outDims
        guard let rendered = compositor.renderFrame(
            outputDimensions: outDims,
            cameraPixelBuffer: camPB,
            segmentationMaskBuffer: cachedMaskBuffer,
            backgroundVideoBuffer: bgVideoPB,
            backgroundStaticTexture: bgStaticTex,
            sceneDepthTexture: depthAnalyzer.depthTexture,
            faceRect: trackedFaceRect,
            cameraCropRect: camCropRect,
            settings: snap,
            produceSink: sinkActive,
            sinkDimensions: sinkDims,
            waitForGPUCompletion: true
        ) else { return }

        broadcastCompletedFrame(rendered.mainPixelBuffer)

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

        // 5. Throttled 2 Hz Telemetry Update
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
            Task { @MainActor [weak self] in
                guard let self = self, self.isPipelineRunning else { return }
                self.fpsTelemetry = fpsVal
                self.gpuMsTelemetry = gpuVal
                self.aneValAssign(aneVal)
                self.resolutionTelemetry = "\(camW)×\(camH) → \(outDims.width)×\(outDims.height)"
            }
        }
    }

    @MainActor private func aneValAssign(_ v: Double) {
        self.aneMsTelemetry = v
    }

    nonisolated private func computeAutoFramingCropRect(snap: RenderSettingsSnapshot) -> SIMD4<Float> {
        guard snap.autoFramingEnabled else {
            smoothedCropCenter = SIMD2<Float>(0.5, 0.5)
            smoothedCropScale = 1.0
            return SIMD4<Float>(0.0, 0.0, 1.0, 1.0)
        }

        let targetScale = 1.0 / max(1.05, min(1.85, snap.autoFramingZoom))
        // Desired framing places eyes/face comfortably in upper-middle third (y ~ 0.40)
        let desiredCenter = SIMD2<Float>(trackedFaceRect.x, trackedFaceRect.y + 0.05)

        // Tripod dead-zone (±4.5%): ignore small head nods; smoothly glide when user shifts position
        let delta = desiredCenter - tripodAnchorCenter
        if hypot(delta.x, delta.y) > 0.045 {
            tripodAnchorCenter = simd_mix(tripodAnchorCenter, desiredCenter, SIMD2<Float>(repeating: 0.08))
        }

        smoothedCropCenter = simd_mix(smoothedCropCenter, tripodAnchorCenter, SIMD2<Float>(repeating: 0.07))
        smoothedCropScale = smoothedCropScale * 0.92 + targetScale * 0.08

        let halfSpan = smoothedCropScale * 0.5
        let clampedCX = max(halfSpan, min(1.0 - halfSpan, smoothedCropCenter.x))
        let clampedCY = max(halfSpan, min(1.0 - halfSpan, smoothedCropCenter.y))
        return SIMD4<Float>(clampedCX - halfSpan, clampedCY - halfSpan, smoothedCropScale, smoothedCropScale)
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

        if indexTip.location.y > indexMCP.location.y + 0.08 && middleTip.location.y > middleMCP.location.y + 0.08 {
            lastReactionCooldownTime = now
            Task { @MainActor [weak self] in self?.settings.triggerReaction(3) }
        } else if thumbTip.location.y > thumbIP.location.y + 0.05 && indexTip.location.y < indexMCP.location.y + 0.02 {
            lastReactionCooldownTime = now
            Task { @MainActor [weak self] in self?.settings.triggerReaction(1) }
        }
    }
}
