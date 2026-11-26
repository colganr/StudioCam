import Foundation
import AppKit
import QuartzCore
import simd

public enum BackgroundMode: Int, Codable, CaseIterable, Identifiable, Sendable {
    case none = 0
    case gradient = 1
    case image = 2
    case video = 3

    public var id: Int { rawValue }
    public var label: String {
        switch self {
        case .none: return "Off"
        case .gradient: return "Gradient"
        case .image: return "Image"
        case .video: return "Video"
        }
    }
}

public enum CaptureResolutionPreset: String, Codable, CaseIterable, Identifiable, Sendable {
    case autoClient = "auto"
    case fhd1080 = "1920x1080"
    case uhd4K = "3840x2160"

    public var id: String { rawValue }
    public var shortLabel: String {
        switch self {
        case .autoClient: return "Auto"
        case .fhd1080: return "1080p"
        case .uhd4K: return "4K"
        }
    }
    public var title: String {
        switch self {
        case .autoClient: return "Auto (Match App)"
        case .fhd1080: return "1080p HD (1920×1080)"
        case .uhd4K: return "4K UHD (3840×2160)"
        }
    }
    public func resolvedDimensions(clientHint: (width: Int, height: Int)?) -> (width: Int, height: Int) {
        switch self {
        case .autoClient:
            if let c = clientHint, c.width > 0 && c.height > 0 {
                return c
            }
            return (1920, 1080)
        case .fhd1080:
            return (1920, 1080)
        case .uhd4K:
            return (3840, 2160)
        }
    }
}

public enum SegmentationQualityPreset: String, Codable, CaseIterable, Identifiable, Sendable {
    case fastVideoCall = "fast"
    case balancedANE = "balanced"
    case accurateMax = "accurate"

    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .fastVideoCall: return "Ultra-Light (Calls) + Guided Hair"
        case .balancedANE: return "Balanced ANE + Guided Hair"
        case .accurateMax: return "Max Neural + Guided Hair"
        }
    }
}

public struct StudioProfilePreset: Identifiable, Codable, Equatable, Sendable {
    public var id: String
    public var name: String
    public var resolution: CaptureResolutionPreset
    public var portraitEnabled: Bool
    public var portraitStrength: Float
    public var depthBokehEnabled: Bool
    public var focalPlaneOffset: Float
    public var studioLightEnabled: Bool
    public var studioLightStrength: Float
    public var edgeLightEnabled: Bool
    public var lightWrapStrength: Float
    public var ambientMatchStrength: Float
    public var autoFramingEnabled: Bool
    public var skinPolishStrength: Float
    public var filmGrainStrength: Float
    public var lutEnabled: Bool
    public var lutFilenameHint: String?
    public var backgroundMode: BackgroundMode

    public static let builtIns: [StudioProfilePreset] = [
        StudioProfilePreset(
            id: "executive_call",
            name: "Executive",
            resolution: .autoClient,
            portraitEnabled: true,
            portraitStrength: 0.62,
            depthBokehEnabled: true,
            focalPlaneOffset: 0.16,
            studioLightEnabled: true,
            studioLightStrength: 0.52,
            edgeLightEnabled: false,
            lightWrapStrength: 0.45,
            ambientMatchStrength: 0.35,
            autoFramingEnabled: false,
            skinPolishStrength: 0.25,
            filmGrainStrength: 0.14,
            lutEnabled: true,
            lutFilenameHint: "Broadcast_Clean_Rec709.cube",
            backgroundMode: .video
        ),
        StudioProfilePreset(
            id: "natural_hallway",
            name: "Real Room",
            resolution: .autoClient,
            portraitEnabled: true,
            portraitStrength: 0.72,
            depthBokehEnabled: true,
            focalPlaneOffset: 0.18,
            studioLightEnabled: true,
            studioLightStrength: 0.48,
            edgeLightEnabled: false,
            lightWrapStrength: 0.20,
            ambientMatchStrength: 0.0,
            autoFramingEnabled: false,
            skinPolishStrength: 0.22,
            filmGrainStrength: 0.10,
            lutEnabled: true,
            lutFilenameHint: "Broadcast_Clean_Rec709.cube",
            backgroundMode: .none
        ),
        StudioProfilePreset(
            id: "creator_4k",
            name: "Creator 4K",
            resolution: .uhd4K,
            portraitEnabled: true,
            portraitStrength: 0.68,
            depthBokehEnabled: true,
            focalPlaneOffset: 0.14,
            studioLightEnabled: true,
            studioLightStrength: 0.60,
            edgeLightEnabled: false,
            lightWrapStrength: 0.55,
            ambientMatchStrength: 0.45,
            autoFramingEnabled: true,
            skinPolishStrength: 0.28,
            filmGrainStrength: 0.20,
            lutEnabled: true,
            lutFilenameHint: "Warm_Cinema_TealAmber.cube",
            backgroundMode: .video
        )
    ]
}

public struct StudioGradientPreset: Identifiable, Equatable, Sendable {
    public let id: String
    public let name: String
    public let topRGB: SIMD4<Float>
    public let bottomRGB: SIMD4<Float>

    public static let presets: [StudioGradientPreset] = [
        .init(id: "apple_blue", name: "Ice Blue", topRGB: SIMD4(0.68, 0.79, 0.96, 1), bottomRGB: SIMD4(0.88, 0.93, 0.99, 1)),
        .init(id: "midnight_slate", name: "Studio Slate", topRGB: SIMD4(0.11, 0.14, 0.20, 1), bottomRGB: SIMD4(0.05, 0.06, 0.09, 1)),
        .init(id: "warm_amber", name: "Tube Amp Glow", topRGB: SIMD4(0.24, 0.14, 0.09, 1), bottomRGB: SIMD4(0.08, 0.05, 0.04, 1)),
        .init(id: "purple_haze", name: "Ultraviolet", topRGB: SIMD4(0.32, 0.22, 0.56, 1), bottomRGB: SIMD4(0.12, 0.09, 0.24, 1)),
        .init(id: "emerald_noir", name: "Emerald", topRGB: SIMD4(0.09, 0.26, 0.24, 1), bottomRGB: SIMD4(0.04, 0.10, 0.11, 1)),
        .init(id: "clean_paper", name: "Keynote Light", topRGB: SIMD4(0.94, 0.95, 0.97, 1), bottomRGB: SIMD4(0.82, 0.85, 0.90, 1))
    ]
}

public struct RenderSettingsSnapshot: Sendable {
    public var resolution: CaptureResolutionPreset
    public var segmentationQuality: SegmentationQualityPreset
    public var backgroundMode: BackgroundMode
    public var gradientPreset: StudioGradientPreset
    public var portraitEnabled: Bool
    public var portraitStrength: Float
    public var depthBokehEnabled: Bool
    public var focalPlaneOffset: Float
    public var studioLightEnabled: Bool
    public var studioLightStrength: Float
    public var edgeLightEnabled: Bool
    public var screenRingLightEnabled: Bool
    public var edgeLightStrength: Float
    public var edgeLightWarmth: Float
    public var spillSuppression: Float
    public var edgeSoftness: Float
    public var lightWrapStrength: Float
    public var ambientMatchStrength: Float
    public var autoFramingEnabled: Bool
    public var autoFramingZoom: Float
    public var skinPolishStrength: Float
    public var filmGrainStrength: Float
    public var mirrorPreview: Bool
    public var reactionsGestureEnabled: Bool
    public var virtualCamEnabled: Bool
    public var lutEnabled: Bool
    public var lutStrength: Float
    public var activeReactionType: UInt32
    public var activeReactionStart: TimeInterval
}

private final class SnapshotStorage: @unchecked Sendable {
    private let lock = NSLock()
    private var value: RenderSettingsSnapshot

    init(_ initial: RenderSettingsSnapshot) {
        self.value = initial
    }

    func read() -> RenderSettingsSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func update(_ body: (inout RenderSettingsSnapshot) -> Void) {
        lock.lock()
        body(&value)
        lock.unlock()
    }
}

@MainActor
public final class StudioSettings: ObservableObject {
    @Published public var selectedCameraUniqueID: String = ""
    @Published public var activeProfileID: String = "executive_call"
    // Default to Auto (Match Client) so 1080p Teams/Meet calls run natively at 1080p with zero 4K downsample!
    @Published public var resolution: CaptureResolutionPreset = .autoClient { didSet { syncSnapshot() } }
    @Published public var segmentationQuality: SegmentationQualityPreset = .fastVideoCall { didSet { syncSnapshot() } }

    @Published public var portraitEnabled: Bool = true { didSet { syncSnapshot() } }
    @Published public var portraitStrength: Float = 0.62 { didSet { syncSnapshot() } }
    @Published public var depthBokehEnabled: Bool = true { didSet { syncSnapshot() } }
    @Published public var focalPlaneOffset: Float = 0.16 { didSet { syncSnapshot() } }

    @Published public var studioLightEnabled: Bool = true { didSet { syncSnapshot() } }
    @Published public var studioLightStrength: Float = 0.55 { didSet { syncSnapshot() } }

    @Published public var edgeLightEnabled: Bool = false { didSet { syncSnapshot() } }
    @Published public var screenRingLightEnabled: Bool = true { didSet { syncSnapshot() } }
    @Published public var edgeLightStrength: Float = 0.60 { didSet { syncSnapshot() } }
    @Published public var edgeLightWarmth: Float = 0.35 { didSet { syncSnapshot() } }

    @Published public var backgroundMode: BackgroundMode = .gradient { didSet { syncSnapshot() } }
    @Published public var selectedGradientID: String = "apple_blue" { didSet { syncSnapshot() } }
    @Published public var customBackgroundURL: URL? = nil
    @Published public var recentBackgroundURLs: [URL] = []

    @Published public var spillSuppression: Float = 0.70 { didSet { syncSnapshot() } }
    @Published public var edgeSoftness: Float = 0.45 { didSet { syncSnapshot() } }
    @Published public var lightWrapStrength: Float = 0.45 { didSet { syncSnapshot() } }
    @Published public var ambientMatchStrength: Float = 0.35 { didSet { syncSnapshot() } }

    @Published public var autoFramingEnabled: Bool = false { didSet { syncSnapshot() } }
    @Published public var autoFramingZoom: Float = 1.22 { didSet { syncSnapshot() } }
    @Published public var skinPolishStrength: Float = 0.25 { didSet { syncSnapshot() } }
    @Published public var filmGrainStrength: Float = 0.14 { didSet { syncSnapshot() } }

    @Published public var mirrorPreview: Bool = false { didSet { syncSnapshot() } }
    @Published public var reactionsGestureEnabled: Bool = false { didSet { syncSnapshot() } }
    @Published public var virtualCamEnabled: Bool = false { didSet { syncSnapshot() } }

    @Published public var lutEnabled: Bool = false { didSet { syncSnapshot() } }
    @Published public var lutStrength: Float = 0.85 { didSet { syncSnapshot() } }
    @Published public var selectedLUTURL: URL? = nil
    @Published public var recentLUTURLs: [URL] = []

    @Published public var customUserProfile: StudioProfilePreset? = nil
    @Published public var expandedRow: String? = "background"
    @Published public var isDropTargeted: Bool = false

    private nonisolated let storage: SnapshotStorage

    public init() {
        let grad = StudioGradientPreset.presets[0]
        self.storage = SnapshotStorage(RenderSettingsSnapshot(
            resolution: .autoClient,
            segmentationQuality: .fastVideoCall,
            backgroundMode: .gradient,
            gradientPreset: grad,
            portraitEnabled: true,
            portraitStrength: 0.62,
            depthBokehEnabled: true,
            focalPlaneOffset: 0.16,
            studioLightEnabled: true,
            studioLightStrength: 0.55,
            edgeLightEnabled: false,
            screenRingLightEnabled: true,
            edgeLightStrength: 0.60,
            edgeLightWarmth: 0.35,
            spillSuppression: 0.70,
            edgeSoftness: 0.45,
            lightWrapStrength: 0.45,
            ambientMatchStrength: 0.35,
            autoFramingEnabled: false,
            autoFramingZoom: 1.22,
            skinPolishStrength: 0.25,
            filmGrainStrength: 0.14,
            mirrorPreview: false,
            reactionsGestureEnabled: false,
            virtualCamEnabled: false,
            lutEnabled: false,
            lutStrength: 0.85,
            activeReactionType: 0,
            activeReactionStart: 0
        ))
        loadPersisted()
        syncSnapshot()
    }

    public var currentGradient: StudioGradientPreset {
        StudioGradientPreset.presets.first(where: { $0.id == selectedGradientID }) ?? StudioGradientPreset.presets[0]
    }

    public func saveCurrentAsCustomProfile() {
        let p = StudioProfilePreset(
            id: "user_custom",
            name: "My Preset",
            resolution: resolution,
            portraitEnabled: portraitEnabled,
            portraitStrength: portraitStrength,
            depthBokehEnabled: depthBokehEnabled,
            focalPlaneOffset: focalPlaneOffset,
            studioLightEnabled: studioLightEnabled,
            studioLightStrength: studioLightStrength,
            edgeLightEnabled: edgeLightEnabled,
            lightWrapStrength: lightWrapStrength,
            ambientMatchStrength: ambientMatchStrength,
            autoFramingEnabled: autoFramingEnabled,
            skinPolishStrength: skinPolishStrength,
            filmGrainStrength: filmGrainStrength,
            lutEnabled: lutEnabled,
            lutFilenameHint: selectedLUTURL?.lastPathComponent,
            backgroundMode: backgroundMode
        )
        customUserProfile = p
        activeProfileID = p.id
        if let data = try? JSONEncoder().encode(p) {
            UserDefaults.standard.set(data, forKey: "sc.customProfile")
        }
    }

    public func triggerReaction(_ type: UInt32) {
        storage.update { s in
            s.activeReactionType = type
            s.activeReactionStart = CACurrentMediaTime()
        }
    }

    public nonisolated func threadSafeSnapshot() -> RenderSettingsSnapshot {
        storage.read()
    }

    public func addRecentURL(_ url: URL) {
        recentBackgroundURLs.removeAll(where: { $0.path == url.path })
        recentBackgroundURLs.insert(url, at: 0)
        if recentBackgroundURLs.count > 8 {
            recentBackgroundURLs = Array(recentBackgroundURLs.prefix(8))
        }
        savePersisted()
    }

    public func addRecentLUTURL(_ url: URL) {
        recentLUTURLs.removeAll(where: { $0.path == url.path })
        recentLUTURLs.insert(url, at: 0)
        if recentLUTURLs.count > 8 {
            recentLUTURLs = Array(recentLUTURLs.prefix(8))
        }
        savePersisted()
    }

    private func syncSnapshot() {
        let grad = currentGradient
        storage.update { s in
            s.resolution = resolution
            s.segmentationQuality = segmentationQuality
            s.backgroundMode = backgroundMode
            s.gradientPreset = grad
            s.portraitEnabled = portraitEnabled
            s.portraitStrength = portraitStrength
            s.depthBokehEnabled = depthBokehEnabled
            s.focalPlaneOffset = focalPlaneOffset
            s.studioLightEnabled = studioLightEnabled
            s.studioLightStrength = studioLightStrength
            s.edgeLightEnabled = edgeLightEnabled
            s.screenRingLightEnabled = screenRingLightEnabled
            s.edgeLightStrength = edgeLightStrength
            s.edgeLightWarmth = edgeLightWarmth
            s.spillSuppression = spillSuppression
            s.edgeSoftness = edgeSoftness
            s.lightWrapStrength = lightWrapStrength
            s.ambientMatchStrength = ambientMatchStrength
            s.autoFramingEnabled = autoFramingEnabled
            s.autoFramingZoom = autoFramingZoom
            s.skinPolishStrength = skinPolishStrength
            s.filmGrainStrength = filmGrainStrength
            s.mirrorPreview = mirrorPreview
            s.reactionsGestureEnabled = reactionsGestureEnabled
            s.virtualCamEnabled = virtualCamEnabled
            s.lutEnabled = lutEnabled
            s.lutStrength = lutStrength
        }
        savePersisted()
    }

    private func savePersisted() {
        let d = UserDefaults.standard
        d.set(resolution.rawValue, forKey: "sc.resolutionV2")
        d.set(segmentationQuality.rawValue, forKey: "sc.segQualityV2")
        d.set(portraitEnabled, forKey: "sc.portrait")
        d.set(portraitStrength, forKey: "sc.portraitStrength")
        d.set(depthBokehEnabled, forKey: "sc.depthBokeh")
        d.set(focalPlaneOffset, forKey: "sc.focalOffset")
        d.set(studioLightEnabled, forKey: "sc.studioLight")
        d.set(studioLightStrength, forKey: "sc.studioLightStrength")
        d.set(edgeLightEnabled, forKey: "sc.edgeLight")
        d.set(screenRingLightEnabled, forKey: "sc.screenRingLight")
        d.set(edgeLightStrength, forKey: "sc.edgeLightStrength")
        d.set(edgeLightWarmth, forKey: "sc.edgeLightWarmth")
        d.set(backgroundMode.rawValue, forKey: "sc.bgMode")
        d.set(selectedGradientID, forKey: "sc.gradID")
        d.set(customBackgroundURL?.path, forKey: "sc.customBG")
        d.set(recentBackgroundURLs.map(\.path), forKey: "sc.recentBGs")
        d.set(lightWrapStrength, forKey: "sc.lightWrap")
        d.set(ambientMatchStrength, forKey: "sc.ambientMatch")
        d.set(autoFramingEnabled, forKey: "sc.autoFrame")
        d.set(autoFramingZoom, forKey: "sc.autoFrameZoom")
        d.set(skinPolishStrength, forKey: "sc.skinPolish")
        d.set(filmGrainStrength, forKey: "sc.filmGrain")
        d.set(lutEnabled, forKey: "sc.lutEnabled")
        d.set(lutStrength, forKey: "sc.lutStrength")
        d.set(selectedLUTURL?.path, forKey: "sc.selectedLUT")
        d.set(recentLUTURLs.map(\.path), forKey: "sc.recentLUTs")
        d.set(spillSuppression, forKey: "sc.spill")
        d.set(edgeSoftness, forKey: "sc.softness")
    }

    private func loadPersisted() {
        let d = UserDefaults.standard
        if let r = d.string(forKey: "sc.resolutionV2"), let p = CaptureResolutionPreset(rawValue: r) { resolution = p }
        if let q = d.string(forKey: "sc.segQualityV2"), let p = SegmentationQualityPreset(rawValue: q) { segmentationQuality = p }
        if d.object(forKey: "sc.portrait") != nil { portraitEnabled = d.bool(forKey: "sc.portrait") }
        if d.object(forKey: "sc.portraitStrength") != nil { portraitStrength = d.float(forKey: "sc.portraitStrength") }
        if d.object(forKey: "sc.depthBokeh") != nil { depthBokehEnabled = d.bool(forKey: "sc.depthBokeh") }
        if d.object(forKey: "sc.focalOffset") != nil { focalPlaneOffset = d.float(forKey: "sc.focalOffset") }
        if d.object(forKey: "sc.studioLight") != nil { studioLightEnabled = d.bool(forKey: "sc.studioLight") }
        if d.object(forKey: "sc.studioLightStrength") != nil { studioLightStrength = d.float(forKey: "sc.studioLightStrength") }
        if d.object(forKey: "sc.edgeLight") != nil { edgeLightEnabled = d.bool(forKey: "sc.edgeLight") }
        if d.object(forKey: "sc.screenRingLight") != nil { screenRingLightEnabled = d.bool(forKey: "sc.screenRingLight") }
        if d.object(forKey: "sc.edgeLightStrength") != nil { edgeLightStrength = d.float(forKey: "sc.edgeLightStrength") }
        if d.object(forKey: "sc.edgeLightWarmth") != nil { edgeLightWarmth = d.float(forKey: "sc.edgeLightWarmth") }
        if d.object(forKey: "sc.bgMode") != nil, let m = BackgroundMode(rawValue: d.integer(forKey: "sc.bgMode")) { backgroundMode = m }
        if let g = d.string(forKey: "sc.gradID") { selectedGradientID = g }
        if let p = d.string(forKey: "sc.customBG"), FileManager.default.fileExists(atPath: p) {
            customBackgroundURL = URL(fileURLWithPath: p)
        }
        if let arr = d.stringArray(forKey: "sc.recentBGs") {
            recentBackgroundURLs = arr.filter { FileManager.default.fileExists(atPath: $0) }.map { URL(fileURLWithPath: $0) }
        }
        if d.object(forKey: "sc.lightWrap") != nil { lightWrapStrength = d.float(forKey: "sc.lightWrap") }
        if d.object(forKey: "sc.ambientMatch") != nil { ambientMatchStrength = d.float(forKey: "sc.ambientMatch") }
        if d.object(forKey: "sc.autoFrame") != nil { autoFramingEnabled = d.bool(forKey: "sc.autoFrame") }
        if d.object(forKey: "sc.autoFrameZoom") != nil { autoFramingZoom = d.float(forKey: "sc.autoFrameZoom") }
        if d.object(forKey: "sc.skinPolish") != nil { skinPolishStrength = d.float(forKey: "sc.skinPolish") }
        if d.object(forKey: "sc.filmGrain") != nil { filmGrainStrength = d.float(forKey: "sc.filmGrain") }
        if d.object(forKey: "sc.lutEnabled") != nil { lutEnabled = d.bool(forKey: "sc.lutEnabled") }
        if d.object(forKey: "sc.lutStrength") != nil { lutStrength = d.float(forKey: "sc.lutStrength") }
        if let lp = d.string(forKey: "sc.selectedLUT"), FileManager.default.fileExists(atPath: lp) {
            selectedLUTURL = URL(fileURLWithPath: lp)
        }
        if let larr = d.stringArray(forKey: "sc.recentLUTs") {
            recentLUTURLs = larr.filter { FileManager.default.fileExists(atPath: $0) }.map { URL(fileURLWithPath: $0) }
        }
        if d.object(forKey: "sc.spill") != nil { spillSuppression = d.float(forKey: "sc.spill") }
        if d.object(forKey: "sc.softness") != nil { edgeSoftness = d.float(forKey: "sc.softness") }
        if let cData = d.data(forKey: "sc.customProfile"),
           let cp = try? JSONDecoder().decode(StudioProfilePreset.self, from: cData) {
            customUserProfile = cp
        }
    }
}
