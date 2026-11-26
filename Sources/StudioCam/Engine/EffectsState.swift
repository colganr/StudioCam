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
        case .video: return "4K Video"
        }
    }
}

public enum CaptureResolutionPreset: String, Codable, CaseIterable, Identifiable, Sendable {
    case uhd4K = "3840x2160"
    case fhd1080 = "1920x1080"

    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .uhd4K: return "4K UHD (3840×2160)"
        case .fhd1080: return "1080p HD (1920×1080)"
        }
    }
    public var width: Int { self == .uhd4K ? 3840 : 1920 }
    public var height: Int { self == .uhd4K ? 2160 : 1080 }
}

public enum SegmentationQualityPreset: String, Codable, CaseIterable, Identifiable, Sendable {
    case balancedANE = "balanced"
    case accurateMax = "accurate"

    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .balancedANE: return "Fast ANE + 4K Guided Hair"
        case .accurateMax: return "Max Neural + 4K Guided Hair"
        }
    }
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
    public var studioLightEnabled: Bool
    public var studioLightStrength: Float
    public var edgeLightEnabled: Bool
    public var screenRingLightEnabled: Bool
    public var edgeLightStrength: Float
    public var edgeLightWarmth: Float
    public var spillSuppression: Float
    public var edgeSoftness: Float
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
    @Published public var resolution: CaptureResolutionPreset = .uhd4K { didSet { syncSnapshot() } }
    @Published public var segmentationQuality: SegmentationQualityPreset = .balancedANE { didSet { syncSnapshot() } }

    @Published public var portraitEnabled: Bool = true { didSet { syncSnapshot() } }
    @Published public var portraitStrength: Float = 0.55 { didSet { syncSnapshot() } }

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
    @Published public var mirrorPreview: Bool = false { didSet { syncSnapshot() } }

    @Published public var reactionsGestureEnabled: Bool = false { didSet { syncSnapshot() } }
    @Published public var virtualCamEnabled: Bool = false { didSet { syncSnapshot() } }
    @Published public var lutEnabled: Bool = false { didSet { syncSnapshot() } }
    @Published public var lutStrength: Float = 0.85 { didSet { syncSnapshot() } }
    @Published public var selectedLUTURL: URL? = nil
    @Published public var recentLUTURLs: [URL] = []
    @Published public var expandedRow: String? = "background"
    @Published public var isDropTargeted: Bool = false

    private nonisolated let storage: SnapshotStorage

    public init() {
        let grad = StudioGradientPreset.presets[0]
        self.storage = SnapshotStorage(RenderSettingsSnapshot(
            resolution: .uhd4K,
            segmentationQuality: .balancedANE,
            backgroundMode: .gradient,
            gradientPreset: grad,
            portraitEnabled: true,
            portraitStrength: 0.55,
            studioLightEnabled: true,
            studioLightStrength: 0.55,
            edgeLightEnabled: false,
            screenRingLightEnabled: true,
            edgeLightStrength: 0.60,
            edgeLightWarmth: 0.35,
            spillSuppression: 0.70,
            edgeSoftness: 0.45,
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
            s.studioLightEnabled = studioLightEnabled
            s.studioLightStrength = studioLightStrength
            s.edgeLightEnabled = edgeLightEnabled
            s.screenRingLightEnabled = screenRingLightEnabled
            s.edgeLightStrength = edgeLightStrength
            s.edgeLightWarmth = edgeLightWarmth
            s.spillSuppression = spillSuppression
            s.edgeSoftness = edgeSoftness
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
        d.set(resolution.rawValue, forKey: "sc.resolution")
        d.set(segmentationQuality.rawValue, forKey: "sc.segQuality")
        d.set(portraitEnabled, forKey: "sc.portrait")
        d.set(portraitStrength, forKey: "sc.portraitStrength")
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
        d.set(lutEnabled, forKey: "sc.lutEnabled")
        d.set(lutStrength, forKey: "sc.lutStrength")
        d.set(selectedLUTURL?.path, forKey: "sc.selectedLUT")
        d.set(recentLUTURLs.map(\.path), forKey: "sc.recentLUTs")
        d.set(spillSuppression, forKey: "sc.spill")
        d.set(edgeSoftness, forKey: "sc.softness")
    }

    private func loadPersisted() {
        let d = UserDefaults.standard
        if let r = d.string(forKey: "sc.resolution"), let p = CaptureResolutionPreset(rawValue: r) { resolution = p }
        if let q = d.string(forKey: "sc.segQuality"), let p = SegmentationQualityPreset(rawValue: q) { segmentationQuality = p }
        if d.object(forKey: "sc.portrait") != nil { portraitEnabled = d.bool(forKey: "sc.portrait") }
        if d.object(forKey: "sc.portraitStrength") != nil { portraitStrength = d.float(forKey: "sc.portraitStrength") }
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
    }
}
