import Foundation
import AVFoundation
import CoreVideo
import Metal
import MetalKit
import ImageIO
import UniformTypeIdentifiers

/// Hardware-accelerated Media Engine video & static-image background provider.
/// Decodes H.264, HEVC (8/10-bit), and ProRes up to 4K60 directly into IOSurface-backed
/// NV12 CVPixelBuffers with zero CPU pixel copies, and auto-pauses when idle.
public final class BackgroundMediaController {
    public static let supportedVideoExtensions: Set<String> = ["mp4", "mov", "m4v", "hevc"]
    public static let supportedImageExtensions: Set<String> = ["png", "jpg", "jpeg", "heic", "heif", "webp", "tiff"]

    private let device: MTLDevice
    private let lock = NSLock()

    private var queuePlayer: AVQueuePlayer?
    private var playerLooper: AVPlayerLooper?
    private var videoOutput: AVPlayerItemVideoOutput?
    private var latestVideoPixelBuffer: CVPixelBuffer?
    private var staticImageTexture: MTLTexture?
    private var currentURL: URL?
    private var isVideoActive: Bool = false
    private var shouldPlay: Bool = false

    public private(set) var mediaSummary: String = "None"

    public init(device: MTLDevice) {
        self.device = device
    }

    public static func isVideoURL(_ url: URL) -> Bool {
        supportedVideoExtensions.contains(url.pathExtension.lowercased())
    }

    /// Loads either a video file (up to 4K60, looped on Apple Media Engine) or a static image (cached once on GPU).
    public func loadMedia(from url: URL, completion: ((Bool, String) -> Void)? = nil) {
        let ext = url.pathExtension.lowercased()
        if Self.supportedVideoExtensions.contains(ext) {
            loadVideo(from: url, completion: completion)
        } else {
            loadStaticImage(from: url, completion: completion)
        }
    }

    private func loadStaticImage(from url: URL, completion: ((Bool, String) -> Void)?) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let cgImg = CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary) else {
                completion?(false, "Could not decode image \(url.lastPathComponent)")
                return
            }

            // Cap static image texture to 3840x2160 max to avoid unnecessary VRAM waste
            let maxW = 3840
            let maxH = 2160
            var w = cgImg.width
            var h = cgImg.height
            if w > maxW || h > maxH {
                let scale = min(Double(maxW) / Double(w), Double(maxH) / Double(h))
                w = max(2, Int(Double(w) * scale))
                h = max(2, Int(Double(h) * scale))
            }

            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
            desc.usage = [.shaderRead]
            guard let tex = self.device.makeTexture(descriptor: desc) else {
                completion?(false, "Failed to allocate GPU texture")
                return
            }

            let bytesPerRow = w * 4
            var raw = [UInt8](repeating: 0, count: bytesPerRow * h)
            let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            raw.withUnsafeMutableBytes { ptr in
                if let ctx = CGContext(
                    data: ptr.baseAddress,
                    width: w,
                    height: h,
                    bitsPerComponent: 8,
                    bytesPerRow: bytesPerRow,
                    space: cs,
                    bitmapInfo: bitmapInfo
                ) {
                    ctx.interpolationQuality = .high
                    ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: w, height: h))
                }
            }
            tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: raw, bytesPerRow: bytesPerRow)

            self.lock.lock()
            self.teardownVideoLocked()
            self.staticImageTexture = tex
            self.currentURL = url
            self.isVideoActive = false
            let summary = "\(url.lastPathComponent) (\(w)×\(h) Image)"
            self.mediaSummary = summary
            self.lock.unlock()
            completion?(true, summary)
        }
    }

    private func loadVideo(from url: URL, completion: ((Bool, String) -> Void)?) {
        let asset = AVURLAsset(url: url, options: [AVURLAssetPreferPreciseDurationAndTimingKey: false])
        Task { [weak self] in
            guard let self = self else { return }
            do {
                let videoTracks = try await asset.loadTracks(withMediaType: .video)
                guard let vTrack = videoTracks.first else {
                    completion?(false, "No video track in \(url.lastPathComponent)")
                    return
                }
                let natSize = try await vTrack.load(.naturalSize)
                let nominalFPS = try await vTrack.load(.nominalFrameRate)
                let w = Int(abs(natSize.width))
                let h = Int(abs(natSize.height))

                // Request Media Engine NV12 IOSurface output (zero CPU conversion)
                let outAttrs: [String: Any] = [
                    kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
                    kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                    kCVPixelBufferMetalCompatibilityKey as String: true
                ]
                let output = AVPlayerItemVideoOutput(pixelBufferAttributes: outAttrs)
                output.suppressesPlayerRendering = true

                let templateItem = AVPlayerItem(asset: asset)
                templateItem.add(output)

                let player = AVQueuePlayer()
                player.isMuted = true
                player.automaticallyWaitsToMinimizeStalling = false
                player.preventsDisplaySleepDuringVideoPlayback = false

                let looper = AVPlayerLooper(player: player, templateItem: templateItem)

                // Ensure any looped items attach our output & disable audio decoding
                let summary = String(format: "%@ (%d×%d @ %.0ffps)", url.lastPathComponent, w, h, nominalFPS > 0 ? nominalFPS : 30)
                let playNow = self.installNewVideoPlayer(player: player, looper: looper, output: output, url: url, summary: summary)

                if playNow {
                    await MainActor.run { player.play() }
                }
                completion?(true, summary)
            } catch {
                completion?(false, "Video load error: \(error.localizedDescription)")
            }
        }
    }

    public func setPlaybackActive(_ active: Bool) {
        lock.lock()
        shouldPlay = active
        let player = queuePlayer
        let videoMode = isVideoActive
        lock.unlock()

        guard let player = player, videoMode else { return }
        DispatchQueue.main.async {
            if active {
                if player.rate == 0 { player.play() }
            } else {
                if player.rate != 0 { player.pause() }
            }
        }
    }

    /// Non-blocking fetch of current background frame for the Metal compositor.
    public func currentBackgroundFrame() -> (pixelBuffer: CVPixelBuffer?, staticTexture: MTLTexture?) {
        lock.lock()
        defer { lock.unlock() }

        if !isVideoActive {
            return (nil, staticImageTexture)
        }

        // Ensure AVPlayerLooper's current item always carries our videoOutput
        if let currentItem = queuePlayer?.currentItem, let out = videoOutput {
            if !currentItem.outputs.contains(out) {
                currentItem.add(out)
                for track in currentItem.tracks where track.assetTrack?.mediaType == .audio {
                    track.isEnabled = false
                }
            }
            let itemTime = out.itemTime(forHostTime: CACurrentMediaTime())
            if itemTime.isValid && out.hasNewPixelBuffer(forItemTime: itemTime) {
                if let pb = out.copyPixelBuffer(forItemTime: itemTime, itemTimeForDisplay: nil) {
                    latestVideoPixelBuffer = pb
                }
            }
        }
        return (latestVideoPixelBuffer, nil)
    }

    private func installNewVideoPlayer(
        player: AVQueuePlayer,
        looper: AVPlayerLooper,
        output: AVPlayerItemVideoOutput,
        url: URL,
        summary: String
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        teardownVideoLocked()
        staticImageTexture = nil
        queuePlayer = player
        playerLooper = looper
        videoOutput = output
        currentURL = url
        isVideoActive = true
        mediaSummary = summary
        return shouldPlay
    }

    private func teardownVideoLocked() {
        if let p = queuePlayer {
            DispatchQueue.main.async { p.pause() }
        }
        playerLooper?.disableLooping()
        playerLooper = nil
        queuePlayer = nil
        videoOutput = nil
        latestVideoPixelBuffer = nil
    }
}
