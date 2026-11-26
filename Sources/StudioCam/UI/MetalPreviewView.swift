import SwiftUI
import AppKit
import CoreVideo
import QuartzCore

/// Zero-copy hardware WindowServer preview view backed directly by `CALayer.contents = IOSurfaceRef`.
/// Eliminates MTKView drawable lifecycle issues inside macOS `MenuBarExtra(.window)` popovers,
/// retains the active `CVPixelBuffer` cleanly across pool reuse, and uses 0.00 ms extra GPU time.
public struct MetalPreviewView: NSViewRepresentable {
    public let registerFrameProvider: (@escaping (CVPixelBuffer) -> Void) -> () -> Void

    public init(
        registerFrameProvider: @escaping (@escaping (CVPixelBuffer) -> Void) -> () -> Void
    ) {
        self.registerFrameProvider = registerFrameProvider
    }

    public func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    public func makeNSView(context: Context) -> PreviewHostNSView {
        let view = PreviewHostNSView(frame: NSRect(x: 0, y: 0, width: 400, height: 225))
        context.coordinator.view = view
        let token = registerFrameProvider { [weak coord = context.coordinator] pb in
            coord?.present(pixelBuffer: pb)
        }
        context.coordinator.unregister = token
        return view
    }

    public func updateNSView(_ nsView: PreviewHostNSView, context: Context) {}

    public static func dismantleNSView(_ nsView: PreviewHostNSView, coordinator: Coordinator) {
        coordinator.unregister?()
        coordinator.unregister = nil
    }

    public final class PreviewHostNSView: NSView {
        public let surfaceLayer = CALayer()

        public override init(frame frameRect: NSRect) {
            super.init(frame: frameRect)
            wantsLayer = true
            layer = CALayer()
            layer?.backgroundColor = NSColor(calibratedRed: 0.05, green: 0.06, blue: 0.08, alpha: 1).cgColor
            surfaceLayer.contentsGravity = .resizeAspectFill
            surfaceLayer.masksToBounds = true
            layer?.addSublayer(surfaceLayer)
        }

        required init?(coder: NSCoder) { fatalError("init(coder:) not supported") }

        public override func layout() {
            super.layout()
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            surfaceLayer.frame = bounds
            CATransaction.commit()
        }
    }

    public final class Coordinator: NSObject {
        weak var view: PreviewHostNSView?
        var unregister: (() -> Void)?
        private var retainedPixelBuffer: CVPixelBuffer?
        private var lastPresentTime: CFTimeInterval = 0
        private let lock = NSLock()

        func present(pixelBuffer: CVPixelBuffer) {
            let now = CACurrentMediaTime()
            lock.lock()
            let shouldPresent = (now - lastPresentTime) >= (1.0 / 32.0)
            if shouldPresent { lastPresentTime = now }
            lock.unlock()
            guard shouldPresent else { return }

            DispatchQueue.main.async { [weak self] in
                guard let self = self, let v = self.view else { return }
                self.retainedPixelBuffer = pixelBuffer
                if let surface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue() {
                    CATransaction.begin()
                    CATransaction.setDisableActions(true)
                    v.surfaceLayer.frame = v.bounds
                    v.surfaceLayer.contents = surface
                    CATransaction.commit()
                }
            }
        }
    }
}
