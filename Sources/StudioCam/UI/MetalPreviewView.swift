import SwiftUI
import MetalKit
import MetalPerformanceShaders
import QuartzCore

/// Zero-CPU Metal preview view. Blits/scales the composited GPU texture directly into the CAMetalDrawable
/// on demand without copying pixels through CoreGraphics or CPU RAM.
public struct MetalPreviewView: NSViewRepresentable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let registerFrameProvider: (@escaping (MTLTexture) -> Void) -> Void

    public init(
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        registerFrameProvider: @escaping (@escaping (MTLTexture) -> Void) -> Void
    ) {
        self.device = device
        self.commandQueue = commandQueue
        self.registerFrameProvider = registerFrameProvider
    }

    public func makeCoordinator() -> Coordinator {
        Coordinator(device: device, commandQueue: commandQueue)
    }

    public func makeNSView(context: Context) -> MTKView {
        let mtk = MTKView(frame: .zero, device: device)
        mtk.colorPixelFormat = .bgra8Unorm
        mtk.framebufferOnly = false
        mtk.isPaused = true
        mtk.enableSetNeedsDisplay = false
        mtk.clearColor = MTLClearColorMake(0.05, 0.05, 0.07, 1.0)
        mtk.delegate = context.coordinator
        context.coordinator.view = mtk

        registerFrameProvider { [weak coord = context.coordinator] tex in
            coord?.present(texture: tex)
        }
        return mtk
    }

    public func updateNSView(_ nsView: MTKView, context: Context) {}

    public final class Coordinator: NSObject, MTKViewDelegate {
        private let commandQueue: MTLCommandQueue
        private let lock = NSLock()
        private var latestTexture: MTLTexture?
        private var lastDrawTime: CFTimeInterval = 0
        weak var view: MTKView?

        init(device: MTLDevice, commandQueue: MTLCommandQueue) {
            self.commandQueue = commandQueue
            super.init()
        }

        func present(texture: MTLTexture) {
            let now = CACurrentMediaTime()
            lock.lock()
            latestTexture = texture
            // Cap preview UI draws at 30 Hz for featherweight WindowServer impact
            let shouldDraw = (now - lastDrawTime) >= (1.0 / 31.0)
            if shouldDraw { lastDrawTime = now }
            lock.unlock()

            if shouldDraw {
                DispatchQueue.main.async { [weak self] in
                    guard let v = self?.view, v.window?.isVisible == true else { return }
                    v.draw()
                }
            }
        }

        public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        public func draw(in view: MTKView) {
            lock.lock()
            let srcTex = latestTexture
            lock.unlock()

            guard let src = srcTex,
                  let drawable = view.currentDrawable,
                  let cmd = commandQueue.makeCommandBuffer() else { return }

            let dst = drawable.texture
            // Use MetalPerformanceShaders Lanczos / Bilinear scale for crisp retina preview
            let scale = MPSImageBilinearScale(device: view.device!)
            let sx = Double(dst.width) / Double(src.width)
            let sy = Double(dst.height) / Double(src.height)
            let s = max(sx, sy)
            let tx = (Double(dst.width) - Double(src.width) * s) * 0.5
            let ty = (Double(dst.height) - Double(src.height) * s) * 0.5
            var transform = MPSScaleTransform(scaleX: s, scaleY: s, translateX: tx, translateY: ty)
            withUnsafePointer(to: &transform) { ptr in
                scale.scaleTransform = ptr
                scale.encode(commandBuffer: cmd, sourceTexture: src, destinationTexture: dst)
            }
            cmd.present(drawable)
            cmd.commit()
        }
    }
}
