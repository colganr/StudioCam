import AppKit
import SwiftUI

/// Replicates macOS `RingLightHelper.xpc` ("Edge Light"):
/// Renders a click-through rounded luminous border around the active display to physically light
/// the user's face in dim rooms, while punching a soft transparent cutout around the mouse cursor.
@MainActor
public final class ScreenRingLightController {
    private var panel: NSPanel?
    private var model = RingLightViewModel()
    private var timer: Timer?

    public init() {}

    public func update(enabled: Bool, intensity: Float, warmth: Float) {
        model.intensity = Double(intensity)
        model.warmth = Double(warmth)

        if enabled {
            showIfNeeded()
        } else {
            hide()
        }
    }

    private func showIfNeeded() {
        guard panel == nil, let screen = NSScreen.main ?? NSScreen.screens.first else { return }

        let p = NSPanel(
            contentRect: screen.frame,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        p.level = .statusBar
        p.backgroundColor = .clear
        p.isOpaque = false
        p.hasShadow = false
        p.ignoresMouseEvents = true
        p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        p.contentView = NSHostingView(rootView: RingLightOverlayView(model: model))
        p.orderFrontRegardless()
        self.panel = p

        // Lightweight 20 Hz cursor proximity tracker (runs ONLY while Screen Ring Light is visible)
        self.timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 20.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self = self, let win = self.panel else { return }
                let mouse = NSEvent.mouseLocation
                let f = win.frame
                let nx = (mouse.x - f.minX) / max(1, f.width)
                let ny = 1.0 - (mouse.y - f.minY) / max(1, f.height)
                self.model.cursorUV = CGPoint(x: nx, y: ny)
            }
        }
    }

    public func hide() {
        timer?.invalidate()
        timer = nil
        panel?.orderOut(nil)
        panel = nil
    }
}

@MainActor
private final class RingLightViewModel: ObservableObject {
    @Published var intensity: Double = 0.6
    @Published var warmth: Double = 0.35
    @Published var cursorUV: CGPoint = CGPoint(x: 0.5, y: 0.5)
}

private struct RingLightOverlayView: View {
    @ObservedObject var model: RingLightViewModel

    var ringColor: Color {
        // Blend between crisp studio 5600K daylight and warm 3800K key light
        let r = 0.94 + 0.06 * model.warmth
        let g = 0.97 - 0.08 * model.warmth
        let b = 1.00 - 0.24 * model.warmth
        return Color(red: r, green: g, blue: b)
    }

    var body: some View {
        GeometryReader { geo in
            let borderWidth = CGFloat(28.0 + 38.0 * model.intensity)
            let cursorPt = CGPoint(x: model.cursorUV.x * geo.size.width, y: model.cursorUV.y * geo.size.height)

            ZStack {
                RoundedRectangle(cornerRadius: 42, style: .continuous)
                    .strokeBorder(ringColor.opacity(0.35 + 0.60 * model.intensity), lineWidth: borderWidth)
                    .blur(radius: 14)
                    .padding(6)
            }
            .compositingGroup()
            .mask(
                ZStack {
                    Color.white
                    // Punch a smooth transparent hole around the cursor so menus/buttons stay unobstructed
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [.black, .black.opacity(0.6), .clear],
                                center: .center,
                                startRadius: 36,
                                endRadius: 135
                            )
                        )
                        .frame(width: 270, height: 270)
                        .position(cursorPt)
                        .blendMode(.destinationOut)
                }
                .compositingGroup()
            )
        }
        .ignoresSafeArea()
    }
}
