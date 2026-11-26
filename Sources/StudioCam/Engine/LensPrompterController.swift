import AppKit
import SwiftUI

/// Under-webcam floating talking-points teleprompter pinned right beneath the MacBook notch / top webcam.
/// Uses `panel.sharingType = .none` so Zoom, Teams, Meet, and OBS screen sharing NEVER see this window!
@MainActor
public final class LensPrompterController: ObservableObject {
    @Published public var isVisible: Bool = false { didSet { syncWindow() } }
    @Published public var isAutoScrolling: Bool = false
    @Published public var scrollSpeedPtsPerSec: Double = 22.0
    @Published public var fontSize: Double = 16.0
    @Published public var backgroundOpacity: Double = 0.78
    @Published public var scriptText: String = """
    • Promise first: state the main takeaway in one sentence.
    • Highlight customer impact & verified milestones.
    • Confirm owners and dates on final action-item readback.
    """ {
        didSet { UserDefaults.standard.set(scriptText, forKey: "sc.prompterText") }
    }

    private var panel: NSPanel?

    public init() {
        if let saved = UserDefaults.standard.string(forKey: "sc.prompterText"), !saved.isEmpty {
            self.scriptText = saved
        }
    }

    private func syncWindow() {
        if isVisible {
            showPanel()
        } else {
            panel?.orderOut(nil)
            panel = nil
            isAutoScrolling = false
        }
    }

    private func showPanel() {
        guard panel == nil, let screen = NSScreen.main ?? NSScreen.screens.first else { return }
        let width: CGFloat = 460
        let height: CGFloat = 148
        let visFrame = screen.visibleFrame
        let x = screen.frame.midX - width * 0.5
        let y = visFrame.maxY - height - 4

        let p = NSPanel(
            contentRect: NSRect(x: x, y: y, width: width, height: height),
            styleMask: [.titled, .closable, .resizable, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        p.titleVisibility = .hidden
        p.titlebarAppearsTransparent = true
        p.isFloatingPanel = true
        p.level = .statusBar
        p.backgroundColor = .clear
        p.isOpaque = false
        p.hasShadow = true
        p.isMovableByWindowBackground = true
        // Invisible to Zoom / Teams / Meet screen sharing!
        p.sharingType = .none
        p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        p.contentView = NSHostingView(rootView: LensPrompterHUDView(controller: self))
        p.orderFrontRegardless()
        self.panel = p
    }
}

private struct LensPrompterHUDView: View {
    @ObservedObject var controller: LensPrompterController
    @StateObject private var scroller = AutoScrollDriver()

    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 8) {
                Image(systemName: "eye.circle.fill")
                    .foregroundStyle(.cyan)
                Text("Lens Prompter (Hidden from Screen Share)")
                    .font(.system(size: 10.5, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.75))
                Spacer()
                Button {
                    controller.isAutoScrolling.toggle()
                    scroller.setScrolling(controller.isAutoScrolling, speed: controller.scrollSpeedPtsPerSec)
                } label: {
                    Image(systemName: controller.isAutoScrolling ? "pause.fill" : "play.fill")
                        .font(.system(size: 10, weight: .bold))
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(Color.cyan.opacity(0.28), in: Capsule())
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)

                Button {
                    scroller.resetTop()
                } label: {
                    Image(systemName: "arrow.counterclockwise")
                        .font(.system(size: 10))
                        .foregroundStyle(.white.opacity(0.7))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.top, 6)

            PrompterScrollTextView(
                text: $controller.scriptText,
                fontSize: CGFloat(controller.fontSize),
                driver: scroller
            )
            .padding(.horizontal, 8)
            .padding(.bottom, 6)
        }
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.black.opacity(controller.backgroundOpacity))
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.16), lineWidth: 1)
                )
        )
        .onChange(of: controller.scrollSpeedPtsPerSec) { _, newSpeed in
            if controller.isAutoScrolling {
                scroller.setScrolling(true, speed: newSpeed)
            }
        }
    }
}

@MainActor
private final class AutoScrollDriver: ObservableObject {
    weak var scrollView: NSScrollView?
    private var timer: Timer?
    private var speed: Double = 22.0

    func setScrolling(_ active: Bool, speed: Double) {
        self.speed = speed
        timer?.invalidate()
        timer = nil
        guard active else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self = self, let sv = self.scrollView, let doc = sv.documentView else { return }
                let curY = sv.contentView.bounds.origin.y
                let maxY = max(0, doc.bounds.height - sv.contentView.bounds.height)
                let nextY = curY + CGFloat(self.speed / 60.0)
                sv.contentView.scroll(to: NSPoint(x: 0, y: nextY >= maxY ? 0 : nextY))
                sv.reflectScrolledClipView(sv.contentView)
            }
        }
    }

    func resetTop() {
        guard let sv = scrollView else { return }
        sv.contentView.scroll(to: .zero)
        sv.reflectScrolledClipView(sv.contentView)
    }
}

private struct PrompterScrollTextView: NSViewRepresentable {
    @Binding var text: String
    let fontSize: CGFloat
    let driver: AutoScrollDriver

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let sv = NSTextView.scrollableTextView()
        sv.drawsBackground = false
        sv.hasVerticalScroller = true
        if let tv = sv.documentView as? NSTextView {
            tv.drawsBackground = false
            tv.textColor = .white
            tv.insertionPointColor = .cyan
            tv.font = .systemFont(ofSize: fontSize, weight: .medium)
            tv.string = text
            tv.delegate = context.coordinator
        }
        driver.scrollView = sv
        return sv
    }

    func updateNSView(_ sv: NSScrollView, context: Context) {
        guard let tv = sv.documentView as? NSTextView else { return }
        if tv.string != text {
            tv.string = text
        }
        tv.font = .systemFont(ofSize: fontSize, weight: .medium)
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: PrompterScrollTextView
        init(_ parent: PrompterScrollTextView) { self.parent = parent }
        func textDidChange(_ notification: Notification) {
            if let tv = notification.object as? NSTextView {
                parent.text = tv.string
            }
        }
    }
}
