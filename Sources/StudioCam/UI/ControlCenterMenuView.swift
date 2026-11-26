import SwiftUI
import MetalPerformanceShaders
import UniformTypeIdentifiers
import AppKit

/// Clean macOS Control Center Video Effects menu-bar UI with 4K Video Background support.
public struct ControlCenterMenuView: View {
    @ObservedObject var pipeline: StudioPipeline
    @ObservedObject var settings: StudioSettings
    let onToggleCleanWindow: () -> Void

    public init(
        pipeline: StudioPipeline,
        settings: StudioSettings,
        onToggleCleanWindow: @escaping () -> Void
    ) {
        self.pipeline = pipeline
        self.settings = settings
        self.onToggleCleanWindow = onToggleCleanWindow
    }

    public var body: some View {
        VStack(spacing: 0) {
            headerBar
            previewStage
            cameraSelectorBar
            Divider().overlay(Color.white.opacity(0.08))
            effectsStack
            Divider().overlay(Color.white.opacity(0.08))
            footerTelemetryBar
        }
        .frame(width: 348)
        .background(Color(nsColor: NSColor(calibratedRed: 0.11, green: 0.13, blue: 0.16, alpha: 0.96)))
        .onAppear {
            pipeline.refreshCameras()
            pipeline.isPopoverVisible = true
        }
        .onDisappear {
            pipeline.isPopoverVisible = false
        }
        .onDrop(of: [.fileURL], isTargeted: $settings.isDropTargeted) { providers in
            handleDrop(providers: providers)
        }
    }

    // MARK: - Top Header

    private var headerBar: some View {
        HStack(spacing: 10) {
            ZStack {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [Color(red: 0.58, green: 0.25, blue: 0.98), Color(red: 0.20, green: 0.68, blue: 0.98)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 28, height: 28)
                Image(systemName: "video.badge.waveform.fill")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(.white)
            }

            VStack(alignment: .leading, spacing: 1) {
                Text("StudioCam 4K")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.white)
                Text(pipeline.activeMediaLabel)
                    .font(.system(size: 10))
                    .foregroundStyle(.white.opacity(0.60))
                    .lineLimit(1)
            }

            Spacer()

            // Zero-Cert Virtual Camera Live Pill
            Button {
                pipeline.toggleVirtualCamera(!settings.virtualCamEnabled)
            } label: {
                HStack(spacing: 5) {
                    Circle()
                        .fill(settings.virtualCamEnabled ? Color.green : Color.white.opacity(0.35))
                        .frame(width: 7, height: 7)
                    Text(settings.virtualCamEnabled ? "VCam Live" : "VCam Off")
                        .font(.system(size: 11, weight: .medium))
                }
                .padding(.horizontal, 9)
                .padding(.vertical, 4)
                .background(
                    Capsule().fill(settings.virtualCamEnabled ? Color.green.opacity(0.22) : Color.white.opacity(0.09))
                )
                .foregroundStyle(settings.virtualCamEnabled ? Color.green : Color.white.opacity(0.85))
            }
            .buttonStyle(.plain)
            .help("Broadcast zero-copy frames into OBS / Elgato Virtual Camera (no Apple Developer cert needed)")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    // MARK: - 16:9 Live Metal Preview Stage

    private var previewStage: some View {
        ZStack(alignment: .bottomTrailing) {
            MetalPreviewView(
                device: pipeline.compositor.device,
                commandQueue: pipeline.compositor.commandQueue,
                registerFrameProvider: { handler in
                    pipeline.onFrameRendered = handler
                }
            )
            .frame(width: 348, height: 196)
            .background(Color.black)

            if settings.isDropTargeted {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(Color.cyan, style: StrokeStyle(lineWidth: 2, dash: [6]))
                    .background(Color.black.opacity(0.55))
                    .overlay(
                        Label("Drop 4K Video, Image, or .cube LUT", systemImage: "film.stack")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.white)
                    )
                    .padding(8)
            }

            HStack(spacing: 6) {
                Button {
                    settings.mirrorPreview.toggle()
                } label: {
                    Image(systemName: "arrow.left.and.right.righttriangle.left.righttriangle.right")
                        .font(.system(size: 11, weight: .semibold))
                        .padding(6)
                        .background(.ultraThinMaterial, in: Circle())
                }
                .buttonStyle(.plain)
                .help("Mirror Horizontal")

                Button {
                    onToggleCleanWindow()
                } label: {
                    Image(systemName: "macwindow.on.rectangle")
                        .font(.system(size: 11, weight: .semibold))
                        .padding(6)
                        .background(.ultraThinMaterial, in: Circle())
                }
                .buttonStyle(.plain)
                .help("Open 4K Clean Feed Window")

                Button {
                    pipeline.toggleRecording()
                } label: {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(pipeline.isRecording ? Color.red : Color.white)
                            .frame(width: 7, height: 7)
                        if pipeline.isRecording {
                            Text("REC")
                                .font(.system(size: 10, weight: .bold))
                        }
                    }
                    .padding(.horizontal, pipeline.isRecording ? 8 : 6)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
                }
                .buttonStyle(.plain)
                .help("Record Hardware 4K HEVC to ~/Movies")
            }
            .foregroundStyle(.white)
            .padding(8)
        }
    }

    // MARK: - Camera & Resolution Picker Row

    private var cameraSelectorBar: some View {
        HStack(spacing: 8) {
            Menu {
                ForEach(pipeline.cameras) { cam in
                    Button {
                        pipeline.selectCamera(uniqueID: cam.id)
                    } label: {
                        HStack {
                            Text(cam.name)
                            if cam.id == settings.selectedCameraUniqueID {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
                Divider()
                Button("Refresh Cameras") { pipeline.refreshCameras() }
            } label: {
                HStack(spacing: 5) {
                    Text(currentCameraName)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.white.opacity(0.85))
                        .lineLimit(1)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.5))
                }
            }
            .buttonStyle(.plain)

            Spacer()

            Picker("", selection: $settings.resolution) {
                Text("4K").tag(CaptureResolutionPreset.uhd4K)
                Text("1080p").tag(CaptureResolutionPreset.fhd1080)
            }
            .pickerStyle(.segmented)
            .frame(width: 104)
            .controlSize(.small)
            .onChange(of: settings.resolution) { _, _ in
                pipeline.applyResolutionChange()
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }

    private var currentCameraName: String {
        pipeline.cameras.first(where: { $0.id == settings.selectedCameraUniqueID })?.name ?? "Select Camera"
    }

    // MARK: - 5 Control Center Feature Rows

    private var effectsStack: some View {
        VStack(spacing: 2) {
            // 1. Portrait
            effectRow(
                id: "portrait",
                icon: "f.cursive",
                title: "Portrait",
                isActive: settings.portraitEnabled,
                onToggle: { settings.portraitEnabled.toggle() }
            ) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Depth Bokeh")
                            .font(.system(size: 11))
                            .foregroundStyle(.white.opacity(0.65))
                        Slider(value: $settings.portraitStrength, in: 0.1...1.0)
                            .controlSize(.small)
                    }
                }
            }

            // 2. Studio Light
            effectRow(
                id: "studiolight",
                icon: "cube",
                title: "Studio Light",
                isActive: settings.studioLightEnabled,
                onToggle: { settings.studioLightEnabled.toggle() }
            ) {
                HStack {
                    Text("Contour & Dim")
                        .font(.system(size: 11))
                        .foregroundStyle(.white.opacity(0.65))
                    Slider(value: $settings.studioLightStrength, in: 0.1...1.0)
                        .controlSize(.small)
                }
            }

            // 3. Edge Light (Shader + Physical Screen Ring Light)
            effectRow(
                id: "edgelight",
                icon: "sun.max.circle",
                title: "Edge Light",
                isActive: settings.edgeLightEnabled,
                onToggle: {
                    settings.edgeLightEnabled.toggle()
                    pipeline.syncEdgeLightOverlay()
                }
            ) {
                VStack(alignment: .leading, spacing: 6) {
                    Toggle("Display Border Ring Light (Cursor Cutout)", isOn: $settings.screenRingLightEnabled)
                        .toggleStyle(.checkbox)
                        .font(.system(size: 11))
                        .foregroundStyle(.white.opacity(0.85))
                        .onChange(of: settings.screenRingLightEnabled) { _, _ in
                            pipeline.syncEdgeLightOverlay()
                        }
                    HStack {
                        Text("Brightness")
                            .font(.system(size: 11))
                            .foregroundStyle(.white.opacity(0.65))
                            .frame(width: 62, alignment: .leading)
                        Slider(value: $settings.edgeLightStrength, in: 0.1...1.0)
                            .controlSize(.small)
                            .onChange(of: settings.edgeLightStrength) { _, _ in pipeline.syncEdgeLightOverlay() }
                    }
                    HStack {
                        Text("Warmth")
                            .font(.system(size: 11))
                            .foregroundStyle(.white.opacity(0.65))
                            .frame(width: 62, alignment: .leading)
                        Slider(value: $settings.edgeLightWarmth, in: 0.0...1.0)
                            .controlSize(.small)
                            .onChange(of: settings.edgeLightWarmth) { _, _ in pipeline.syncEdgeLightOverlay() }
                    }
                }
            }

            // 4. Reactions
            effectRow(
                id: "reactions",
                icon: "plus.magnifyingglass",
                title: "Reactions",
                isActive: settings.reactionsGestureEnabled,
                onToggle: { settings.reactionsGestureEnabled.toggle() }
            ) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        reactionTriggerButton("👍 Burst", type: 1)
                        reactionTriggerButton("❤️ Heart", type: 2)
                        reactionTriggerButton("🎉 Confetti", type: 3)
                        reactionTriggerButton("⚡ Lasers", type: 4)
                    }
                    Toggle("Enable Hand Gesture Detection (👍 / ✌️)", isOn: $settings.reactionsGestureEnabled)
                        .toggleStyle(.checkbox)
                        .font(.system(size: 10.5))
                        .foregroundStyle(.white.opacity(0.7))
                }
            }

            // 5. Background (with 4K Video + Image + Gradient Picker!)
            effectRow(
                id: "background",
                icon: "person.and.background.dotted",
                title: "Background",
                isActive: settings.backgroundMode != .none,
                trailingAccessory: AnyView(backgroundSwatchPreview),
                onToggle: {
                    if settings.backgroundMode == .none {
                        if let custom = settings.customBackgroundURL {
                            pipeline.selectCustomBackground(url: custom)
                        } else {
                            settings.backgroundMode = .gradient
                        }
                    } else {
                        settings.backgroundMode = .none
                        pipeline.updatePowerState()
                    }
                }
            ) {
                backgroundExpandedControls
            }

            // 6. Whole-Composite 3D Color LUT (.cube)
            effectRow(
                id: "lut",
                icon: "camera.filters",
                title: "Color LUT",
                isActive: settings.lutEnabled,
                trailingAccessory: AnyView(lutBadgeView),
                onToggle: {
                    pipeline.toggleLUT()
                }
            ) {
                lutExpandedControls
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private var lutBadgeView: some View {
        Text(settings.lutEnabled ? (settings.selectedLUTURL?.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "_", with: " ") ?? "Active") : "Off")
            .font(.system(size: 10, weight: .medium))
            .lineLimit(1)
            .frame(maxWidth: 110, alignment: .trailing)
            .foregroundStyle(settings.lutEnabled ? Color.cyan : Color.white.opacity(0.45))
    }

    private var lutExpandedControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Button {
                    openLUTFilePicker()
                } label: {
                    Label("Choose .cube 3D LUT…", systemImage: "cube.transparent")
                        .font(.system(size: 11.5, weight: .medium))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 5)
                        .background(Color.accentColor.opacity(0.85), in: RoundedRectangle(cornerRadius: 6))
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)

                if settings.lutEnabled {
                    Button("Off") {
                        settings.lutEnabled = false
                    }
                    .font(.system(size: 11))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(Color.white.opacity(0.10), in: RoundedRectangle(cornerRadius: 6))
                    .foregroundStyle(.white.opacity(0.8))
                    .buttonStyle(.plain)
                }
            }

            if !settings.recentLUTURLs.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(settings.recentLUTURLs, id: \.path) { url in
                            let selected = settings.lutEnabled && (settings.selectedLUTURL?.path == url.path)
                            Button {
                                pipeline.selectLUT(url: url)
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "camera.filters")
                                        .font(.system(size: 9))
                                    Text(url.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "_", with: " "))
                                        .font(.system(size: 10.5, weight: .medium))
                                        .lineLimit(1)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(
                                    RoundedRectangle(cornerRadius: 5)
                                        .fill(selected ? Color.cyan.opacity(0.30) : Color.white.opacity(0.08))
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 5)
                                        .stroke(selected ? Color.cyan : Color.clear, lineWidth: 1)
                                )
                                .foregroundStyle(.white)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }

            HStack {
                Text("Intensity")
                    .font(.system(size: 10.5))
                    .foregroundStyle(.white.opacity(0.65))
                    .frame(width: 58, alignment: .leading)
                Slider(value: $settings.lutStrength, in: 0.0...1.0)
                    .controlSize(.small)
                Text("\(Int(settings.lutStrength * 100))%")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.65))
                    .frame(width: 32, alignment: .trailing)
            }
        }
    }

    private var backgroundSwatchPreview: some View {
        let g = settings.currentGradient
        return RoundedRectangle(cornerRadius: 5, style: .continuous)
            .fill(
                LinearGradient(
                    colors: [
                        Color(red: Double(g.topRGB.x), green: Double(g.topRGB.y), blue: Double(g.topRGB.z)),
                        Color(red: Double(g.bottomRGB.x), green: Double(g.bottomRGB.y), blue: Double(g.bottomRGB.z))
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .overlay(
                Group {
                    if settings.backgroundMode == .video {
                        Image(systemName: "play.fill")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(.white)
                    } else if settings.backgroundMode == .image {
                        Image(systemName: "photo.fill")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(.white)
                    }
                }
            )
            .frame(width: 34, height: 22)
            .overlay(RoundedRectangle(cornerRadius: 5).stroke(Color.white.opacity(0.25), lineWidth: 1))
    }

    private var backgroundExpandedControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Choose 4K Video or Image Button
            HStack(spacing: 6) {
                Button {
                    openMediaFilePicker()
                } label: {
                    Label("Choose 4K Video / Image…", systemImage: "film")
                        .font(.system(size: 11.5, weight: .medium))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 5)
                        .background(Color.accentColor.opacity(0.85), in: RoundedRectangle(cornerRadius: 6))
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)

                if settings.backgroundMode != .none {
                    Button("Off") {
                        settings.backgroundMode = .none
                        pipeline.updatePowerState()
                    }
                    .font(.system(size: 11))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(Color.white.opacity(0.10), in: RoundedRectangle(cornerRadius: 6))
                    .foregroundStyle(.white.opacity(0.8))
                    .buttonStyle(.plain)
                }
            }

            // Recent / Sample 4K Videos & Images
            if !settings.recentBackgroundURLs.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(settings.recentBackgroundURLs, id: \.path) { url in
                            let isVid = BackgroundMediaController.isVideoURL(url)
                            let selected = (settings.customBackgroundURL?.path == url.path) && (settings.backgroundMode == .video || settings.backgroundMode == .image)
                            Button {
                                pipeline.selectCustomBackground(url: url)
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: isVid ? "video.fill" : "photo")
                                        .font(.system(size: 9))
                                    Text(url.deletingPathExtension().lastPathComponent)
                                        .font(.system(size: 10.5, weight: .medium))
                                        .lineLimit(1)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(
                                    RoundedRectangle(cornerRadius: 5)
                                        .fill(selected ? Color.cyan.opacity(0.30) : Color.white.opacity(0.08))
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 5)
                                        .stroke(selected ? Color.cyan : Color.clear, lineWidth: 1)
                                )
                                .foregroundStyle(.white)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }

            // Studio Gradients Strip
            HStack(spacing: 7) {
                ForEach(StudioGradientPreset.presets) { preset in
                    let isSel = (settings.backgroundMode == .gradient && settings.selectedGradientID == preset.id)
                    Button {
                        pipeline.selectGradientPreset(preset)
                    } label: {
                        RoundedRectangle(cornerRadius: 5)
                            .fill(
                                LinearGradient(
                                    colors: [
                                        Color(red: Double(preset.topRGB.x), green: Double(preset.topRGB.y), blue: Double(preset.topRGB.z)),
                                        Color(red: Double(preset.bottomRGB.x), green: Double(preset.bottomRGB.y), blue: Double(preset.bottomRGB.z))
                                    ],
                                    startPoint: .top,
                                    endPoint: .bottom
                                )
                            )
                            .frame(width: 36, height: 22)
                            .overlay(
                                RoundedRectangle(cornerRadius: 5)
                                    .stroke(isSel ? Color.white : Color.white.opacity(0.18), lineWidth: isSel ? 2 : 1)
                            )
                    }
                    .buttonStyle(.plain)
                    .help(preset.name)
                }
            }

            // Matte Quality & Hair Spill Suppression Controls
            HStack {
                Text("Hair Spill")
                    .font(.system(size: 10.5))
                    .foregroundStyle(.white.opacity(0.65))
                    .frame(width: 58, alignment: .leading)
                Slider(value: $settings.spillSuppression, in: 0.0...1.0)
                    .controlSize(.mini)
                Text("Feather")
                    .font(.system(size: 10.5))
                    .foregroundStyle(.white.opacity(0.65))
                Slider(value: $settings.edgeSoftness, in: 0.05...0.95)
                    .controlSize(.mini)
            }

            Picker("Matte Engine", selection: $settings.segmentationQuality) {
                ForEach(SegmentationQualityPreset.allCases) { q in
                    Text(q.title).tag(q)
                }
            }
            .font(.system(size: 10.5))
            .controlSize(.mini)
            .foregroundStyle(.white.opacity(0.75))
        }
    }

    private func reactionTriggerButton(_ label: String, type: UInt32) -> some View {
        Button {
            settings.triggerReaction(type)
        } label: {
            Text(label)
                .font(.system(size: 10.5, weight: .medium))
                .padding(.horizontal, 7)
                .padding(.vertical, 4)
                .background(Color.white.opacity(0.10), in: RoundedRectangle(cornerRadius: 5))
                .foregroundStyle(.white)
        }
        .buttonStyle(.plain)
    }

    private func effectRow<Content: View>(
        id: String,
        icon: String,
        title: String,
        isActive: Bool,
        trailingAccessory: AnyView? = nil,
        onToggle: @escaping () -> Void,
        @ViewBuilder details: () -> Content
    ) -> some View {
        let isExpanded = (settings.expandedRow == id)
        return VStack(spacing: 6) {
            HStack(spacing: 10) {
                Button(action: onToggle) {
                    ZStack {
                        Circle()
                            .fill(isActive ? Color.accentColor : Color.white.opacity(0.12))
                            .frame(width: 28, height: 28)
                        Image(systemName: icon)
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(isActive ? Color.white : Color.white.opacity(0.82))
                    }
                }
                .buttonStyle(.plain)

                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        settings.expandedRow = isExpanded ? nil : id
                    }
                } label: {
                    HStack {
                        Text(title)
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.white)
                        Spacer()
                        if let acc = trailingAccessory {
                            acc
                        }
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.right")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.42))
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 4)

            if isExpanded {
                details()
                    .padding(.horizontal, 10)
                    .padding(.bottom, 6)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .fill(isExpanded ? Color.white.opacity(0.05) : Color.clear)
        )
    }

    // MARK: - Footer Hardware Telemetry + Quit

    private var footerTelemetryBar: some View {
        HStack(spacing: 8) {
            Text(String(
                format: "%.0f fps · GPU %.2f ms · ANE %.1f ms",
                pipeline.fpsTelemetry,
                pipeline.gpuMsTelemetry,
                pipeline.aneMsTelemetry
            ))
            .font(.system(size: 10, design: .monospaced))
            .foregroundStyle(.white.opacity(0.55))

            Spacer()

            Button("Quit") {
                pipeline.sinkBridge.stop()
                NSApplication.shared.terminate(nil)
            }
            .font(.system(size: 10.5, weight: .medium))
            .foregroundStyle(.white.opacity(0.65))
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }

    // MARK: - Native macOS File Picker & Drag-Drop

    private func openMediaFilePicker() {
        let panel = NSOpenPanel()
        panel.title = "Choose 4K Video or Image Background"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.mpeg4Movie, .quickTimeMovie, .movie, .video, .image, .heic, .png, .jpeg]
        NSApp.activate(ignoringOtherApps: true)
        if panel.runModal() == .OK, let url = panel.url {
            pipeline.selectCustomBackground(url: url)
        }
    }

    private func openLUTFilePicker() {
        let panel = NSOpenPanel()
        panel.title = "Choose .cube 3D or 1D Color LUT"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if let cubeType = UTType(filenameExtension: "cube") {
            panel.allowedContentTypes = [cubeType]
        }
        NSApp.activate(ignoringOtherApps: true)
        if panel.runModal() == .OK, let url = panel.url {
            pipeline.selectLUT(url: url)
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
            guard let data = item as? Data,
                  let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
            Task { @MainActor in
                if LUTLoader.isLUTFile(url) {
                    pipeline.selectLUT(url: url)
                } else {
                    pipeline.selectCustomBackground(url: url)
                }
            }
        }
        return true
    }
}
