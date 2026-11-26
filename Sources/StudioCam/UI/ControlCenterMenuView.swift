import SwiftUI
import UniformTypeIdentifiers
import AppKit

/// Clean macOS Control Center-inspired Camera Studio popover with Studio Profiles,
/// Client-Matched Resolution, ANE 3D Scene-Depth Bokeh, Optical Light Wrap, Auto-Framing,
/// 3D .cube LUTs, Unified 35mm Grain, and Screen-Share-Invisible Lens Prompter.
public struct ControlCenterMenuView: View {
    @ObservedObject var pipeline: StudioPipeline
    @ObservedObject var settings: StudioSettings
    @ObservedObject var prompter: LensPrompterController
    let onToggleCleanWindow: () -> Void

    public init(
        pipeline: StudioPipeline,
        settings: StudioSettings,
        prompter: LensPrompterController,
        onToggleCleanWindow: @escaping () -> Void
    ) {
        self.pipeline = pipeline
        self.settings = settings
        self.prompter = prompter
        self.onToggleCleanWindow = onToggleCleanWindow
    }

    public var body: some View {
        VStack(spacing: 0) {
            headerBar
            studioProfilesBar
            previewStage
            cameraAndResolutionBar
            Divider().overlay(Color.white.opacity(0.08))
            ScrollView(.vertical, showsIndicators: false) {
                effectsStack
            }
            .frame(maxHeight: 340)
            Divider().overlay(Color.white.opacity(0.08))
            footerTelemetryBar
        }
        .frame(width: 362)
        .background(Color(nsColor: NSColor(calibratedRed: 0.11, green: 0.13, blue: 0.16, alpha: 0.97)))
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
        HStack(spacing: 9) {
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
                Text("StudioCam")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.white)
                Text(pipeline.resolutionTelemetry)
                    .font(.system(size: 9.5, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.58))
                    .lineLimit(1)
            }

            Spacer()

            // Quick Lens Prompter Toggle (Invisible to Screen Share)
            Button {
                prompter.isVisible.toggle()
            } label: {
                Image(systemName: prompter.isVisible ? "text.viewfinder" : "eye")
                    .font(.system(size: 11.5, weight: .semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(
                        Capsule().fill(prompter.isVisible ? Color.cyan.opacity(0.28) : Color.white.opacity(0.09))
                    )
                    .foregroundStyle(prompter.isVisible ? Color.cyan : Color.white.opacity(0.82))
            }
            .buttonStyle(.plain)
            .help("Toggle Under-Camera Lens Prompter (invisible to Zoom/Teams screen share)")

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
            .help("Broadcast zero-copy frames into OBS / Elgato Virtual Camera")
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
        .padding(.bottom, 6)
    }

    // MARK: - One-Click Studio Profiles Bar

    private var studioProfilesBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(StudioProfilePreset.builtIns) { prof in
                    profilePill(prof)
                }
                if let custom = settings.customUserProfile {
                    profilePill(custom)
                }
                Button {
                    settings.saveCurrentAsCustomProfile()
                } label: {
                    Label("Save", systemImage: "plus")
                        .font(.system(size: 10, weight: .medium))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3.5)
                        .background(Color.white.opacity(0.07), in: Capsule())
                        .foregroundStyle(.white.opacity(0.75))
                }
                .buttonStyle(.plain)
                .help("Save current settings as My Preset")
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 7)
        }
    }

    private func profilePill(_ prof: StudioProfilePreset) -> some View {
        let active = (settings.activeProfileID == prof.id)
        return Button {
            pipeline.applyStudioProfile(prof)
        } label: {
            Text(prof.name)
                .font(.system(size: 10.5, weight: active ? .semibold : .medium))
                .padding(.horizontal, 9)
                .padding(.vertical, 3.5)
                .background(
                    Capsule().fill(active ? Color.accentColor.opacity(0.85) : Color.white.opacity(0.09))
                )
                .foregroundStyle(.white)
        }
        .buttonStyle(.plain)
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
            .frame(width: 362, height: 204)
            .background(Color.black)

            if settings.isDropTargeted {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(Color.cyan, style: StrokeStyle(lineWidth: 2, dash: [6]))
                    .background(Color.black.opacity(0.55))
                    .overlay(
                        Label("Drop Video, Room Image, or .cube LUT", systemImage: "film.stack")
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
                .help("Open Clean Feed Window")

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
                .help("Record Hardware HEVC to ~/Movies")
            }
            .foregroundStyle(.white)
            .padding(8)
        }
    }

    // MARK: - Camera & Client-Matched Resolution Row

    private var cameraAndResolutionBar: some View {
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
                        .foregroundStyle(.white.opacity(0.88))
                        .lineLimit(1)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.5))
                }
            }
            .buttonStyle(.plain)

            Spacer()

            Picker("", selection: $settings.resolution) {
                ForEach(CaptureResolutionPreset.allCases) { p in
                    Text(p.shortLabel).tag(p)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 142)
            .controlSize(.small)
            .help("Auto matches the active resolution requested by Teams/Meet/Zoom (e.g. 1080p) without wasted 4K downsampling")
            .onChange(of: settings.resolution) { _, _ in
                pipeline.applyResolutionChange()
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var currentCameraName: String {
        pipeline.cameras.first(where: { $0.id == settings.selectedCameraUniqueID })?.name ?? "Select Camera"
    }

    // MARK: - Studio Feature Stack

    private var effectsStack: some View {
        VStack(spacing: 2) {
            // 1. Portrait + ANE 3D Hallway/Room Depth Bokeh
            effectRow(
                id: "portrait",
                icon: "f.cursive",
                title: "Portrait & 3D Depth",
                isActive: settings.portraitEnabled,
                trailingAccessory: AnyView(
                    Text(settings.depthBokehEnabled ? "3D f/\(String(format: "%.1f", 4.0 - 2.8 * settings.portraitStrength))" : "Flat")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(settings.portraitEnabled ? Color.cyan : Color.white.opacity(0.4))
                ),
                onToggle: { settings.portraitEnabled.toggle() }
            ) {
                VStack(alignment: .leading, spacing: 6) {
                    Toggle("3D Scene-Depth Bokeh (Real & Virtual Rooms)", isOn: $settings.depthBokehEnabled)
                        .toggleStyle(.checkbox)
                        .font(.system(size: 11))
                        .foregroundStyle(.white.opacity(0.88))
                    HStack {
                        Text("Aperture")
                            .font(.system(size: 10.5))
                            .foregroundStyle(.white.opacity(0.65))
                            .frame(width: 66, alignment: .leading)
                        Slider(value: $settings.portraitStrength, in: 0.1...1.0)
                            .controlSize(.small)
                    }
                    if settings.depthBokehEnabled {
                        HStack {
                            Text("Near Focus")
                                .font(.system(size: 10.5))
                                .foregroundStyle(.white.opacity(0.65))
                                .frame(width: 66, alignment: .leading)
                            Slider(value: $settings.focalPlaneOffset, in: 0.0...0.55)
                                .controlSize(.small)
                        }
                        Text(pipeline.depthAnalyzer.statusLabel)
                            .font(.system(size: 9.5, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.5))
                    }
                }
            }

            // 2. Lossless Auto-Framing ("Center Stage" with Tripod Damper)
            effectRow(
                id: "autoframe",
                icon: "person.crop.rectangle",
                title: "Auto-Framing",
                isActive: settings.autoFramingEnabled,
                onToggle: {
                    settings.autoFramingEnabled.toggle()
                    pipeline.reconfigureCaptureInput()
                }
            ) {
                VStack(alignment: .leading, spacing: 5) {
                    HStack {
                        Text("Framing Zoom")
                            .font(.system(size: 10.5))
                            .foregroundStyle(.white.opacity(0.65))
                            .frame(width: 76, alignment: .leading)
                        Slider(value: $settings.autoFramingZoom, in: 1.08...1.65)
                            .controlSize(.small)
                        Text(String(format: "%.2f×", settings.autoFramingZoom))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.65))
                    }
                    Text("Uses 4K sensor crop headroom + tripod dead-zone damping.")
                        .font(.system(size: 9.5))
                        .foregroundStyle(.white.opacity(0.48))
                }
            }

            // 3. Studio Light
            effectRow(
                id: "studiolight",
                icon: "cube",
                title: "Studio Light",
                isActive: settings.studioLightEnabled,
                onToggle: { settings.studioLightEnabled.toggle() }
            ) {
                HStack {
                    Text("Contour & Dim")
                        .font(.system(size: 10.5))
                        .foregroundStyle(.white.opacity(0.65))
                    Slider(value: $settings.studioLightStrength, in: 0.1...1.0)
                        .controlSize(.small)
                }
            }

            // 4. Edge Light
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
                        .onChange(of: settings.screenRingLightEnabled) { _, _ in pipeline.syncEdgeLightOverlay() }
                    HStack {
                        Text("Brightness")
                            .font(.system(size: 10.5))
                            .foregroundStyle(.white.opacity(0.65))
                            .frame(width: 62, alignment: .leading)
                        Slider(value: $settings.edgeLightStrength, in: 0.1...1.0)
                            .controlSize(.small)
                            .onChange(of: settings.edgeLightStrength) { _, _ in pipeline.syncEdgeLightOverlay() }
                    }
                    HStack {
                        Text("Warmth")
                            .font(.system(size: 10.5))
                            .foregroundStyle(.white.opacity(0.65))
                            .frame(width: 62, alignment: .leading)
                        Slider(value: $settings.edgeLightWarmth, in: 0.0...1.0)
                            .controlSize(.small)
                            .onChange(of: settings.edgeLightWarmth) { _, _ in pipeline.syncEdgeLightOverlay() }
                    }
                }
            }

            // 5. Background + Optical Light Wrap + Live Room Ambient Match
            effectRow(
                id: "background",
                icon: "person.and.background.dotted",
                title: "Background & Wrap",
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
                        pipeline.depthAnalyzer.invalidateSceneCache()
                        pipeline.updatePowerState()
                    }
                }
            ) {
                backgroundExpandedControls
            }

            // 6. Color LUT + Unified 35mm Film Grain + Skin Polish
            effectRow(
                id: "lut",
                icon: "camera.filters",
                title: "Color LUT & Film",
                isActive: settings.lutEnabled || settings.filmGrainStrength > 0.02,
                trailingAccessory: AnyView(lutBadgeView),
                onToggle: { pipeline.toggleLUT() }
            ) {
                lutAndFilmExpandedControls
            }

            // 7. Lens Prompter & Reactions
            effectRow(
                id: "prompter",
                icon: "text.viewfinder",
                title: "Prompter & Reactions",
                isActive: prompter.isVisible || settings.reactionsGestureEnabled,
                onToggle: { prompter.isVisible.toggle() }
            ) {
                VStack(alignment: .leading, spacing: 6) {
                    Toggle("Show Invisible-to-Screen-Share Lens Prompter", isOn: $prompter.isVisible)
                        .toggleStyle(.checkbox)
                        .font(.system(size: 11))
                        .foregroundStyle(.white.opacity(0.88))
                    if prompter.isVisible {
                        HStack {
                            Text("Scroll Speed")
                                .font(.system(size: 10.5))
                                .foregroundStyle(.white.opacity(0.65))
                                .frame(width: 72, alignment: .leading)
                            Slider(value: $prompter.scrollSpeedPtsPerSec, in: 8...65)
                                .controlSize(.mini)
                            Text("Font")
                                .font(.system(size: 10.5))
                                .foregroundStyle(.white.opacity(0.65))
                            Slider(value: $prompter.fontSize, in: 12...26)
                                .controlSize(.mini)
                        }
                    }
                    HStack(spacing: 8) {
                        reactionTriggerButton("👍 Burst", type: 1)
                        reactionTriggerButton("❤️ Heart", type: 2)
                        reactionTriggerButton("🎉 Confetti", type: 3)
                        reactionTriggerButton("⚡ Lasers", type: 4)
                    }
                    Toggle("Hand Gesture Reactions (👍 / ✌️)", isOn: $settings.reactionsGestureEnabled)
                        .toggleStyle(.checkbox)
                        .font(.system(size: 10.5))
                        .foregroundStyle(.white.opacity(0.7))
                }
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
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
                        Image(systemName: "play.fill").font(.system(size: 9, weight: .bold)).foregroundStyle(.white)
                    } else if settings.backgroundMode == .image {
                        Image(systemName: "photo.fill").font(.system(size: 9, weight: .bold)).foregroundStyle(.white)
                    }
                }
            )
            .frame(width: 34, height: 22)
            .overlay(RoundedRectangle(cornerRadius: 5).stroke(Color.white.opacity(0.25), lineWidth: 1))
    }

    private var backgroundExpandedControls: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 6) {
                Button { openMediaFilePicker() } label: {
                    Label("Choose Video / Room Image…", systemImage: "film")
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
                        pipeline.depthAnalyzer.invalidateSceneCache()
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

            if !settings.recentBackgroundURLs.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(settings.recentBackgroundURLs, id: \.path) { url in
                            let isVid = BackgroundMediaController.isVideoURL(url)
                            let selected = (settings.customBackgroundURL?.path == url.path) && (settings.backgroundMode == .video || settings.backgroundMode == .image)
                            Button { pipeline.selectCustomBackground(url: url) } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: isVid ? "video.fill" : "photo").font(.system(size: 9))
                                    Text(url.deletingPathExtension().lastPathComponent)
                                        .font(.system(size: 10.5, weight: .medium))
                                        .lineLimit(1)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(RoundedRectangle(cornerRadius: 5).fill(selected ? Color.cyan.opacity(0.30) : Color.white.opacity(0.08)))
                                .overlay(RoundedRectangle(cornerRadius: 5).stroke(selected ? Color.cyan : Color.clear, lineWidth: 1))
                                .foregroundStyle(.white)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }

            HStack(spacing: 7) {
                ForEach(StudioGradientPreset.presets) { preset in
                    let isSel = (settings.backgroundMode == .gradient && settings.selectedGradientID == preset.id)
                    Button { pipeline.selectGradientPreset(preset) } label: {
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
                            .frame(width: 36, height: 20)
                            .overlay(RoundedRectangle(cornerRadius: 5).stroke(isSel ? Color.white : Color.white.opacity(0.18), lineWidth: isSel ? 2 : 1))
                    }
                    .buttonStyle(.plain)
                    .help(preset.name)
                }
            }

            // Optical Light Wrap + Live Room Color Match
            HStack {
                Text("Light Wrap")
                    .font(.system(size: 10.5))
                    .foregroundStyle(.white.opacity(0.65))
                    .frame(width: 62, alignment: .leading)
                Slider(value: $settings.lightWrapStrength, in: 0.0...1.0)
                    .controlSize(.mini)
                Text("Color Match")
                    .font(.system(size: 10.5))
                    .foregroundStyle(.white.opacity(0.65))
                Slider(value: $settings.ambientMatchStrength, in: 0.0...1.0)
                    .controlSize(.mini)
            }

            HStack {
                Text("Hair Spill")
                    .font(.system(size: 10.5))
                    .foregroundStyle(.white.opacity(0.65))
                    .frame(width: 62, alignment: .leading)
                Slider(value: $settings.spillSuppression, in: 0.0...1.0)
                    .controlSize(.mini)
                Text("Feather")
                    .font(.system(size: 10.5))
                    .foregroundStyle(.white.opacity(0.65))
                Slider(value: $settings.edgeSoftness, in: 0.05...0.95)
                    .controlSize(.mini)
            }
        }
    }

    private var lutBadgeView: some View {
        Text(settings.lutEnabled ? (settings.selectedLUTURL?.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "_", with: " ") ?? "Active") : "Off")
            .font(.system(size: 10, weight: .medium))
            .lineLimit(1)
            .frame(maxWidth: 110, alignment: .trailing)
            .foregroundStyle(settings.lutEnabled ? Color.cyan : Color.white.opacity(0.45))
    }

    private var lutAndFilmExpandedControls: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 6) {
                Button { openLUTFilePicker() } label: {
                    Label("Choose .cube 3D LUT…", systemImage: "cube.transparent")
                        .font(.system(size: 11.5, weight: .medium))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 5)
                        .background(Color.accentColor.opacity(0.85), in: RoundedRectangle(cornerRadius: 6))
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)

                if settings.lutEnabled {
                    Button("LUT Off") { settings.lutEnabled = false }
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
                            Button { pipeline.selectLUT(url: url) } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "camera.filters").font(.system(size: 9))
                                    Text(url.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "_", with: " "))
                                        .font(.system(size: 10.5, weight: .medium))
                                        .lineLimit(1)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(RoundedRectangle(cornerRadius: 5).fill(selected ? Color.cyan.opacity(0.30) : Color.white.opacity(0.08)))
                                .overlay(RoundedRectangle(cornerRadius: 5).stroke(selected ? Color.cyan : Color.clear, lineWidth: 1))
                                .foregroundStyle(.white)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }

            HStack {
                Text("LUT Mix")
                    .font(.system(size: 10.5))
                    .foregroundStyle(.white.opacity(0.65))
                    .frame(width: 62, alignment: .leading)
                Slider(value: $settings.lutStrength, in: 0.0...1.0)
                    .controlSize(.mini)
                Text("\(Int(settings.lutStrength * 100))%")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.65))
                    .frame(width: 32, alignment: .trailing)
            }

            HStack {
                Text("35mm Grain")
                    .font(.system(size: 10.5))
                    .foregroundStyle(.white.opacity(0.65))
                    .frame(width: 62, alignment: .leading)
                Slider(value: $settings.filmGrainStrength, in: 0.0...0.65)
                    .controlSize(.mini)
                Text("Skin Soft")
                    .font(.system(size: 10.5))
                    .foregroundStyle(.white.opacity(0.65))
                Slider(value: $settings.skinPolishStrength, in: 0.0...0.85)
                    .controlSize(.mini)
            }
        }
    }

    private func reactionTriggerButton(_ label: String, type: UInt32) -> some View {
        Button { settings.triggerReaction(type) } label: {
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
                            .frame(width: 27, height: 27)
                        Image(systemName: icon)
                            .font(.system(size: 12.5, weight: .semibold))
                            .foregroundStyle(isActive ? Color.white : Color.white.opacity(0.82))
                    }
                }
                .buttonStyle(.plain)

                Button {
                    withAnimation(.easeInOut(duration: 0.14)) {
                        settings.expandedRow = isExpanded ? nil : id
                    }
                } label: {
                    HStack {
                        Text(title)
                            .font(.system(size: 12.5, weight: .medium))
                            .foregroundStyle(.white)
                        Spacer()
                        if let acc = trailingAccessory { acc }
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.right")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.42))
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 3.5)

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

    // MARK: - Footer Hardware Telemetry

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
        .padding(.vertical, 7)
    }

    // MARK: - Pickers & Drag-Drop

    private func openMediaFilePicker() {
        let panel = NSOpenPanel()
        panel.title = "Choose Video or Room Image Background"
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
