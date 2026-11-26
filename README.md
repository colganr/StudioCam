# StudioCam

Native macOS menu-bar camera studio built in Swift + Metal for Apple Silicon (`M1` / `M2` / `M3` / `M4`+, macOS 14+). Combines hair-accurate Apple Neural Engine background segmentation, **client-matched 1080p/4K video backgrounds**, **on-chip ANE 3D scene-depth optical bokeh** (for both real hallways/rooms and virtual room backgrounds), **optical light wrap**, **lossless 4K auto-framing**, **invisible-to-screen-share lens prompter**, and **whole-composite `.cube` 3D Color LUTs + 35mm sensor grain**.

## Key Features

- **Client-Matched Resolution Pipeline (`Auto` / `1080p` / `4K`):** Queries and listens to `kCMIOStreamPropertyFormatDescription` on the active virtual camera stream so 1080p apps like Microsoft Teams, Google Meet, and Zoom run a pure native `1920×1080` capture, decode, and Metal composite pipeline with **zero wasted 4K-to-1080p downsampling**.
- **On-Chip ANE 3D Scene-Depth Optical Bokeh (`DepthAnythingV2SmallF16P6`):** Analyzes **both** your physical room/hallway (masking out the foreground subject so near walls to far hallways span the full depth curve) **and** virtual room backgrounds (images or looping videos). Pairs a near Gaussian pass with a **16-tap Golden-Angle Vogel Spiral Optical Disc Bokeh** pass so distant hallway lights bloom into realistic `f/1.4` lens circles at `~0%` ongoing ANE/CPU cost.
- **Optical Light Wrap + Live Room Color Match:** Dynamically matches foreground white balance to the virtual room's ambient color and wraps soft background highlight bloom across hair and shoulder rims.
- **4K Video Backgrounds (`.mp4`, `.mov`, `.m4v` — HEVC, H.264, ProRes):** Hardware-decoded on the Apple Silicon Media Engine via `AVPlayerLooper` + `IOSurface` (`0` CPU pixel copies).
- **Lossless Cinema Auto-Framing:** Uses 4K sensor headroom + ANE face tracking with a critically-damped tripod dead-zone so small nods stay rock steady while chair shifts smoothly reframe.
- **Whole-Composite `.cube` 3D LUTs + Unified 35mm Sensor Grain + UVC Skin Polish:** Hardware trilinear `MTLTextureType.type3D` color grading, subtle organic sensor grain across foreground + background together, and frequency-aware UVC digital oversharpening polish.
- **Invisible-to-Screen-Share Lens Prompter:** Floating talking-points HUD pinned directly under your webcam using `NSWindow.sharingType = .none` so Zoom/Teams screen shares never capture your notes.
- **One-Click Studio Profiles & True `0.0%` Idle Cost:** Switch instantly between `Executive`, `Real Room`, `Creator 4K`, and custom presets; sleeps completely (`0.0%` CPU/GPU) when inactive.

## Build & Run

```bash
bash scripts/build_app.sh
open build/StudioCam.app
```

## Run Hardware Benchmark

```bash
./build/StudioCam.app/Contents/MacOS/StudioCam --benchmark
```
