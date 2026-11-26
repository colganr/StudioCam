# StudioCam 4K

Native macOS menu-bar camera studio built in Swift + Metal for Apple Silicon (`M1` / `M2` / `M3` / `M4`+, macOS 14+). Combines hair-accurate Apple Neural Engine background segmentation, **up to 4K (`3840×2160`) looping video backgrounds**, optical Portrait bokeh, Studio Light, physical display-border Edge Light, and whole-composite `.cube` 3D Color LUTs at `< 2 ms` GPU per frame.

## Key Features

- **4K Video Backgrounds (`.mp4`, `.mov`, `.m4v` — HEVC 8/10-bit, H.264, ProRes):** Decoded on the dedicated Apple Silicon Media Engine via `AVPlayerLooper` + `IOSurface` with zero CPU pixel copies. Also supports static images (`HEIC`, `PNG`, `JPG`) and studio gradients.
- **Hair-Accurate 4K Matting at Minimal Cost:** Pairs low-resolution **Apple Neural Engine (ANE)** person segmentation (`640×360` GPU `IOSurface` fast path) with an O(1) separable **Metal Guided-Filter** upsampler against native 4K camera luma (`alpha = a * I_4K + b`), **Color-Similarity-Gated Temporal EMA** (zero edge shimmer, zero motion ghosting), and **Hair Color-Spill Suppression**.
- **Whole-Composite `.cube` 3D & 1D Color LUTs:** Hardware trilinear `MTLTextureType.type3D` (`rgba16Float`) lookup grades both foreground camera and 4K background cohesively in a single shader fetch (`< 0.2 ms` GPU), with 4 built-in `33×33×33` broadcast presets and `0–100%` intensity control.
- **Unified Studio Controls:**
  - **Portrait:** Optical `MPSImageGaussianBlur` depth bokeh (blurs room when Background is Off, or adds cinematic depth-of-field to 4K video/image backgrounds when On).
  - **Studio Light:** Face-tracked contour key light + subtle background dimming.
  - **Edge Light:** Shader rim/fill illumination **plus** an optional physical display-border ring light (`NSPanel`) with a live mouse-cursor cutout.
  - **Reactions:** One-click and optional hand-gesture GPU effects.
- **Universal Camera Support & Auto-Failover:** Works with built-in MacBook `FaceTime HD Camera`, iPhone Continuity Camera, Apple Studio Display, external 4K/1080p/720p UVC webcams, and HDMI capture cards, with live USB unplug failover.
- **Zero-Cert Virtual Webcam Bridge:** Runs ad-hoc signed out of the box and streams zero-copy `IOSurface` frames into an installed CoreMediaIO sink stream (such as `OBS Virtual Camera`) for Zoom, Teams, Meet, and browsers.
- **True `0.0%` Idle Cost:** Automatically stops camera capture, Media Engine video decoding, ANE inference, and Metal rendering whenever the menu-bar popover, Clean Feed window, Virtual Cam, and Recorder are inactive.

## Build & Run

```bash
bash scripts/build_app.sh
open build/StudioCam.app
```

## Run 4K Hardware Benchmark

```bash
./build/StudioCam.app/Contents/MacOS/StudioCam --benchmark
```

## Verified Apple M3 Pro Performance (`3840×2160` 4K)

| Preset (`Portrait` + `Studio Light` + `Background` @ `3840×2160`) | GPU / Frame | Uncapped 4K FPS | CPU @ 4K30 (1 Core) |
| :--- | :--- | :--- | :--- |
| Static 4K Image | `1.65 ms` | `78.1 FPS` | `14.0%` |
| Static 4K Image + `33³` `.cube` LUT | `1.70 ms` | `83.1 FPS` | `13.5%` |
| 4K30 HEVC Video | `1.80 ms` | `83.8 FPS` | `13.2%` |
| 4K30 HEVC Video + `33³` `.cube` LUT | `1.98 ms` | `83.2 FPS` | `13.5%` |
| Idle in Menu Bar (Popover & VCam Off) | `0.00 ms` | — | `0.0%` |
