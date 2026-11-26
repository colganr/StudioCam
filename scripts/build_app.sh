#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Building StudioCam (Release arm64)..."
swift build -c release

APP_DIR="$ROOT_DIR/build/StudioCam.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RES_DIR="$CONTENTS_DIR/Resources"

rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RES_DIR"

cp "$ROOT_DIR/.build/release/StudioCam" "$MACOS_DIR/StudioCam"
cp "$ROOT_DIR/Sources/StudioCam/Shaders/StudioCamShaders.metal" "$RES_DIR/StudioCamShaders.metal"
if [ -d "$ROOT_DIR/SampleBackgrounds" ]; then
    cp -R "$ROOT_DIR/SampleBackgrounds" "$RES_DIR/SampleBackgrounds"
fi

echo "[2/4] Precompiling Metal shaders (if Metal toolchain is available)..."
if xcrun -f metal >/dev/null 2>&1 && xcrun -f metallib >/dev/null 2>&1; then
    if xcrun metal -O3 -ffast-math -c "$ROOT_DIR/Sources/StudioCam/Shaders/StudioCamShaders.metal" -o "$ROOT_DIR/build/StudioCamShaders.air" 2>/dev/null; then
        xcrun metallib "$ROOT_DIR/build/StudioCamShaders.air" -o "$RES_DIR/default.metallib"
        rm -f "$ROOT_DIR/build/StudioCamShaders.air"
        echo "  -> Compiled $RES_DIR/default.metallib"
    else
        echo "  -> Metal CLI toolchain component not installed; bundled StudioCamShaders.metal for instant runtime MTLDevice compilation."
    fi
fi

cat > "$CONTENTS_DIR/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>StudioCam 4K</string>
    <key>CFBundleExecutable</key>
    <string>StudioCam</string>
    <key>CFBundleIdentifier</key>
    <string>com.thetonegeek.studiocam</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>StudioCam</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSCameraUsageDescription</key>
    <string>StudioCam 4K captures your camera for real-time Apple Neural Engine background replacement, 4K video backgrounds, and studio lighting.</string>
</dict>
</plist>
PLIST

echo "[3/4] Ad-hoc signing StudioCam.app (zero Apple Developer certificate required)..."
codesign --force --deep --sign - "$APP_DIR"

echo "[4/4] Verifying bundle signature..."
codesign --verify --deep --strict --verbose=2 "$APP_DIR"
echo "Done! Built: $APP_DIR"
