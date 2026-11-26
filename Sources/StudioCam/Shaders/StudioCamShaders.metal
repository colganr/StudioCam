#include <metal_stdlib>
using namespace metal;

// Exact 240-byte 16-byte-aligned layout shared between Metal and Swift (`CompositeUniformsGPU`)
struct CompositeUniforms {
    uint2 outputSize;          // 0
    uint2 camSize;             // 8
    uint2 bgSize;              // 16
    uint camIsYUV;             // 24
    uint camFullRange;         // 28

    uint bgMode;               // 32: 0=none, 1=gradient, 2=image, 3=video
    uint bgIsYUV;              // 36
    uint portraitEnabled;      // 40
    float portraitStrength;    // 44

    uint depthBokehEnabled;    // 48: 1 = ANE 3D scene-depth graded optical bokeh
    float focalPlaneOffset;    // 52: 0.0..0.6
    uint studioLightEnabled;   // 56
    float studioLightStrength; // 60

    uint edgeLightEnabled;     // 64
    float edgeLightStrength;   // 68
    float edgeLightWarmth;     // 72
    float spillSuppression;    // 76

    float edgeSoftness;        // 80
    uint mirrorHorizontal;     // 84
    float lightWrapStrength;   // 88: Optical Light Wrap (0..1)
    float ambientMatchStrength;// 92: Live BG Ambient Color Match (0..1)

    float skinPolishStrength;  // 96: UVC Forehead/Cheek De-Harshness (0..1)
    float filmGrainStrength;   // 100: Unified 35mm Sensor Grain (0..1)
    uint reactionType;         // 104
    float reactionProgress;    // 108

    float timeSeconds;         // 112
    float lutStrength;         // 116
    uint lutEnabled;           // 120
    uint lutDimension;         // 124

    float4 gradTopColor;       // 128
    float4 gradBottomColor;    // 144
    float4 faceRect;           // 160
    float4 camCropRect;        // 176: [originX, originY, widthFrac, heightFrac] for lossless Auto-Framing
    float4 lutDomainMin;       // 192
    float4 lutDomainMax;       // 208
    float4 matteTuning;        // 224: x=matteChoke (-0.25..0.25), y=temporalStability (0..1), z=unused, w=unused
};

struct GuidedUniforms {
    uint2 gridSize;
    int radius;
    float eps;
    float temporalBlend;
    float colorSimSigma;
    uint hasHistory;
    uint camIsYUV;
    uint camFullRange;
};

inline float3 yuvToRGB(float ySample, float2 cbcrSample, bool fullRange) {
    float y = fullRange ? ySample : ((ySample - (16.0f / 255.0f)) * (255.0f / 219.0f));
    float cb = cbcrSample.x - 0.5f;
    float cr = cbcrSample.y - 0.5f;
    float r = y + 1.5748f * cr;
    float g = y - 0.1873f * cb - 0.4681f * cr;
    float b = y + 1.8556f * cb;
    return saturate(float3(r, g, b));
}

inline float rgbLuma(float3 rgb) {
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

inline float2 aspectFillUV(float2 dstUV, uint2 srcSize, uint2 dstSize) {
    if (srcSize.x == 0 || srcSize.y == 0 || dstSize.x == 0 || dstSize.y == 0) {
        return dstUV;
    }
    float srcAspect = float(srcSize.x) / float(srcSize.y);
    float dstAspect = float(dstSize.x) / float(dstSize.y);
    float2 uv = dstUV;
    if (srcAspect > dstAspect) {
        float scale = dstAspect / srcAspect;
        uv.x = (dstUV.x - 0.5f) * scale + 0.5f;
    } else {
        float scale = srcAspect / dstAspect;
        uv.y = (dstUV.y - 0.5f) * scale + 0.5f;
    }
    return clamp(uv, 0.0005f, 0.9995f);
}

inline float3 sampleCameraRGB(
    texture2d<float, access::sample> plane0,
    texture2d<float, access::sample> plane1,
    sampler linSamp,
    float2 pUV,
    uint camIsYUV,
    uint camFullRange
) {
    if (camIsYUV != 0) {
        float y = plane0.sample(linSamp, pUV).r;
        float2 cbcr = plane1.sample(linSamp, pUV).rg;
        return yuvToRGB(y, cbcr, camFullRange != 0);
    }
    return plane0.sample(linSamp, pUV).rgb;
}

// Pass 1: 3x3 denoised guide + Portrait.framework-faithful RGB Color-Similarity Temporal EMA
// Gates temporal lock STRICTLY on denoised camera RGB motion (never on raw Vision mask jumps),
// crushing frame-to-frame Vision edge shimmer by ~8x while remaining instant when you move.
kernel void guidedPrepAndTemporalEMA(
    texture2d<float, access::sample> camPlane0 [[texture(0)]],
    texture2d<float, access::sample> camPlane1 [[texture(1)]],
    texture2d<float, access::sample> rawMaskTex [[texture(2)]],
    texture2d<float, access::read>   prevGuideAndMask [[texture(3)]],
    texture2d<float, access::write>  nextGuideAndMask [[texture(4)]],
    texture2d<float, access::write>  statsOut [[texture(5)]],
    constant GuidedUniforms& u [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= u.gridSize.x || gid.y >= u.gridSize.y) return;

    constexpr sampler linSamp(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 invGrid = 1.0f / float2(u.gridSize);
    float2 uv = (float2(gid) + 0.5f) * invGrid;

    // 5-tap denoised camera RGB suppresses webcam ISO noise so stationary pixels stay rock-locked
    float3 c0 = sampleCameraRGB(camPlane0, camPlane1, linSamp, uv, u.camIsYUV, u.camFullRange);
    float3 cL = sampleCameraRGB(camPlane0, camPlane1, linSamp, uv + float2(-invGrid.x, 0.0f), u.camIsYUV, u.camFullRange);
    float3 cR = sampleCameraRGB(camPlane0, camPlane1, linSamp, uv + float2( invGrid.x, 0.0f), u.camIsYUV, u.camFullRange);
    float3 cU = sampleCameraRGB(camPlane0, camPlane1, linSamp, uv + float2(0.0f, -invGrid.y), u.camIsYUV, u.camFullRange);
    float3 cD = sampleCameraRGB(camPlane0, camPlane1, linSamp, uv + float2(0.0f,  invGrid.y), u.camIsYUV, u.camFullRange);
    float3 smoothRGB = 0.40f * c0 + 0.15f * (cL + cR + cU + cD);

    // 5-tap spatial anti-pop on raw Vision mask before temporal integration
    float m0 = saturate(rawMaskTex.sample(linSamp, uv).r);
    float mL = saturate(rawMaskTex.sample(linSamp, uv + float2(-invGrid.x, 0.0f)).r);
    float mR = saturate(rawMaskTex.sample(linSamp, uv + float2( invGrid.x, 0.0f)).r);
    float mU = saturate(rawMaskTex.sample(linSamp, uv + float2(0.0f, -invGrid.y)).r);
    float mD = saturate(rawMaskTex.sample(linSamp, uv + float2(0.0f,  invGrid.y)).r);
    float p = 0.44f * m0 + 0.14f * (mL + mR + mU + mD);

    float3 nextStoredRGB = smoothRGB;
    if (u.hasHistory != 0) {
        float4 prev = prevGuideAndMask.read(gid);
        float3 prevRGB = prev.rgb;
        float prevMask = prev.a;

        // 1. Shadow-sensitive Weber + Absolute RGB Color-Similarity Weight
        // Catches low-contrast movement (e.g. dark shirt swaying across a shadowed office wall)
        // while staying rock-locked on stationary 5-tap denoised webcam ISO grain.
        float absDiff = length(smoothRGB - prevRGB);
        float refLuma = max(0.24f, max(rgbLuma(smoothRGB), rgbLuma(prevRGB)));
        float weberDiff = absDiff / refLuma;
        float motionMag = max(absDiff, weberDiff * 0.32f);
        float simWeight = exp(-(motionMag * motionMag) / max(1e-5f, 2.0f * u.colorSimSigma * u.colorSimSigma));

        // 2. Wide-stencil (±2 guide texels ≈ ±7.5px at 1080p) Coherent-Flip Ghost Buster
        // Stationary 1–6px Vision edge shimmer straddles the boundary (mWakeMin low, mWakeMax high) -> stays locked.
        // Fast side-to-side body sway creates a coherent vacated wake (mWakeMax < 0.25) or newly covered body
        // (mWakeMin > 0.75) -> flushes trailing & leading ghosts to 0 on frame 1 even on identical shirt/wall colors!
        float2 wakeStep = invGrid * 2.0f;
        float wL = saturate(rawMaskTex.sample(linSamp, uv + float2(-wakeStep.x, 0.0f)).r);
        float wR = saturate(rawMaskTex.sample(linSamp, uv + float2( wakeStep.x, 0.0f)).r);
        float wU = saturate(rawMaskTex.sample(linSamp, uv + float2(0.0f, -wakeStep.y)).r);
        float wD = saturate(rawMaskTex.sample(linSamp, uv + float2(0.0f,  wakeStep.y)).r);
        float mWakeMax = max(m0, max(max(wL, wR), max(wU, wD)));
        float mWakeMin = min(m0, min(min(wL, wR), min(wU, wD)));

        float wakeGone = smoothstep(0.35f, 0.65f, prevMask) * (1.0f - smoothstep(0.12f, 0.34f, mWakeMax));
        float bodyArrived = (1.0f - smoothstep(0.35f, 0.65f, prevMask)) * smoothstep(0.66f, 0.88f, mWakeMin);
        // Only release on large coherent flips (>0.46); normal stationary Vision shimmer stays below ~0.35
        float largeJump = smoothstep(0.46f, 0.76f, abs(p - prevMask));
        float ghostFlush = saturate(max(max(wakeGone, bodyArrived), largeJump));

        float blend = saturate(u.temporalBlend * simWeight * (1.0f - ghostFlush));
        p = mix(p, prevMask, blend);
        // Keep stored reference RGB crisp so rapid consecutive sway frames never smear into prevRGB
        nextStoredRGB = mix(smoothRGB, prevRGB, blend * 0.20f);
    }

    float luma = rgbLuma(smoothRGB);
    nextGuideAndMask.write(float4(nextStoredRGB, p), gid);
    statsOut.write(float4(luma, p, luma * luma, luma * p), gid);
}

// Pass 2a: Separable horizontal box filter
kernel void boxFilterHorizontal(
    texture2d<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant GuidedUniforms& u [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= u.gridSize.x || gid.y >= u.gridSize.y) return;
    int r = u.radius;
    int maxX = int(u.gridSize.x) - 1;
    float4 acc = float4(0.0f);
    for (int dx = -r; dx <= r; ++dx) {
        int sx = clamp(int(gid.x) + dx, 0, maxX);
        acc += src.read(uint2(sx, gid.y));
    }
    dst.write(acc / float(2 * r + 1), gid);
}

// Pass 2b: Separable vertical box filter + linear guided-filter coefficients (a, b)
kernel void boxFilterVerticalAndCoeffs(
    texture2d<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> abOut [[texture(1)]],
    constant GuidedUniforms& u [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= u.gridSize.x || gid.y >= u.gridSize.y) return;
    int r = u.radius;
    int maxY = int(u.gridSize.y) - 1;
    float4 acc = float4(0.0f);
    for (int dy = -r; dy <= r; ++dy) {
        int sy = clamp(int(gid.y) + dy, 0, maxY);
        acc += src.read(uint2(gid.x, sy));
    }
    float4 mean = acc / float(2 * r + 1);
    float varI = max(0.0f, mean.z - mean.x * mean.x);
    float covIP = mean.w - mean.x * mean.y;
    // Clamp linear gain `a` so low-contrast webcam ISO noise can NEVER blow up into boiling halos
    float a = clamp(covIP / (varI + u.eps), -2.2f, 2.2f);
    float b = mean.y - a * mean.x;
    abOut.write(float4(a, b, 0.0f, 0.0f), gid);
}

// Pass 2c: 3x3 smoothing of (a, b)
kernel void smoothGuidedCoeffs(
    texture2d<float, access::read> abIn [[texture(0)]],
    texture2d<float, access::write> abSmoothOut [[texture(1)]],
    constant GuidedUniforms& u [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= u.gridSize.x || gid.y >= u.gridSize.y) return;
    int maxX = int(u.gridSize.x) - 1;
    int maxY = int(u.gridSize.y) - 1;
    float2 acc = float2(0.0f);
    for (int dy = -1; dy <= 1; ++dy) {
        int sy = clamp(int(gid.y) + dy, 0, maxY);
        for (int dx = -1; dx <= 1; ++dx) {
            int sx = clamp(int(gid.x) + dx, 0, maxX);
            acc += abIn.read(uint2(sx, sy)).rg;
        }
    }
    abSmoothOut.write(float4(acc * (1.0f / 9.0f), 0.0f, 0.0f), gid);
}

// Fast downsample pass for ANE Vision input & multi-stop optical bokeh pyramid
kernel void downsampleHalfBGRA(
    texture2d<float, access::sample> plane0 [[texture(0)]],
    texture2d<float, access::sample> plane1 [[texture(1)]],
    texture2d<float, access::write> halfOut [[texture(2)]],
    constant uint3& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 sz = uint2(halfOut.get_width(), halfOut.get_height());
    if (gid.x >= sz.x || gid.y >= sz.y) return;
    constexpr sampler linSamp(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5f) / float2(sz);
    float3 rgb;
    if (params.x != 0) {
        float y = plane0.sample(linSamp, uv).r;
        float2 cbcr = plane1.sample(linSamp, uv).rg;
        rgb = yuvToRGB(y, cbcr, params.y != 0);
    } else {
        rgb = plane0.sample(linSamp, uv).rgb;
    }
    halfOut.write(float4(rgb, 1.0f), gid);
}

// 16-Tap Golden-Angle Vogel Spiral Optical Lens Disc Bokeh + Specular Highlight Bloom (480x270 -> < 0.11 ms GPU!)
// Makes distant hallway lamps, windows, and highlights bloom into real circular f/1.4 optical bokeh discs!
kernel void opticalBokehDiscPass(
    texture2d<float, access::sample> srcNearBlur [[texture(0)]],
    texture2d<float, access::write>  dstFarBokeh [[texture(1)]],
    constant float& radiusScale [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 sz = uint2(dstFarBokeh.get_width(), dstFarBokeh.get_height());
    if (gid.x >= sz.x || gid.y >= sz.y) return;

    constexpr sampler linSamp(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5f) / float2(sz);
    float2 aspectRadius = float2(1.0f / 1.7777f, 1.0f) * (0.012f + 0.026f * saturate(radiusScale));

    float3 weightedSum = float3(0.0f);
    float weightTotal = 0.0f;

    // Center tap + 16 Golden-Angle Vogel disc taps
    float3 centerC = srcNearBlur.sample(linSamp, uv).rgb;
    float centerL = rgbLuma(centerC);
    float centerW = 1.0f + 4.5f * pow(max(0.0f, centerL - 0.62f), 2.0f);
    weightedSum += centerC * centerW;
    weightTotal += centerW;

    const float goldenAngle = 2.39996323f;
    for (int i = 1; i <= 16; ++i) {
        float r = sqrt(float(i) / 16.5f);
        float theta = float(i) * goldenAngle;
        float2 offset = float2(cos(theta), sin(theta)) * r * aspectRadius;
        float3 c = srcNearBlur.sample(linSamp, uv + offset).rgb;
        float l = rgbLuma(c);
        // Optical highlight bloom weight so bright lights form crisp bokeh circles
        float w = 1.0f + 5.5f * pow(max(0.0f, l - 0.60f), 2.0f);
        weightedSum += c * w;
        weightTotal += w;
    }

    dstFarBokeh.write(float4(saturate(weightedSum / max(1e-4f, weightTotal)), 1.0f), gid);
}

inline float3 applyReactionOverlay(float3 color, float2 uv, uint reactionType, float progress, float t) {
    if (reactionType == 0 || progress <= 0.0f || progress >= 1.0f) return color;
    float fade = sin(progress * 3.14159265f);
    if (reactionType == 4) {
        float beam1 = exp(-120.0f * fabs(uv.y - (0.85f - 0.65f * progress) - 0.18f * (uv.x - 0.5f)));
        float beam2 = exp(-120.0f * fabs(uv.y - (0.80f - 0.70f * progress) + 0.22f * (uv.x - 0.5f)));
        color += fade * (beam1 * float3(1.0f, 0.15f, 0.55f) + beam2 * float3(0.15f, 0.85f, 1.0f));
    } else if (reactionType == 3) {
        float2 cell = floor(uv * float2(28.0f, 16.0f) + float2(0.0f, -progress * 10.0f));
        float h = fract(sin(dot(cell, float2(12.9898f, 78.233f))) * 43758.5453f);
        if (h > 0.90f) {
            float3 c = 0.5f + 0.5f * cos(6.28318f * (h + float3(0.0f, 0.33f, 0.67f)) + t * 4.0f);
            color = mix(color, c, fade * 0.85f);
        }
    } else if (reactionType == 2) {
        float2 p = (uv - float2(0.78f, 0.72f - 0.38f * progress)) * float2(16.0f, -9.0f);
        p.y -= sqrt(max(0.0f, fabs(p.x))) * 0.65f;
        float d = length(p) - 0.85f;
        float mask = 1.0f - smoothstep(-0.08f, 0.08f, d);
        color = mix(color, float3(1.0f, 0.20f, 0.36f), mask * fade * 0.92f);
    } else if (reactionType == 1) {
        float2 p = (uv - float2(0.22f, 0.68f - 0.30f * progress)) * float2(1.777f, 1.0f);
        float r = length(p);
        float ring = exp(-90.0f * fabs(r - (0.04f + 0.08f * progress)));
        color += ring * fade * float3(0.25f, 0.85f, 1.0f);
    }
    return saturate(color);
}

// Pass 3: Single Fused Studio Compositor
kernel void compositeStudioFrame(
    texture2d<float, access::sample> camPlane0      [[texture(0)]],
    texture2d<float, access::sample> camPlane1      [[texture(1)]],
    texture2d<float, access::sample> abTex          [[texture(2)]],
    texture2d<float, access::sample> bgPlane0       [[texture(3)]],
    texture2d<float, access::sample> bgPlane1       [[texture(4)]],
    texture2d<float, access::sample> nearBlurTex    [[texture(5)]],
    texture2d<float, access::write>  outBGRA        [[texture(6)]],
    texture3d<float, access::sample> lutTex3D       [[texture(7)]],
    texture2d<float, access::sample> farBokehTex    [[texture(8)]],
    texture2d<float, access::sample> sceneDepthTex  [[texture(9)]],
    texture2d<float, access::sample> guideMaskTex   [[texture(10)]],
    constant CompositeUniforms& u [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= u.outputSize.x || gid.y >= u.outputSize.y) return;

    constexpr sampler linSamp(coord::normalized, address::clamp_to_edge, filter::linear);

    float2 dstUV = (float2(gid) + 0.5f) / float2(u.outputSize);
    float2 sampleUV = dstUV;
    if (u.mirrorHorizontal != 0) {
        sampleUV.x = 1.0f - sampleUV.x;
    }

    // 1. Lossless Auto-Framing Crop + Aspect-Fill Camera UV
    float2 croppedUV = clamp(u.camCropRect.xy + sampleUV * u.camCropRect.zw, 0.0005f, 0.9995f);
    float2 camUV = aspectFillUV(croppedUV, u.camSize, u.outputSize);

    float3 fgRGB;
    if (u.camIsYUV != 0) {
        float y = camPlane0.sample(linSamp, camUV).r;
        float2 cbcr = camPlane1.sample(linSamp, camUV).rg;
        fgRGB = yuvToRGB(y, cbcr, u.camFullRange != 0);
    } else {
        fgRGB = camPlane0.sample(linSamp, camUV).rgb;
    }
    float fgLuma = rgbLuma(fgRGB);

    // 2. Hybrid RGB Joint-Bilateral + Clamped Luma Guided Hair Matte
    // Solves BOTH iso-luma color boundaries (blue shirt vs beige wall) AND fine hair wisps without noise boil!
    float2 ab = abTex.sample(linSamp, camUV).rg;
    float guidedAlpha = saturate(ab.x * fgLuma + ab.y);

    // 5-tap RGB color-distance joint-bilateral upsample of stabilized temporal matte
    float2 gStep = float2(1.5f / 512.0f, 1.5f / 288.0f);
    float2 offsets[5] = {
        float2(0.0f, 0.0f),
        float2(-gStep.x, 0.0f), float2(gStep.x, 0.0f),
        float2(0.0f, -gStep.y), float2(0.0f, gStep.y)
    };
    float jbSum = 0.0f;
    float jbW = 0.0f;
    const float invTwoSigma2 = 1.0f / (2.0f * 0.085f * 0.085f);
    for (int k = 0; k < 5; ++k) {
        float4 gm = guideMaskTex.sample(linSamp, camUV + offsets[k]);
        float3 diff = fgRGB - gm.rgb;
        float w = exp(-dot(diff, diff) * invTwoSigma2) * (k == 0 ? 1.5f : 1.0f);
        jbSum += gm.a * w;
        jbW += w;
    }
    float jbAlpha = (jbW > 1e-4f) ? (jbSum / jbW) : guidedAlpha;

    // Blend RGB joint-bilateral (65%, super calm + color-accurate) with clamped luma guided (35%, fine hair strands)
    float rawAlpha = saturate(mix(jbAlpha, guidedAlpha, 0.35f) - u.matteTuning.x);

    // Calibrated feather curve (avoids harsh high-slope smoothstep edge boiling!)
    float feather = clamp(u.edgeSoftness, 0.08f, 0.95f);
    float lowEdge = mix(0.24f, 0.06f, feather);
    float highEdge = mix(0.76f, 0.94f, feather);
    float linearAlpha = saturate((rawAlpha - lowEdge) / max(0.05f, highEdge - lowEdge));
    // Gentle S-curve (only 35% smoothstep blend so edge derivative stays calm & anti-shimmer)
    float alpha = mix(linearAlpha, linearAlpha * linearAlpha * (3.0f - 2.0f * linearAlpha), 0.35f);

    // 2b. Subtle Face UVC De-Harshness (softens digital webcam forehead/under-eye oversharpening, keeps eyes/glasses sharp)
    if (u.skinPolishStrength > 0.01f && alpha > 0.65f) {
        float2 faceCenter = u.faceRect.xy;
        if (u.mirrorHorizontal != 0) faceCenter.x = 1.0f - faceCenter.x;
        float2 fd = (dstUV - faceCenter) / max(u.faceRect.zw * 1.15f, float2(0.10f, 0.14f));
        float inFace = 1.0f - smoothstep(0.65f, 1.25f, length(fd));
        bool warmSkin = (fgRGB.r > fgRGB.g * 0.95f) && (fgRGB.g > fgRGB.b * 0.82f);
        if (inFace > 0.01f && warmSkin) {
            // 4-tap local micro-neighbor smooth that preserves high-contrast features (eyes/glasses/stubble)
            float2 stepUV = 1.35f / float2(u.outputSize);
            float3 n1 = u.camIsYUV != 0
                ? yuvToRGB(camPlane0.sample(linSamp, camUV + float2(stepUV.x, 0)).r, camPlane1.sample(linSamp, camUV + float2(stepUV.x, 0)).rg, u.camFullRange != 0)
                : camPlane0.sample(linSamp, camUV + float2(stepUV.x, 0)).rgb;
            float3 n2 = u.camIsYUV != 0
                ? yuvToRGB(camPlane0.sample(linSamp, camUV - float2(stepUV.x, 0)).r, camPlane1.sample(linSamp, camUV - float2(stepUV.x, 0)).rg, u.camFullRange != 0)
                : camPlane0.sample(linSamp, camUV - float2(stepUV.x, 0)).rgb;
            float3 localAvg = 0.5f * (n1 + n2);
            float detailDiff = length(fgRGB - localAvg);
            float edgeKeep = 1.0f - smoothstep(0.025f, 0.095f, detailDiff);
            fgRGB = mix(fgRGB, localAvg, inFace * edgeKeep * saturate(u.skinPolishStrength) * 0.55f);
            fgLuma = rgbLuma(fgRGB);
        }
    }

    // 3. Evaluate Background Layer + True 3D Scene-Depth Graded Optical Bokeh (works on Real Room AND Virtual Rooms!)
    float3 bgRGB = fgRGB;
    float2 activeSceneUV = camUV;
    if (u.bgMode == 0) {
        activeSceneUV = camUV;
        if (u.portraitEnabled != 0 && u.portraitStrength > 0.01f) {
            float3 nearSoft = nearBlurTex.sample(linSamp, camUV).rgb;
            if (u.depthBokehEnabled != 0) {
                float dist = saturate(sceneDepthTex.sample(linSamp, camUV).r);
                float coc = saturate((dist - u.focalPlaneOffset) / max(0.12f, 1.0f - u.focalPlaneOffset)) * saturate(u.portraitStrength);
                float3 farBokeh = farBokehTex.sample(linSamp, camUV).rgb;
                // Progressive 3-stop optical lens curve: Sharp -> Gentle Near Blur -> Deep Golden-Angle Hallway Bokeh
                float3 stage1 = mix(fgRGB, nearSoft, smoothstep(0.0f, 0.42f, coc));
                bgRGB = mix(stage1, farBokeh, smoothstep(0.32f, 0.95f, coc));
            } else {
                bgRGB = mix(fgRGB, nearSoft, saturate(u.portraitStrength));
            }
        }
    } else if (u.bgMode == 1) {
        float t = saturate(dstUV.y * 0.85f + dstUV.x * 0.15f);
        bgRGB = mix(u.gradTopColor.rgb, u.gradBottomColor.rgb, smoothstep(0.0f, 1.0f, t));
    } else {
        float2 bgUV = aspectFillUV(dstUV, u.bgSize, u.outputSize);
        activeSceneUV = bgUV;
        float3 sharpBG;
        if (u.bgIsYUV != 0) {
            float by = bgPlane0.sample(linSamp, bgUV).r;
            float2 bcbcr = bgPlane1.sample(linSamp, bgUV).rg;
            sharpBG = yuvToRGB(by, bcbcr, false);
        } else {
            sharpBG = bgPlane0.sample(linSamp, bgUV).rgb;
        }

        if (u.portraitEnabled != 0 && u.portraitStrength > 0.01f) {
            float3 nearSoft = nearBlurTex.sample(linSamp, bgUV).rgb;
            if (u.depthBokehEnabled != 0) {
                // Depth map computed directly on the virtual room image/video!
                float dist = saturate(sceneDepthTex.sample(linSamp, bgUV).r);
                float coc = saturate((dist - u.focalPlaneOffset) / max(0.12f, 1.0f - u.focalPlaneOffset)) * saturate(u.portraitStrength);
                float3 farBokeh = farBokehTex.sample(linSamp, bgUV).rgb;
                float3 stage1 = mix(sharpBG, nearSoft, smoothstep(0.0f, 0.42f, coc));
                bgRGB = mix(stage1, farBokeh, smoothstep(0.32f, 0.95f, coc));
            } else {
                bgRGB = mix(sharpBG, nearSoft, saturate(u.portraitStrength));
            }
        } else {
            bgRGB = sharpBG;
        }
    }

    // 4. Hair Color-Spill Suppression + Live Ambient Color Match + Optical Light Wrap
    if (u.bgMode != 0) {
        // 4a. Live Ambient Room Color Match (nudges subject tint to match virtual room lighting dynamically)
        if (u.ambientMatchStrength > 0.01f && alpha > 0.05f) {
            float3 roomAmb = (u.bgMode == 1)
                ? 0.5f * (u.gradTopColor.rgb + u.gradBottomColor.rgb)
                : 0.25f * (
                    farBokehTex.sample(linSamp, float2(0.25f, 0.35f)).rgb +
                    farBokehTex.sample(linSamp, float2(0.75f, 0.35f)).rgb +
                    farBokehTex.sample(linSamp, float2(0.25f, 0.65f)).rgb +
                    farBokehTex.sample(linSamp, float2(0.75f, 0.65f)).rgb
                );
            float ambLuma = max(0.08f, rgbLuma(roomAmb));
            float3 ambTint = clamp(roomAmb / ambLuma, float3(0.78f), float3(1.26f));
            fgRGB = saturate(fgRGB * mix(float3(1.0f), ambTint, saturate(u.ambientMatchStrength) * 0.28f));
        }

        // 4b. Semi-transparent hair spill suppression
        if (alpha > 0.03f && alpha < 0.94f && u.spillSuppression > 0.01f) {
            float edgeBand = 4.0f * alpha * (1.0f - alpha);
            float3 neutralFG = float3(fgLuma);
            float bgLuma = max(0.05f, rgbLuma(bgRGB));
            float3 ambientTintedFG = neutralFG * mix(float3(1.0f), bgRGB / bgLuma, 0.28f);
            fgRGB = mix(fgRGB, ambientTintedFG, edgeBand * u.spillSuppression * 0.65f);
        }

        // 4c. Optical Light Wrap (bleeds bright virtual background glow softly over subject rim!)
        if (u.lightWrapStrength > 0.01f && alpha > 0.10f) {
            float3 localBGGlow = (u.bgMode == 1) ? bgRGB : nearBlurTex.sample(linSamp, activeSceneUV).rgb;
            float glowEnergy = smoothstep(0.25f, 0.95f, rgbLuma(localBGGlow));
            float rimWrap = (1.0f - smoothstep(0.30f, 0.98f, alpha)) * smoothstep(0.05f, 0.45f, alpha);
            fgRGB = saturate(fgRGB + localBGGlow * (glowEnergy * rimWrap * saturate(u.lightWrapStrength) * 0.55f));
        }
    }

    // 5. Studio Light (SRLv2 contour lift + background dimming)
    if (u.studioLightEnabled != 0 && u.studioLightStrength > 0.001f) {
        float s = saturate(u.studioLightStrength);
        bgRGB *= mix(1.0f, 0.74f, s);

        float2 faceCenter = u.faceRect.xy;
        if (u.mirrorHorizontal != 0) faceCenter.x = 1.0f - faceCenter.x;
        float2 radii = max(u.faceRect.zw, float2(0.18f, 0.28f));
        float2 d = (dstUV - faceCenter) / (radii * 1.65f);
        float keyFalloff = exp(-0.5f * dot(d, d));
        float shadowLift = (1.0f - fgLuma) * fgLuma * 4.0f;
        fgRGB = saturate(fgRGB + fgRGB * (0.28f * s * keyFalloff * (0.45f + 0.55f * shadowLift)));
    }

    // 6. Edge Light
    if (u.edgeLightEnabled != 0 && u.edgeLightStrength > 0.001f) {
        float e = saturate(u.edgeLightStrength);
        float3 lightTint = mix(float3(0.92f, 0.96f, 1.0f), float3(1.0f, 0.88f, 0.74f), saturate(u.edgeLightWarmth));
        float rim = 4.0f * alpha * (1.0f - alpha);
        float borderProx = 1.0f - (4.0f * dstUV.x * (1.0f - dstUV.x)) * (4.0f * dstUV.y * (1.0f - dstUV.y));
        float fill = 0.14f * e + 0.22f * e * rim + 0.08f * e * borderProx;
        fgRGB = saturate(fgRGB + lightTint * fill * (1.0f - 0.55f * fgRGB));
    }

    // 7. Final Guided Alpha Composite
    bool needComposite = (u.bgMode != 0) || (u.portraitEnabled != 0) || (u.studioLightEnabled != 0);
    float3 finalRGB = needComposite ? mix(bgRGB, fgRGB, alpha) : fgRGB;

    // 8. Whole-Composite Trilinear 3D Color LUT
    if (u.lutEnabled != 0 && u.lutDimension >= 2 && u.lutStrength > 0.001f) {
        float3 domRange = max(float3(1e-5f), u.lutDomainMax.rgb - u.lutDomainMin.rgb);
        float3 normRGB = saturate((finalRGB - u.lutDomainMin.rgb) / domRange);
        float dimF = float(u.lutDimension);
        float3 lutCoord = normRGB * ((dimF - 1.0f) / dimF) + (0.5f / dimF);
        float3 gradedRGB = lutTex3D.sample(linSamp, lutCoord).rgb;
        finalRGB = mix(finalRGB, saturate(gradedRGB), saturate(u.lutStrength));
    }

    // 9. Unified 35mm Film / Camera Sensor Grain (marries foreground + virtual BG sensor texture!)
    if (u.filmGrainStrength > 0.005f) {
        float2 seed = float2(gid) + float2(fract(u.timeSeconds * 24.0f) * 197.3f, fract(u.timeSeconds * 17.0f) * 311.7f);
        float n = fract(sin(dot(seed, float2(12.9898f, 78.233f))) * 43758.5453f) - 0.5f;
        float yFinal = rgbLuma(finalRGB);
        float midtoneMask = 4.0f * yFinal * (1.0f - yFinal);
        finalRGB = saturate(finalRGB + n * (0.065f * saturate(u.filmGrainStrength) * (0.30f + 0.70f * midtoneMask)));
    }

    // 10. Optional Reaction Overlay
    if (u.reactionType != 0) {
        finalRGB = applyReactionOverlay(finalRGB, dstUV, u.reactionType, u.reactionProgress, u.timeSeconds);
    }

    outBGRA.write(float4(saturate(finalRGB), 1.0f), gid);
}

// GPU scaler (only invoked if an explicit 4K local override is simultaneously feeding a 1080p sink)
kernel void scaleBGRAForSink(
    texture2d<float, access::sample> srcBGRA [[texture(0)]],
    texture2d<float, access::write>  dstBGRA [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dstSize = uint2(dstBGRA.get_width(), dstBGRA.get_height());
    if (gid.x >= dstSize.x || gid.y >= dstSize.y) return;
    constexpr sampler linSamp(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5f) / float2(dstSize);
    float2 texel = 0.25f / float2(dstSize);
    float3 c = 0.25f * (
        srcBGRA.sample(linSamp, uv + float2(-texel.x, -texel.y)).rgb +
        srcBGRA.sample(linSamp, uv + float2( texel.x, -texel.y)).rgb +
        srcBGRA.sample(linSamp, uv + float2(-texel.x,  texel.y)).rgb +
        srcBGRA.sample(linSamp, uv + float2( texel.x,  texel.y)).rgb
    );
    dstBGRA.write(float4(c, 1.0f), gid);
}
