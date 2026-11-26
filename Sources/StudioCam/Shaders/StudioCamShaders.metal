#include <metal_stdlib>
using namespace metal;

struct CompositeUniforms {
    uint2 outputSize;
    uint2 camSize;
    uint2 bgSize;

    uint camIsYUV;          // 1 = biplanar NV12 (420v/420f), 0 = BGRA/RGBA
    uint camFullRange;      // 1 = 420f, 0 = 420v
    uint bgMode;            // 0 = none (original cam), 1 = gradient, 2 = static image (BGRA), 3 = video (NV12/BGRA)
    uint bgIsYUV;           // 1 = biplanar NV12, 0 = BGRA/RGBA

    uint portraitEnabled;
    float portraitStrength; // 0..1

    uint studioLightEnabled;
    float studioLightStrength; // 0..1

    uint edgeLightEnabled;
    float edgeLightStrength;   // 0..1
    float edgeLightWarmth;     // 0 = cool daylight, 1 = warm tungsten

    float spillSuppression;    // 0..1
    float edgeSoftness;        // 0..1
    uint mirrorHorizontal;     // 0 or 1

    float4 gradTopColor;
    float4 gradBottomColor;
    float4 faceRect;           // normalized [cx, cy, rx, ry] in UV space

    uint reactionType;         // 0 = none, 1 = thumbsUp, 2 = heart, 3 = confetti, 4 = lasers
    float reactionProgress;    // 0..1
    float timeSeconds;
    float lutStrength;         // 0..1

    uint lutEnabled;           // 0 or 1
    uint lutDimension;         // e.g. 33
    uint _lutPad0;
    uint _lutPad1;
    float4 lutDomainMin;
    float4 lutDomainMax;
};

struct GuidedUniforms {
    uint2 gridSize;
    int radius;
    float eps;
    float temporalBlend;       // e.g. 0.78
    float colorSimSigma;       // e.g. 0.055
    uint hasHistory;
    uint camIsYUV;
    uint camFullRange;
};

inline float3 yuvToRGB(float ySample, float2 cbcrSample, bool fullRange) {
    float y = fullRange ? ySample : ((ySample - (16.0f / 255.0f)) * (255.0f / 219.0f));
    float cb = cbcrSample.x - 0.5f;
    float cr = cbcrSample.y - 0.5f;
    // Rec.709 conversion
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

// Pass 1: Sample camera guide + ANE segmentation mask, apply Portrait.framework-style
// color-similarity-gated temporal EMA, and pack (I, p, I*I, I*p) for O(1) separable box filtering.
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
    float2 uv = (float2(gid) + 0.5f) / float2(u.gridSize);

    float3 rgb;
    if (u.camIsYUV != 0) {
        float y = camPlane0.sample(linSamp, uv).r;
        float2 cbcr = camPlane1.sample(linSamp, uv).rg;
        rgb = yuvToRGB(y, cbcr, u.camFullRange != 0);
    } else {
        rgb = camPlane0.sample(linSamp, uv).rgb;
    }

    float luma = rgbLuma(rgb);
    float p = saturate(rawMaskTex.sample(linSamp, uv).r);

    // Color-similarity-gated temporal EMA (modeled on Portrait.framework's
    // temporalFilterExponentialMovingAverageColorSimilarities):
    // Steady color -> high temporal stability (zero edge shimmer);
    // Sudden color change -> instant responsiveness (zero motion ghosting).
    if (u.hasHistory != 0) {
        float4 prev = prevGuideAndMask.read(gid);
        float3 prevRGB = prev.rgb;
        float prevMask = prev.a;
        float colorDiff = length(rgb - prevRGB);
        float simWeight = exp(-(colorDiff * colorDiff) / max(1e-5f, 2.0f * u.colorSimSigma * u.colorSimSigma));
        float maskDelta = fabs(p - prevMask);
        float motionGate = 1.0f - smoothstep(0.18f, 0.55f, maskDelta);
        float blend = u.temporalBlend * simWeight * motionGate;
        p = mix(p, prevMask, saturate(blend));
    }

    nextGuideAndMask.write(float4(rgb, p), gid);
    statsOut.write(float4(luma, p, luma * luma, luma * p), gid);
}

// Pass 2a: Separable horizontal box filter on (I, p, I*I, I*p)
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
    float4 mean = acc / float(2 * r + 1); // (meanI, meanP, meanII, meanIP)

    float varI = max(0.0f, mean.z - mean.x * mean.x);
    float covIP = mean.w - mean.x * mean.y;
    float a = covIP / (varI + u.eps);
    float b = mean.y - a * mean.x;
    abOut.write(float4(a, b, 0.0f, 0.0f), gid);
}

// Pass 2c: Lightweight 3x3 mean of (a, b) for smooth hair-edge upsampling
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

// Optional downsample pass for fast MPSImageGaussianBlur when Portrait bokeh is active
kernel void downsampleHalfBGRA(
    texture2d<float, access::sample> plane0 [[texture(0)]],
    texture2d<float, access::sample> plane1 [[texture(1)]],
    texture2d<float, access::write> halfOut [[texture(2)]],
    constant uint3& params [[buffer(0)]], // x=isYUV, y=fullRange, z=unused
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

// Procedural subtle Reaction overlay (zero external texture overhead)
inline float3 applyReactionOverlay(float3 color, float2 uv, uint reactionType, float progress, float t) {
    if (reactionType == 0 || progress <= 0.0f || progress >= 1.0f) return color;
    float fade = sin(progress * 3.14159265f);
    if (reactionType == 4) {
        // Lasers sweep
        float beam1 = exp(-120.0f * fabs(uv.y - (0.85f - 0.65f * progress) - 0.18f * (uv.x - 0.5f)));
        float beam2 = exp(-120.0f * fabs(uv.y - (0.80f - 0.70f * progress) + 0.22f * (uv.x - 0.5f)));
        color += fade * (beam1 * float3(1.0f, 0.15f, 0.55f) + beam2 * float3(0.15f, 0.85f, 1.0f));
    } else if (reactionType == 3) {
        // Confetti sparkles
        float2 cell = floor(uv * float2(28.0f, 16.0f) + float2(0.0f, -progress * 10.0f));
        float h = fract(sin(dot(cell, float2(12.9898f, 78.233f))) * 43758.5453f);
        if (h > 0.90f) {
            float3 c = 0.5f + 0.5f * cos(6.28318f * (h + float3(0.0f, 0.33f, 0.67f)) + t * 4.0f);
            color = mix(color, c, fade * 0.85f);
        }
    } else if (reactionType == 2) {
        // Floating warm heart glow
        float2 p = (uv - float2(0.78f, 0.72f - 0.38f * progress)) * float2(16.0f, -9.0f);
        p.y -= sqrt(max(0.0f, fabs(p.x))) * 0.65f;
        float d = length(p) - 0.85f;
        float mask = 1.0f - smoothstep(-0.08f, 0.08f, d);
        color = mix(color, float3(1.0f, 0.20f, 0.36f), mask * fade * 0.92f);
    } else if (reactionType == 1) {
        // Crisp glowing ring-burst badge
        float2 p = (uv - float2(0.22f, 0.68f - 0.30f * progress)) * float2(1.777f, 1.0f);
        float r = length(p);
        float ring = exp(-90.0f * fabs(r - (0.04f + 0.08f * progress)));
        color += ring * fade * float3(0.25f, 0.85f, 1.0f);
    }
    return saturate(color);
}

// Pass 3: Single Fused 4K Studio Compositor
// Evaluates guided 4K hair matte, color-spill suppression, 4K video/image/gradient background,
// Portrait depth bokeh, SRLv2-style Studio Light, and Edge Light in one pass.
kernel void compositeStudioFrame(
    texture2d<float, access::sample> camPlane0   [[texture(0)]],
    texture2d<float, access::sample> camPlane1   [[texture(1)]],
    texture2d<float, access::sample> abTex       [[texture(2)]],
    texture2d<float, access::sample> bgPlane0    [[texture(3)]],
    texture2d<float, access::sample> bgPlane1    [[texture(4)]],
    texture2d<float, access::sample> blurredTex  [[texture(5)]],
    texture2d<float, access::write>  outBGRA     [[texture(6)]],
    texture3d<float, access::sample> lutTex3D    [[texture(7)]],
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

    // 1. Aspect-fill camera UV + full-4K foreground RGB & Luma
    float2 camUV = aspectFillUV(sampleUV, u.camSize, u.outputSize);
    float3 fgRGB;
    if (u.camIsYUV != 0) {
        float y = camPlane0.sample(linSamp, camUV).r;
        float2 cbcr = camPlane1.sample(linSamp, camUV).rg;
        fgRGB = yuvToRGB(y, cbcr, u.camFullRange != 0);
    } else {
        fgRGB = camPlane0.sample(linSamp, camUV).rgb;
    }
    float fgLuma = rgbLuma(fgRGB);

    // 2. Evaluate 4K Guided Hair Matte: alpha = a * I_4K + b
    float2 ab = abTex.sample(linSamp, camUV).rg;
    float rawAlpha = saturate(ab.x * fgLuma + ab.y);
    float lowEdge = mix(0.02f, 0.14f, 1.0f - u.edgeSoftness);
    float highEdge = mix(0.98f, 0.86f, 1.0f - u.edgeSoftness);
    float alpha = smoothstep(lowEdge, highEdge, rawAlpha);

    // 3. Evaluate Background Layer (Original / Gradient / 4K Static Image / 4K Video + Portrait Bokeh)
    float3 bgRGB = fgRGB;
    if (u.bgMode == 0) {
        // Real room background (with Portrait blur when enabled)
        if (u.portraitEnabled != 0) {
            float3 blurredCam = blurredTex.sample(linSamp, camUV).rgb;
            bgRGB = mix(fgRGB, blurredCam, saturate(u.portraitStrength));
        }
    } else if (u.bgMode == 1) {
        // Studio vertical/diagonal gradient (matching Portrait.framework gradient style)
        float t = saturate(dstUV.y * 0.85f + dstUV.x * 0.15f);
        bgRGB = mix(u.gradTopColor.rgb, u.gradBottomColor.rgb, smoothstep(0.0f, 1.0f, t));
    } else {
        // Static Image (bgMode == 2) or 4K Looping Video (bgMode == 3)
        float2 bgUV = aspectFillUV(dstUV, u.bgSize, u.outputSize);
        if (u.portraitEnabled != 0 && u.portraitStrength > 0.01f) {
            float3 sharpBG;
            if (u.bgIsYUV != 0) {
                float by = bgPlane0.sample(linSamp, bgUV).r;
                float2 bcbcr = bgPlane1.sample(linSamp, bgUV).rg;
                sharpBG = yuvToRGB(by, bcbcr, false);
            } else {
                sharpBG = bgPlane0.sample(linSamp, bgUV).rgb;
            }
            float3 softBG = blurredTex.sample(linSamp, bgUV).rgb;
            bgRGB = mix(sharpBG, softBG, saturate(u.portraitStrength));
        } else {
            if (u.bgIsYUV != 0) {
                float by = bgPlane0.sample(linSamp, bgUV).r;
                float2 bcbcr = bgPlane1.sample(linSamp, bgUV).rg;
                bgRGB = yuvToRGB(by, bcbcr, false);
            } else {
                bgRGB = bgPlane0.sample(linSamp, bgUV).rgb;
            }
        }
    }

    // 4. Hair / Semi-Transparent Edge Color-Spill Suppression (modeled on PTSpillCorrection)
    // Neutralizes room wall/monitor tint bleeding through fine hair wisps and harmonizes with new BG.
    if (u.bgMode != 0 && alpha > 0.03f && alpha < 0.94f && u.spillSuppression > 0.01f) {
        float edgeBand = 4.0f * alpha * (1.0f - alpha); // peaks at alpha = 0.5
        float3 neutralFG = float3(fgLuma);
        float bgLuma = max(0.05f, rgbLuma(bgRGB));
        float3 ambientTintedFG = neutralFG * mix(float3(1.0f), bgRGB / bgLuma, 0.25f);
        fgRGB = mix(fgRGB, ambientTintedFG, edgeBand * u.spillSuppression * 0.65f);
    }

    // 5. Studio Light (modeled on Portrait.framework SRLv2: face/subject contour lift + BG dimming)
    if (u.studioLightEnabled != 0 && u.studioLightStrength > 0.001f) {
        float s = saturate(u.studioLightStrength);
        // Subtle studio background darkening (studiolightBackgroundDimming)
        bgRGB *= mix(1.0f, 0.72f, s);

        // Soft elliptical face/torso key light centered on tracked face
        float2 faceCenter = u.faceRect.xy;
        if (u.mirrorHorizontal != 0) {
            faceCenter.x = 1.0f - faceCenter.x;
        }
        float2 radii = max(u.faceRect.zw, float2(0.18f, 0.28f));
        float2 d = (dstUV - faceCenter) / (radii * 1.65f);
        float keyFalloff = exp(-0.5f * dot(d, d));

        // Lift midtones and shadows without clipping highlights (SRLv2 tone curve)
        float shadowLift = (1.0f - fgLuma) * fgLuma * 4.0f;
        fgRGB += fgRGB * (0.30f * s * keyFalloff * (0.45f + 0.55f * shadowLift));
        fgRGB = saturate(fgRGB);
    }

    // 6. Edge Light (rim & ring-light subject illumination)
    if (u.edgeLightEnabled != 0 && u.edgeLightStrength > 0.001f) {
        float e = saturate(u.edgeLightStrength);
        float3 coolTint = float3(0.92f, 0.96f, 1.0f);
        float3 warmTint = float3(1.0f, 0.88f, 0.74f);
        float3 lightTint = mix(coolTint, warmTint, saturate(u.edgeLightWarmth));

        // Rim glow along silhouette + soft fill from screen borders
        float rim = 4.0f * alpha * (1.0f - alpha);
        float borderProx = 1.0f - (4.0f * dstUV.x * (1.0f - dstUV.x)) * (4.0f * dstUV.y * (1.0f - dstUV.y));
        float fill = 0.14f * e + 0.22f * e * rim + 0.08f * e * borderProx;
        fgRGB = saturate(fgRGB + lightTint * fill * (1.0f - 0.55f * fgRGB));
    }

    // 7. Final Alpha Composite (Premultiplied-free crisp guided blend)
    bool needComposite = (u.bgMode != 0) || (u.portraitEnabled != 0) || (u.studioLightEnabled != 0);
    float3 finalRGB = needComposite ? mix(bgRGB, fgRGB, alpha) : fgRGB;

    // 8. Whole-Composite Trilinear 3D Color LUT (grades foreground + background cohesively in 1 fetch!)
    if (u.lutEnabled != 0 && u.lutDimension >= 2 && u.lutStrength > 0.001f) {
        float3 domRange = max(float3(1e-5f), u.lutDomainMax.rgb - u.lutDomainMin.rgb);
        float3 normRGB = saturate((finalRGB - u.lutDomainMin.rgb) / domRange);
        float dimF = float(u.lutDimension);
        // Half-texel center mapping so identity .cube LUTs have 0.0000 black/white drift
        float3 lutCoord = normRGB * ((dimF - 1.0f) / dimF) + (0.5f / dimF);
        float3 gradedRGB = lutTex3D.sample(linSamp, lutCoord).rgb;
        finalRGB = mix(finalRGB, saturate(gradedRGB), saturate(u.lutStrength));
    }

    // 9. Optional Reaction Overlay
    if (u.reactionType != 0) {
        finalRGB = applyReactionOverlay(finalRGB, dstUV, u.reactionType, u.reactionProgress, u.timeSeconds);
    }

    // Write BGRA8Unorm -> Metal .bgra8Unorm swizzles float4(R, G, B, A) automatically!
    outBGRA.write(float4(saturate(finalRGB), 1.0f), gid);
}

// Fast Lanczos/Bilinear GPU scaler for feeding 1080p Virtual Camera Sink from 4K composited texture
kernel void scaleBGRAForSink(
    texture2d<float, access::sample> srcBGRA [[texture(0)]],
    texture2d<float, access::write>  dstBGRA [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dstSize = uint2(dstBGRA.get_width(), dstBGRA.get_height());
    if (gid.x >= dstSize.x || gid.y >= dstSize.y) return;
    constexpr sampler linSamp(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5f) / float2(dstSize);
    // 4-tap box-tent supersample for crisp 4K -> 1080p downscale
    float2 texel = 0.25f / float2(dstSize);
    float3 c = 0.25f * (
        srcBGRA.sample(linSamp, uv + float2(-texel.x, -texel.y)).rgb +
        srcBGRA.sample(linSamp, uv + float2( texel.x, -texel.y)).rgb +
        srcBGRA.sample(linSamp, uv + float2(-texel.x,  texel.y)).rgb +
        srcBGRA.sample(linSamp, uv + float2( texel.x,  texel.y)).rgb
    );
    dstBGRA.write(float4(c, 1.0f), gid);
}
