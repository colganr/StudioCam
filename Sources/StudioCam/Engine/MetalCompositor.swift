import Foundation
import Metal
import MetalPerformanceShaders
import CoreVideo
import IOSurface
import QuartzCore
import simd

/// Exact 240-byte 16-byte-aligned match to `CompositeUniforms` in `StudioCamShaders.metal`.
private struct CompositeUniformsGPU {
    var outputSize: SIMD2<UInt32>          // 0
    var camSize: SIMD2<UInt32>             // 8
    var bgSize: SIMD2<UInt32>              // 16
    var camIsYUV: UInt32                   // 24
    var camFullRange: UInt32               // 28

    var bgMode: UInt32                     // 32
    var bgIsYUV: UInt32                    // 36
    var portraitEnabled: UInt32            // 40
    var portraitStrength: Float            // 44

    var depthBokehEnabled: UInt32          // 48
    var focalPlaneOffset: Float            // 52
    var studioLightEnabled: UInt32         // 56
    var studioLightStrength: Float         // 60

    var edgeLightEnabled: UInt32           // 64
    var edgeLightStrength: Float           // 68
    var edgeLightWarmth: Float             // 72
    var spillSuppression: Float            // 76

    var edgeSoftness: Float                // 80
    var mirrorHorizontal: UInt32           // 84
    var lightWrapStrength: Float           // 88
    var ambientMatchStrength: Float        // 92

    var skinPolishStrength: Float          // 96
    var filmGrainStrength: Float           // 100
    var reactionType: UInt32               // 104
    var reactionProgress: Float            // 108

    var timeSeconds: Float                 // 112
    var lutStrength: Float                 // 116
    var lutEnabled: UInt32                 // 120
    var lutDimension: UInt32               // 124

    var gradTopColor: SIMD4<Float>         // 128
    var gradBottomColor: SIMD4<Float>      // 144
    var faceRect: SIMD4<Float>             // 160
    var camCropRect: SIMD4<Float>          // 176
    var lutDomainMin: SIMD4<Float>         // 192
    var lutDomainMax: SIMD4<Float>         // 208
    var matteTuning: SIMD4<Float>          // 224: x=matteChoke, y=temporalStability
}

private struct GuidedUniformsGPU {
    var gridSize: SIMD2<UInt32>
    var radius: Int32
    var eps: Float
    var temporalBlend: Float
    var colorSimSigma: Float
    var hasHistory: UInt32
    var camIsYUV: UInt32
    var camFullRange: UInt32
}

public final class MetalStudioCompositor {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private var textureCache: CVMetalTextureCache!

    private let psoLumaCoarse: MTLComputePipelineState
    private let psoLKCoarseFlow: MTLComputePipelineState
    private let psoGuidedPrep: MTLComputePipelineState
    private let psoBoxH: MTLComputePipelineState
    private let psoBoxV: MTLComputePipelineState
    private let psoSmoothAB: MTLComputePipelineState
    private let psoFMLFECoarseH: MTLComputePipelineState
    private let psoFMLFECoarseV: MTLComputePipelineState
    private let psoFMLFEFine: MTLComputePipelineState
    private let psoDownsampleHalf: MTLComputePipelineState
    private let psoOpticalBokehDisc: MTLComputePipelineState
    private let psoComposite: MTLComputePipelineState
    private let psoScaleSink: MTLComputePipelineState

    private let guideWidth = 512
    private let guideHeight = 288
    private let coarseWidth = 128
    private let coarseHeight = 72
    private var lumaCoarseA: MTLTexture
    private var lumaCoarseB: MTLTexture
    private var coarseFlowTex: MTLTexture
    private var historyTexA: MTLTexture
    private var historyTexB: MTLTexture
    private var pingHistoryToggle = false
    private var hasHistory = false

    private var statsTex: MTLTexture
    private var boxTempTex: MTLTexture
    private var abRawTex: MTLTexture
    private var abSmoothTex: MTLTexture
    private var fmlfeTempTex: MTLTexture
    private var fmlfeCoarseTex: MTLTexture
    private var fmlfeFineTex: MTLTexture
    private var dummy1x1Tex: MTLTexture
    private var fallbackDepthTex: MTLTexture
    private var dummy3DLUTTex: MTLTexture

    private let lutLock = NSLock()
    private var activeLUT: Loaded3DLUT?

    // Multi-stop optical bokeh working textures (960x540 nearBlur + 480x270 Golden-Angle farDiscBokeh)
    private var halfSrcTex: MTLTexture?
    private var nearBlurTex: MTLTexture?
    private var farBokehTex: MTLTexture?
    private var cachedBlurSigma: Float = -1
    private var gaussianBlur: MPSImageGaussianBlur?

    // IOSurface-backed output pools
    private var mainOutputPool: CVPixelBufferPool?
    private var mainPoolSize: (Int, Int) = (0, 0)
    private var sinkOutputPool: CVPixelBufferPool?
    private var sinkPoolSize: (Int, Int) = (0, 0)
    private var visionInputPool: CVPixelBufferPool?
    private var visionPoolSize: (Int, Int) = (0, 0)

    private let metricsLock = NSLock()
    private var _smoothedGPUMilliseconds: Double = 0.0
    public var smoothedGPUMilliseconds: Double {
        metricsLock.lock()
        defer { metricsLock.unlock() }
        return _smoothedGPUMilliseconds
    }

    public init(shaderSourceFallback: String) throws {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let queue = dev.makeCommandQueue() else {
            throw NSError(domain: "StudioCam.Metal", code: 1, userInfo: [NSLocalizedDescriptionKey: "Metal GPU unavailable"])
        }
        self.device = dev
        self.commandQueue = queue

        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, dev, nil, &cache)
        self.textureCache = cache!

        let lib: MTLLibrary
        if let metallibURL = Bundle.main.url(forResource: "default", withExtension: "metallib"),
           let precompiled = try? dev.makeLibrary(URL: metallibURL),
           precompiled.makeFunction(name: "fmlfeFineRefine") != nil {
            lib = precompiled
        } else {
            let opts = MTLCompileOptions()
            opts.fastMathEnabled = true
            lib = try dev.makeLibrary(source: shaderSourceFallback, options: opts)
        }

        func makePSO(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = lib.makeFunction(name: name) else {
                throw NSError(domain: "StudioCam.Metal", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing Metal kernel \(name)"])
            }
            return try dev.makeComputePipelineState(function: fn)
        }

        self.psoLumaCoarse = try makePSO("downsampleLumaCoarse")
        self.psoLKCoarseFlow = try makePSO("computeLKCoarseFlow")
        self.psoGuidedPrep = try makePSO("guidedPrepAndTemporalEMA")
        self.psoBoxH = try makePSO("boxFilterHorizontal")
        self.psoBoxV = try makePSO("boxFilterVerticalAndCoeffs")
        self.psoSmoothAB = try makePSO("smoothGuidedCoeffs")
        self.psoFMLFECoarseH = try makePSO("fmlfeCoarseHorizontal")
        self.psoFMLFECoarseV = try makePSO("fmlfeCoarseVertical")
        self.psoFMLFEFine = try makePSO("fmlfeFineRefine")
        self.psoDownsampleHalf = try makePSO("downsampleHalfBGRA")
        self.psoOpticalBokehDisc = try makePSO("opticalBokehDiscPass")
        self.psoComposite = try makePSO("compositeStudioFrame")
        self.psoScaleSink = try makePSO("scaleBGRAForSink")

        func makeGuideTex(_ fmt: MTLPixelFormat, w: Int = 512, h: Int = 288) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            return dev.makeTexture(descriptor: d)!
        }

        self.lumaCoarseA = makeGuideTex(.r16Float, w: 128, h: 72)
        self.lumaCoarseB = makeGuideTex(.r16Float, w: 128, h: 72)
        self.coarseFlowTex = makeGuideTex(.rg16Float, w: 128, h: 72)
        self.historyTexA = makeGuideTex(.rgba16Float)
        self.historyTexB = makeGuideTex(.rgba16Float)
        self.statsTex = makeGuideTex(.rgba16Float)
        self.boxTempTex = makeGuideTex(.rgba16Float)
        self.abRawTex = makeGuideTex(.rg16Float)
        self.abSmoothTex = makeGuideTex(.rg16Float)
        self.fmlfeTempTex = makeGuideTex(.rgba16Float, w: 128, h: 72)
        self.fmlfeCoarseTex = makeGuideTex(.rgba16Float, w: 128, h: 72)
        self.fmlfeFineTex = makeGuideTex(.rgba16Float)

        let d1 = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        d1.usage = [.shaderRead]
        self.dummy1x1Tex = dev.makeTexture(descriptor: d1)!
        var zero: UInt32 = 0xFF808080
        self.dummy1x1Tex.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &zero, bytesPerRow: 4)

        // Smooth architectural perspective fallback depth map (in case ANE model is still compiling on frame 1)
        let fbW = 64, fbH = 36
        let dfb = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: fbW, height: fbH, mipmapped: false)
        dfb.usage = [.shaderRead]
        let fbTex = dev.makeTexture(descriptor: dfb)!
        self.fallbackDepthTex = fbTex
        var fbPixels = [UInt16](repeating: 0, count: fbW * fbH)
        for y in 0..<fbH {
            let ny = Float(y) / Float(fbH - 1)
            for x in 0..<fbW {
                let nx = Float(x) / Float(fbW - 1)
                let vpDist = hypot((nx - 0.5) * 1.2, (ny - 0.42))
                let approxDepth = max(Float(0), min(Float(1), 1.0 - vpDist * 1.15))
                fbPixels[y * fbW + x] = Float16(approxDepth).bitPattern
            }
        }
        fbPixels.withUnsafeBytes { ptr in
            fbTex.replace(region: MTLRegionMake2D(0, 0, fbW, fbH), mipmapLevel: 0, withBytes: ptr.baseAddress!, bytesPerRow: fbW * 2)
        }

        let d3 = MTLTextureDescriptor()
        d3.textureType = .type3D
        d3.pixelFormat = .rgba16Float
        d3.width = 2; d3.height = 2; d3.depth = 2
        d3.usage = [.shaderRead]
        self.dummy3DLUTTex = dev.makeTexture(descriptor: d3)!
    }

    public func setActiveLUT(_ lut: Loaded3DLUT?) {
        lutLock.lock()
        activeLUT = lut
        lutLock.unlock()
    }

    public func resetTemporalHistory() {
        hasHistory = false
    }

    public func makeVisionInputBuffer(from cameraPixelBuffer: CVPixelBuffer, quality: SegmentationQualityPreset = .balancedANE) -> CVPixelBuffer {
        // Core ML ANE models (BGMv2 Clean-Plate & MODNet Webcam) run natively at 512x288 (16:9) —
        // GPU-downscaling directly into a 512x288 IOSurface BGRA buffer avoids CPU Vision/CoreML resize!
        let targetW: Int
        let targetH: Int
        switch quality {
        case .cleanPlateBGMv2, .modnetWebcamANE:
            targetW = 512; targetH = 288
        case .fastVideoCall:
            targetW = 640; targetH = 360
        case .balancedANE, .accurateMax:
            targetW = 960; targetH = 540
        }
        let w = CVPixelBufferGetWidth(cameraPixelBuffer)
        let h = CVPixelBufferGetHeight(cameraPixelBuffer)
        let fmt = CVPixelBufferGetPixelFormatType(cameraPixelBuffer)
        if w == targetW && h == targetH && fmt == kCVPixelFormatType_32BGRA {
            return cameraPixelBuffer
        }
        if visionInputPool == nil || visionPoolSize != (targetW, targetH) {
            visionInputPool = Self.createIOSurfacePool(width: targetW, height: targetH)
            visionPoolSize = (targetW, targetH)
        }
        guard let pool = visionInputPool else { return cameraPixelBuffer }
        var dstPBOpt: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &dstPBOpt)
        guard let dstPB = dstPBOpt,
              let dstTex = makeBGRAWriteTexture(from: dstPB) else {
            return cameraPixelBuffer
        }
        let (camP0, camP1, camIsYUV, camFullRange) = wrapPixelBuffer(cameraPixelBuffer)
        guard let p0 = camP0,
              let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            return cameraPixelBuffer
        }
        let p1 = camPlane0OrSecond(camP1)
        var params = SIMD3<UInt32>(camIsYUV ? 1 : 0, camFullRange ? 1 : 0, 0)
        enc.setComputePipelineState(psoDownsampleHalf)
        enc.setTexture(p0, index: 0)
        enc.setTexture(p1, index: 1)
        enc.setTexture(dstTex, index: 2)
        enc.setBytes(&params, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 0)
        enc.dispatchThreads(
            MTLSize(width: targetW, height: targetH, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return dstPB
    }

    public struct RenderResult {
        public let mainPixelBuffer: CVPixelBuffer
        public let mainTexture: MTLTexture
        public let sinkPixelBuffer: CVPixelBuffer?
    }

    public func renderFrame(
        outputDimensions: (width: Int, height: Int),
        cameraPixelBuffer: CVPixelBuffer,
        segmentationMaskBuffer: CVPixelBuffer?,
        backgroundVideoBuffer: CVPixelBuffer?,
        backgroundStaticTexture: MTLTexture?,
        sceneDepthTexture: MTLTexture?,
        faceRect: SIMD4<Float>,
        cameraCropRect: SIMD4<Float>,
        settings: RenderSettingsSnapshot,
        produceSink: Bool,
        sinkDimensions: (width: Int, height: Int) = (1920, 1080),
        waitForGPUCompletion: Bool = true
    ) -> RenderResult? {
        let outW = outputDimensions.width
        let outH = outputDimensions.height

        guard let outPB = allocateFromMainPool(width: outW, height: outH),
              let outTex = makeBGRAWriteTexture(from: outPB) else {
            return nil
        }

        let (camP0, camP1, camIsYUV, camFullRange) = wrapPixelBuffer(cameraPixelBuffer)
        guard let camPlane0 = camP0 else { return nil }
        let camPlane1 = camPlane0OrSecond(camP1)

        guard let cmd = commandQueue.makeCommandBuffer() else { return nil }

        // 1. Pyramidal LKT Optical Flow + Noise-Safe Guided-Filter Hair Matte + FMLFE Foreground Decontamination
        let needsMask = (settings.backgroundMode != .none) || settings.portraitEnabled || settings.studioLightEnabled || settings.edgeLightEnabled
        var currentGuideMaskTex: MTLTexture = pingHistoryToggle ? historyTexB : historyTexA
        if needsMask, let maskPB = segmentationMaskBuffer, let maskTex = makeSinglePlaneR8Texture(from: maskPB) {
            let prevHist = pingHistoryToggle ? historyTexB : historyTexA
            let nextHist = pingHistoryToggle ? historyTexA : historyTexB
            let prevLumaC = pingHistoryToggle ? lumaCoarseB : lumaCoarseA
            let currLumaC = pingHistoryToggle ? lumaCoarseA : lumaCoarseB
            currentGuideMaskTex = nextHist
            pingHistoryToggle.toggle()

            let tempBlend = Float(0.70 + 0.24 * max(0.0, min(1.0, settings.temporalStability)))
            var gu = GuidedUniformsGPU(
                gridSize: SIMD2(UInt32(guideWidth), UInt32(guideHeight)),
                radius: 3,
                eps: 0.016,
                temporalBlend: tempBlend,
                colorSimSigma: 0.036,
                hasHistory: hasHistory ? 1 : 0,
                camIsYUV: camIsYUV ? 1 : 0,
                camFullRange: camFullRange ? 1 : 0
            )
            let hadHist = hasHistory
            hasHistory = true

            if let enc = cmd.makeComputeCommandEncoder() {
                let tg = MTLSize(width: 16, height: 16, depth: 1)
                let coarseGrid = MTLSize(width: coarseWidth, height: coarseHeight, depth: 1)
                let grid = MTLSize(width: guideWidth, height: guideHeight, depth: 1)

                // Pass 0a: Coarse 128x72 Luma
                var camFlags = SIMD2<UInt32>(camIsYUV ? 1 : 0, camFullRange ? 1 : 0)
                enc.setComputePipelineState(psoLumaCoarse)
                enc.setTexture(camPlane0, index: 0)
                enc.setTexture(camPlane1, index: 1)
                enc.setTexture(currLumaC, index: 2)
                enc.setBytes(&camFlags, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
                enc.dispatchThreads(coarseGrid, threadsPerThreadgroup: tg)

                // Pass 0b: Coarse 5x5 Lucas-Kanade Optical Flow
                enc.setComputePipelineState(psoLKCoarseFlow)
                enc.setTexture(currLumaC, index: 0)
                enc.setTexture(hadHist ? prevLumaC : currLumaC, index: 1)
                enc.setTexture(coarseFlowTex, index: 2)
                enc.dispatchThreads(coarseGrid, threadsPerThreadgroup: tg)

                // Pass 1: LKT-Warped Temporal EMA + Guided Stats
                enc.setComputePipelineState(psoGuidedPrep)
                enc.setTexture(camPlane0, index: 0)
                enc.setTexture(camPlane1, index: 1)
                enc.setTexture(maskTex, index: 2)
                enc.setTexture(prevHist, index: 3)
                enc.setTexture(nextHist, index: 4)
                enc.setTexture(statsTex, index: 5)
                enc.setTexture(coarseFlowTex, index: 6)
                enc.setBytes(&gu, length: MemoryLayout<GuidedUniformsGPU>.stride, index: 0)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)

                // Pass 2a..2c: Separable O(1) Guided Filter
                enc.setComputePipelineState(psoBoxH)
                enc.setTexture(statsTex, index: 0)
                enc.setTexture(boxTempTex, index: 1)
                enc.setBytes(&gu, length: MemoryLayout<GuidedUniformsGPU>.stride, index: 0)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)

                enc.setComputePipelineState(psoBoxV)
                enc.setTexture(boxTempTex, index: 0)
                enc.setTexture(abRawTex, index: 1)
                enc.setBytes(&gu, length: MemoryLayout<GuidedUniformsGPU>.stride, index: 0)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)

                enc.setComputePipelineState(psoSmoothAB)
                enc.setTexture(abRawTex, index: 0)
                enc.setTexture(abSmoothTex, index: 1)
                enc.setBytes(&gu, length: MemoryLayout<GuidedUniformsGPU>.stride, index: 0)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)

                // Pass 2d..2f: Fast Multi-Level Foreground Color Estimation (FMLFE) for Hair/Edge Decontamination
                if settings.backgroundMode != .none && settings.spillSuppression > 0.01 {
                    enc.setComputePipelineState(psoFMLFECoarseH)
                    enc.setTexture(nextHist, index: 0)
                    enc.setTexture(fmlfeTempTex, index: 1)
                    enc.dispatchThreads(coarseGrid, threadsPerThreadgroup: tg)

                    enc.setComputePipelineState(psoFMLFECoarseV)
                    enc.setTexture(fmlfeTempTex, index: 0)
                    enc.setTexture(fmlfeCoarseTex, index: 1)
                    enc.dispatchThreads(coarseGrid, threadsPerThreadgroup: tg)

                    enc.setComputePipelineState(psoFMLFEFine)
                    enc.setTexture(nextHist, index: 0)
                    enc.setTexture(fmlfeCoarseTex, index: 1)
                    enc.setTexture(fmlfeFineTex, index: 2)
                    enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
                }

                enc.endEncoding()
            }
        }

        // 2. Prepare Background Layer Textures
        var bgPlane0: MTLTexture = dummy1x1Tex
        var bgPlane1: MTLTexture = dummy1x1Tex
        var bgIsYUV: Bool = false
        var bgW: UInt32 = UInt32(outW)
        var bgH: UInt32 = UInt32(outH)
        var effectiveBGMode = settings.backgroundMode

        if settings.backgroundMode == .video {
            if let vPB = backgroundVideoBuffer {
                let (vp0, vp1, vYUV, _) = wrapPixelBuffer(vPB)
                if let validVP0 = vp0 {
                    bgPlane0 = validVP0
                    bgPlane1 = camPlane0OrSecond(vp1)
                    bgIsYUV = vYUV
                    bgW = UInt32(CVPixelBufferGetWidth(vPB))
                    bgH = UInt32(CVPixelBufferGetHeight(vPB))
                } else {
                    effectiveBGMode = .gradient
                }
            } else {
                effectiveBGMode = .gradient
            }
        } else if settings.backgroundMode == .image {
            if let imgTex = backgroundStaticTexture {
                bgPlane0 = imgTex
                bgPlane1 = dummy1x1Tex
                bgIsYUV = false
                bgW = UInt32(imgTex.width)
                bgH = UInt32(imgTex.height)
            } else {
                effectiveBGMode = .gradient
            }
        }

        // 3. Multi-Stop Optical Bokeh + Light Wrap / Ambient Pyramid (960x540 Near Blur + 480x270 Golden-Angle Far Disc Bokeh)
        var nearOutTex: MTLTexture = dummy1x1Tex
        var farOutTex: MTLTexture = dummy1x1Tex
        let needsPyramid = (effectiveBGMode != .gradient) && (
            (settings.portraitEnabled && settings.portraitStrength > 0.01) ||
            (effectiveBGMode != .none && (settings.lightWrapStrength > 0.01 || settings.ambientMatchStrength > 0.01))
        )

        if needsPyramid {
            let nearSigma: Float = settings.depthBokehEnabled
                ? (2.5 + 5.5 * settings.portraitStrength)
                : (4.0 + 18.0 * settings.portraitStrength)
            ensureBokehPyramidTextures(sigma: nearSigma)

            if let hSrc = halfSrcTex, let nBlur = nearBlurTex, let fBokeh = farBokehTex, let mpsBlur = gaussianBlur {
                if let enc = cmd.makeComputeCommandEncoder() {
                    enc.setComputePipelineState(psoDownsampleHalf)
                    if effectiveBGMode == .none {
                        enc.setTexture(camPlane0, index: 0)
                        enc.setTexture(camPlane1, index: 1)
                        var p = SIMD3<UInt32>(camIsYUV ? 1 : 0, camFullRange ? 1 : 0, 0)
                        enc.setBytes(&p, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 0)
                    } else {
                        enc.setTexture(bgPlane0, index: 0)
                        enc.setTexture(bgPlane1, index: 1)
                        var p = SIMD3<UInt32>(bgIsYUV ? 1 : 0, 0, 0)
                        enc.setBytes(&p, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 0)
                    }
                    enc.setTexture(hSrc, index: 2)
                    enc.dispatchThreads(MTLSize(width: hSrc.width, height: hSrc.height, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                    enc.endEncoding()
                }
                mpsBlur.encode(commandBuffer: cmd, sourceTexture: hSrc, destinationTexture: nBlur)
                nearOutTex = nBlur

                if (settings.portraitEnabled && settings.depthBokehEnabled) || settings.ambientMatchStrength > 0.01 {
                    if let enc = cmd.makeComputeCommandEncoder() {
                        var rad = settings.portraitStrength
                        enc.setComputePipelineState(psoOpticalBokehDisc)
                        enc.setTexture(nBlur, index: 0)
                        enc.setTexture(fBokeh, index: 1)
                        enc.setBytes(&rad, length: MemoryLayout<Float>.stride, index: 0)
                        enc.dispatchThreads(MTLSize(width: fBokeh.width, height: fBokeh.height, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                        enc.endEncoding()
                    }
                    farOutTex = fBokeh
                } else {
                    farOutTex = nBlur
                }
            }
        }

        // 4. Reaction progress
        let now = CACurrentMediaTime()
        var rType = settings.activeReactionType
        var rProg: Float = 0
        if rType != 0 {
            let elapsed = now - settings.activeReactionStart
            if elapsed >= 0 && elapsed < 2.2 {
                rProg = Float(elapsed / 2.2)
            } else {
                rType = 0
            }
        }

        // 5. Fused Single-Pass Composite (at exact client-requested resolution!)
        lutLock.lock()
        let currentLUT = activeLUT
        lutLock.unlock()

        let useLUT = settings.lutEnabled && (currentLUT != nil)
        let lutTex = currentLUT?.texture ?? dummy3DLUTTex
        let depthTex = sceneDepthTexture ?? fallbackDepthTex
        let camGammaPower = Self.detectCameraDisplayGammaMatch(cameraPixelBuffer, isYUV: camIsYUV)

        var cu = CompositeUniformsGPU(
            outputSize: SIMD2(UInt32(outW), UInt32(outH)),
            camSize: SIMD2(UInt32(CVPixelBufferGetWidth(cameraPixelBuffer)), UInt32(CVPixelBufferGetHeight(cameraPixelBuffer))),
            bgSize: SIMD2(bgW, bgH),
            camIsYUV: camIsYUV ? 1 : 0,
            camFullRange: camFullRange ? 1 : 0,
            bgMode: UInt32(effectiveBGMode.rawValue),
            bgIsYUV: bgIsYUV ? 1 : 0,
            portraitEnabled: settings.portraitEnabled ? 1 : 0,
            portraitStrength: settings.portraitStrength,
            depthBokehEnabled: settings.depthBokehEnabled ? 1 : 0,
            focalPlaneOffset: settings.focalPlaneOffset,
            studioLightEnabled: settings.studioLightEnabled ? 1 : 0,
            studioLightStrength: settings.studioLightStrength,
            edgeLightEnabled: settings.edgeLightEnabled ? 1 : 0,
            edgeLightStrength: settings.edgeLightStrength,
            edgeLightWarmth: settings.edgeLightWarmth,
            spillSuppression: settings.spillSuppression,
            edgeSoftness: settings.edgeSoftness,
            mirrorHorizontal: settings.mirrorPreview ? 1 : 0,
            lightWrapStrength: settings.lightWrapStrength,
            ambientMatchStrength: settings.ambientMatchStrength,
            skinPolishStrength: settings.skinPolishStrength,
            filmGrainStrength: settings.filmGrainStrength,
            reactionType: rType,
            reactionProgress: rProg,
            timeSeconds: Float(fmod(now, 1000.0)),
            lutStrength: settings.lutStrength,
            lutEnabled: useLUT ? 1 : 0,
            lutDimension: currentLUT?.dimension ?? 0,
            gradTopColor: settings.gradientPreset.topRGB,
            gradBottomColor: settings.gradientPreset.bottomRGB,
            faceRect: faceRect,
            camCropRect: cameraCropRect,
            lutDomainMin: currentLUT?.domainMin ?? SIMD4<Float>(0, 0, 0, 0),
            lutDomainMax: currentLUT?.domainMax ?? SIMD4<Float>(1, 1, 1, 1),
            matteTuning: SIMD4<Float>(settings.matteChoke, settings.temporalStability, camGammaPower, 0)
        )

        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(psoComposite)
            enc.setTexture(camPlane0, index: 0)
            enc.setTexture(camPlane1, index: 1)
            enc.setTexture(abSmoothTex, index: 2)
            enc.setTexture(bgPlane0, index: 3)
            enc.setTexture(bgPlane1, index: 4)
            enc.setTexture(nearOutTex, index: 5)
            enc.setTexture(outTex, index: 6)
            enc.setTexture(lutTex, index: 7)
            enc.setTexture(farOutTex, index: 8)
            enc.setTexture(depthTex, index: 9)
            enc.setTexture(currentGuideMaskTex, index: 10)
            enc.setTexture(fmlfeFineTex, index: 11)
            enc.setBytes(&cu, length: MemoryLayout<CompositeUniformsGPU>.stride, index: 0)
            enc.dispatchThreads(
                MTLSize(width: outW, height: outH, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
            )
            enc.endEncoding()
        }

        // 6. Virtual Camera Sink Output:
        // In Auto (Match App) or 1080p mode, outW==sW && outH==sH -> 100% ZERO-COPY (`sinkPBOut = outPB`)!
        var sinkPBOut: CVPixelBuffer? = nil
        if produceSink {
            let sW = max(2, sinkDimensions.width)
            let sH = max(2, sinkDimensions.height)
            if outW == sW && outH == sH {
                sinkPBOut = outPB
            } else if let sPB = allocateFromSinkPool(width: sW, height: sH), let sTex = makeBGRAWriteTexture(from: sPB) {
                if let enc = cmd.makeComputeCommandEncoder() {
                    enc.setComputePipelineState(psoScaleSink)
                    enc.setTexture(outTex, index: 0)
                    enc.setTexture(sTex, index: 1)
                    enc.dispatchThreads(
                        MTLSize(width: sW, height: sH, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
                    )
                    enc.endEncoding()
                }
                sinkPBOut = sPB
            }
        }

        cmd.addCompletedHandler { [weak self] completedCmd in
            let gpuMs = (completedCmd.gpuEndTime - completedCmd.gpuStartTime) * 1000.0
            guard gpuMs > 0, let self = self else { return }
            self.metricsLock.lock()
            if self._smoothedGPUMilliseconds == 0 {
                self._smoothedGPUMilliseconds = gpuMs
            } else {
                self._smoothedGPUMilliseconds = self._smoothedGPUMilliseconds * 0.88 + gpuMs * 0.12
            }
            self.metricsLock.unlock()
        }

        cmd.commit()
        if waitForGPUCompletion {
            // Guarantees GPU has finished writing every IOSurface pixel before CALayer preview & Teams/OBS read it (~0.8-1.5ms)
            cmd.waitUntilCompleted()
        } else {
            cmd.waitUntilScheduled()
        }

        return RenderResult(mainPixelBuffer: outPB, mainTexture: outTex, sinkPixelBuffer: sinkPBOut)
    }

    // MARK: - Helpers

    private func camPlane0OrSecond(_ tex: MTLTexture?) -> MTLTexture {
        tex ?? dummy1x1Tex
    }

    private func wrapPixelBuffer(_ pb: CVPixelBuffer) -> (MTLTexture?, MTLTexture?, Bool, Bool) {
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        if fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
            let yW = CVPixelBufferGetWidthOfPlane(pb, 0)
            let yH = CVPixelBufferGetHeightOfPlane(pb, 0)
            let uvW = CVPixelBufferGetWidthOfPlane(pb, 1)
            let uvH = CVPixelBufferGetHeightOfPlane(pb, 1)

            var yRef: CVMetalTexture?
            var uvRef: CVMetalTexture?
            CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, pb, nil, .r8Unorm, yW, yH, 0, &yRef)
            CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, pb, nil, .rg8Unorm, uvW, uvH, 1, &uvRef)
            return (yRef.flatMap { CVMetalTextureGetTexture($0) }, uvRef.flatMap { CVMetalTextureGetTexture($0) }, true, fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
        } else {
            let w = CVPixelBufferGetWidth(pb)
            let h = CVPixelBufferGetHeight(pb)
            var ref: CVMetalTexture?
            CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, pb, nil, .bgra8Unorm, w, h, 0, &ref)
            return (ref.flatMap { CVMetalTextureGetTexture($0) }, nil, false, false)
        }
    }

    private func makeSinglePlaneR8Texture(from pb: CVPixelBuffer) -> MTLTexture? {
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        var ref: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, pb, nil, .r8Unorm, w, h, 0, &ref)
        return ref.flatMap { CVMetalTextureGetTexture($0) }
    }

    private func makeBGRAWriteTexture(from pb: CVPixelBuffer) -> MTLTexture? {
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        var ref: CVMetalTexture?
        let attrs: [CFString: Any] = [
            kCVMetalTextureUsage: MTLTextureUsage.shaderRead.rawValue | MTLTextureUsage.shaderWrite.rawValue | MTLTextureUsage.renderTarget.rawValue
        ]
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, pb, attrs as CFDictionary, .bgra8Unorm, w, h, 0, &ref)
        return ref.flatMap { CVMetalTextureGetTexture($0) }
    }

    private func ensureBokehPyramidTextures(sigma: Float) {
        if halfSrcTex == nil {
            let dHalf = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 960, height: 540, mipmapped: false)
            dHalf.usage = [.shaderRead, .shaderWrite]
            dHalf.storageMode = .private
            halfSrcTex = device.makeTexture(descriptor: dHalf)
            nearBlurTex = device.makeTexture(descriptor: dHalf)

            let dQuarter = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 480, height: 270, mipmapped: false)
            dQuarter.usage = [.shaderRead, .shaderWrite]
            dQuarter.storageMode = .private
            farBokehTex = device.makeTexture(descriptor: dQuarter)
        }
        if abs(cachedBlurSigma - sigma) > 0.35 {
            cachedBlurSigma = sigma
            let blur = MPSImageGaussianBlur(device: device, sigma: sigma)
            blur.edgeMode = .clamp
            gaussianBlur = blur
        }
    }

    private static func detectCameraDisplayGammaMatch(_ pb: CVPixelBuffer, isYUV: Bool) -> Float {
        guard isYUV else { return 1.0 }
        // macOS AVFoundation UVC/built-in cameras tag 420v frames with SMPTE_240M_1995 or ITU_R_709_2
        // (~1.96 Apple QuickTime camera gamma). Native macOS display ColorSync converts 1.96 -> 2.2 sRGB
        // (pow(x, 1.12)). Applying the exact same transform keeps virtual-cam exposure & skin richness identical.
        if let tf = CVBufferCopyAttachment(pb, kCVImageBufferTransferFunctionKey, nil) {
            let s = "\(tf)"
            if s.contains("sRGB") || s.contains("IEC_sRGB") {
                return 1.0
            }
            if s.contains("240M") || s.contains("709") {
                return 1.12
            }
        }
        return 1.12
    }

    private static func attachSRGBColorTags(_ pb: CVPixelBuffer) {
        CVBufferSetAttachment(pb, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(pb, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_sRGB, .shouldPropagate)
        CVBufferSetAttachment(pb, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
    }

    private func allocateFromMainPool(width: Int, height: Int) -> CVPixelBuffer? {
        if mainPoolSize != (width, height) || mainOutputPool == nil {
            mainOutputPool = Self.createIOSurfacePool(width: width, height: height)
            mainPoolSize = (width, height)
        }
        guard let pool = mainOutputPool else { return nil }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb)
        if let validPB = pb { Self.attachSRGBColorTags(validPB) }
        return pb
    }

    private func allocateFromSinkPool(width: Int, height: Int) -> CVPixelBuffer? {
        if sinkPoolSize != (width, height) || sinkOutputPool == nil {
            sinkOutputPool = Self.createIOSurfacePool(width: width, height: height)
            sinkPoolSize = (width, height)
        }
        guard let pool = sinkOutputPool else { return nil }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb)
        if let validPB = pb { Self.attachSRGBColorTags(validPB) }
        return pb
    }

    private static func createIOSurfacePool(width: Int, height: Int) -> CVPixelBufferPool? {
        let poolAttrs: [CFString: Any] = [kCVPixelBufferPoolMinimumBufferCountKey: 4]
        let pbAttrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferMetalCompatibilityKey: true
        ]
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary, pbAttrs as CFDictionary, &pool)
        return pool
    }
}
