import Foundation
import CoreMediaIO
import CoreMedia
import CoreVideo

/// Pushes zero-copy IOSurface-backed CVPixelBuffers into an already-installed, Apple-notarized
/// CoreMediaIO Virtual Camera Sink Stream (such as OBS Virtual Camera or Elgato Virtual Camera),
/// enabling system-wide virtual webcam output in Zoom, Teams, Meet, and browsers WITHOUT requiring
/// a paid Apple Developer System Extension certificate.
public final class CMIOSinkBridge {
    public struct SinkTarget: Identifiable, Equatable {
        public let deviceID: CMIODeviceID
        public let streamID: CMIOStreamID
        public let name: String
        public let width: Int
        public let height: Int
        public var id: String { "\(deviceID):\(streamID)" }
    }

    private let lock = NSLock()
    private var activeTarget: SinkTarget?
    private var formatListenerStreamID: CMIOStreamID = 0
    private var formatListenerBlock: CMIOObjectPropertyListenerBlock?
    public var onActiveFormatChanged: (((Int, Int)) -> Void)?

    public var activeDimensions: (width: Int, height: Int) {
        lock.lock()
        let streamID = activeTarget?.streamID
        let fallback = (activeTarget?.width ?? 1920, activeTarget?.height ?? 1080)
        lock.unlock()
        if let st = streamID {
            let live = Self.queryActiveOrPreferredStreamDimensions(streamID: st)
            if live.0 > 0 && live.1 > 0 { return live }
        }
        return fallback
    }
    private var bufferQueue: CMSimpleQueue?
    private var formatDescription: CMVideoFormatDescription?
    private var sequenceNumber: UInt64 = 0
    private var isStarted: Bool = false

    public init() {}

    /// Discovers all installed CoreMediaIO Virtual Camera devices that expose an output Sink Stream (dir == 0).
    /// Prioritizes OBS Virtual Camera first so Camera Hub does not contend for Elgato Virtual Camera.
    public static func discoverSinkTargets() -> [SinkTarget] {
        var prop = CMIOObjectPropertyAddress(
            mSelector: CMIOObjectPropertySelector(kCMIOHardwarePropertyDevices),
            mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
            mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain)
        )
        var dataSize: UInt32 = 0
        guard CMIOObjectGetPropertyDataSize(CMIOObjectID(kCMIOObjectSystemObject), &prop, 0, nil, &dataSize) == noErr,
              dataSize > 0 else { return [] }

        let count = Int(dataSize) / MemoryLayout<CMIODeviceID>.size
        var devices = [CMIODeviceID](repeating: 0, count: count)
        guard CMIOObjectGetPropertyData(CMIOObjectID(kCMIOObjectSystemObject), &prop, 0, nil, dataSize, &dataSize, &devices) == noErr else {
            return []
        }

        var found: [SinkTarget] = []
        for dev in devices {
            var nameProp = CMIOObjectPropertyAddress(
                mSelector: CMIOObjectPropertySelector(kCMIOObjectPropertyName),
                mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
                mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain)
            )
            var cfName: CFString = "" as CFString
            var nameSize = UInt32(MemoryLayout<CFString>.size)
            _ = withUnsafeMutablePointer(to: &cfName) { ptr in
                CMIOObjectGetPropertyData(dev, &nameProp, 0, nil, nameSize, &nameSize, ptr)
            }
            let devName = cfName as String

            var sProp = CMIOObjectPropertyAddress(
                mSelector: CMIOObjectPropertySelector(kCMIODevicePropertyStreams),
                mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
                mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain)
            )
            var sSize: UInt32 = 0
            guard CMIOObjectGetPropertyDataSize(dev, &sProp, 0, nil, &sSize) == noErr, sSize > 0 else { continue }
            let sCount = Int(sSize) / MemoryLayout<CMIOStreamID>.size
            var streams = [CMIOStreamID](repeating: 0, count: sCount)
            guard CMIOObjectGetPropertyData(dev, &sProp, 0, nil, sSize, &sSize, &streams) == noErr else { continue }

            for st in streams {
                var dProp = CMIOObjectPropertyAddress(
                    mSelector: CMIOObjectPropertySelector(kCMIOStreamPropertyDirection),
                    mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
                    mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain)
                )
                var dir: UInt32 = 99
                var dSz = UInt32(MemoryLayout<UInt32>.size)
                if CMIOObjectGetPropertyData(st, &dProp, 0, nil, dSz, &dSz, &dir) == noErr, dir == 0 {
                    let (sw, sh) = queryActiveOrPreferredStreamDimensions(streamID: st)
                    found.append(SinkTarget(deviceID: dev, streamID: st, name: devName, width: sw, height: sh))
                }
            }
        }

        return found.sorted { a, b in
            let aOBS = a.name.localizedCaseInsensitiveContains("OBS")
            let bOBS = b.name.localizedCaseInsensitiveContains("OBS")
            if aOBS != bOBS { return aOBS }
            return a.name < b.name
        }
    }

    /// Queries the stream's currently negotiated format (`kCMIOStreamPropertyFormatDescription`) first
    /// so we match the exact resolution requested by Teams / Meet / Zoom / OBS (e.g. 1080p or 720p)
    /// without initializing a wasted 4K capture/downsample pipeline.
    public static func queryActiveOrPreferredStreamDimensions(streamID: CMIOStreamID) -> (Int, Int) {
        var activeProp = CMIOObjectPropertyAddress(
            mSelector: CMIOObjectPropertySelector(kCMIOStreamPropertyFormatDescription),
            mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
            mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain)
        )
        var activeFmt: CMFormatDescription? = nil
        var aSz = UInt32(MemoryLayout<CMFormatDescription?>.size)
        let aStatus = withUnsafeMutablePointer(to: &activeFmt) { ptr in
            CMIOObjectGetPropertyData(streamID, &activeProp, 0, nil, aSz, &aSz, ptr)
        }
        if aStatus == noErr, let fmt = activeFmt {
            let d = CMVideoFormatDescriptionGetDimensions(fmt)
            if d.width > 0 && d.height > 0 {
                return (Int(d.width), Int(d.height))
            }
        }

        var fProp = CMIOObjectPropertyAddress(
            mSelector: CMIOObjectPropertySelector(kCMIOStreamPropertyFormatDescriptions),
            mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
            mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain)
        )
        var fmts: CFArray? = nil
        var fSz = UInt32(MemoryLayout<CFArray?>.size)
        _ = withUnsafeMutablePointer(to: &fmts) { ptr in
            CMIOObjectGetPropertyData(streamID, &fProp, 0, nil, fSz, &fSz, ptr)
        }
        if let arr = fmts as? [CMFormatDescription], let first = arr.first {
            let d = CMVideoFormatDescriptionGetDimensions(first)
            if d.width > 0 && d.height > 0 {
                return (Int(d.width), Int(d.height))
            }
        }
        return (1920, 1080)
    }

    public func start(preferredTarget: SinkTarget? = nil) -> (Bool, String) {
        lock.lock()
        defer { lock.unlock() }
        if isStarted, let t = activeTarget {
            return (true, t.name)
        }

        let targets = Self.discoverSinkTargets()
        guard let chosen = preferredTarget ?? targets.first else {
            return (false, "No CMIO Sink Extension found")
        }

        var unmanagedQueue: Unmanaged<CMSimpleQueue>? = nil
        let qStatus = CMIOStreamCopyBufferQueue(chosen.streamID, { _, _, _ in }, nil, &unmanagedQueue)
        guard qStatus == noErr, let queue = unmanagedQueue?.takeRetainedValue() else {
            return (false, "Queue error (\(qStatus)) on \(chosen.name)")
        }

        let startStatus = CMIODeviceStartStream(chosen.deviceID, chosen.streamID)
        guard startStatus == noErr else {
            return (false, "StartStream error (\(startStatus)) on \(chosen.name)")
        }

        self.activeTarget = chosen
        self.bufferQueue = queue
        self.formatDescription = nil
        self.sequenceNumber = 0
        self.isStarted = true
        installFormatChangeListener(streamID: chosen.streamID)
        let live = Self.queryActiveOrPreferredStreamDimensions(streamID: chosen.streamID)
        return (true, "\(chosen.name) (\(live.0)×\(live.1))")
    }

    public func stop() {
        lock.lock()
        removeFormatChangeListenerLocked()
        if isStarted, let t = activeTarget {
            _ = CMIODeviceStopStream(t.deviceID, t.streamID)
        }
        isStarted = false
        bufferQueue = nil
        activeTarget = nil
        formatDescription = nil
        lock.unlock()
    }

    private func installFormatChangeListener(streamID: CMIOStreamID) {
        removeFormatChangeListenerLocked()
        var addr = CMIOObjectPropertyAddress(
            mSelector: CMIOObjectPropertySelector(kCMIOStreamPropertyFormatDescription),
            mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
            mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain)
        )
        let block: CMIOObjectPropertyListenerBlock = { [weak self] _, _ in
            guard let self = self else { return }
            let dims = Self.queryActiveOrPreferredStreamDimensions(streamID: streamID)
            self.lock.lock()
            self.formatDescription = nil
            self.lock.unlock()
            self.onActiveFormatChanged?(dims)
        }
        CMIOObjectAddPropertyListenerBlock(streamID, &addr, DispatchQueue.main, block)
        formatListenerStreamID = streamID
        formatListenerBlock = block
    }

    private func removeFormatChangeListenerLocked() {
        if formatListenerStreamID != 0, let block = formatListenerBlock {
            var addr = CMIOObjectPropertyAddress(
                mSelector: CMIOObjectPropertySelector(kCMIOStreamPropertyFormatDescription),
                mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
                mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain)
            )
            CMIOObjectRemovePropertyListenerBlock(formatListenerStreamID, &addr, DispatchQueue.main, block)
        }
        formatListenerStreamID = 0
        formatListenerBlock = nil
    }

    public func enqueueFrame(_ pixelBuffer: CVPixelBuffer, fps: Int32 = 30) {
        lock.lock()
        defer { lock.unlock() }
        guard isStarted, let queue = bufferQueue else { return }

        // Drop gracefully if downstream consumer isn't draining fast enough
        let cap = CMSimpleQueueGetCapacity(queue)
        if cap > 0 && (Float(CMSimpleQueueGetCount(queue)) / Float(cap)) >= 0.85 {
            return
        }

        if formatDescription == nil {
            var desc: CMVideoFormatDescription?
            CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pixelBuffer, formatDescriptionOut: &desc)
            formatDescription = desc
        }
        guard let fmt = formatDescription else { return }

        let now = CMClockGetTime(CMClockGetHostTimeClock())
        var timing = CMSampleTimingInfo(
            duration: CMTimeMake(value: 1, timescale: fps),
            presentationTimeStamp: now,
            decodeTimeStamp: .invalid
        )

        var sampleBuffer: CMSampleBuffer?
        let status = CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: fmt,
            sampleTiming: &timing,
            sampleBufferOut: &sampleBuffer
        )
        if status == noErr, let sb = sampleBuffer {
            sequenceNumber &+= 1
            let retained = Unmanaged.passRetained(sb).toOpaque()
            if CMSimpleQueueEnqueue(queue, element: retained) != noErr {
                Unmanaged<CMSampleBuffer>.fromOpaque(retained).release()
            }
        }
    }
}
