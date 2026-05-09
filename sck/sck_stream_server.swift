/// SCStream-based screenshot server.
/// Protocol (binary, over stdio):
///   stdin:  one line "screenshot\n" per request
///   stdout: 4-byte big-endian uint32 (PNG length) + PNG bytes
///   stderr: human-readable log lines

import AppKit
import Foundation
import ScreenCaptureKit
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import CoreMedia

// Keep stream & server alive at global scope — ARC would release them when the Task exits.
var gStream: SCStream?
var gServer: FrameServer?

final class FrameServer: NSObject, SCStreamOutput, SCStreamDelegate {
    private var latestFrame: CGImage?
    private let lock = NSLock()

    // MARK: - SCStreamOutput
    func stream(_ stream: SCStream,
                didOutputSampleBuffer buffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .screen,
              let imageBuffer = CMSampleBufferGetImageBuffer(buffer) else { return }
        let ciImage = CIImage(cvImageBuffer: imageBuffer)
        let ctx = CIContext()
        guard let cg = ctx.createCGImage(ciImage, from: ciImage.extent) else { return }
        lock.lock()
        let first = latestFrame == nil
        latestFrame = cg
        lock.unlock()
        if first { fputs("first frame \(cg.width)x\(cg.height)\n", stderr) }
    }

    // MARK: - SCStreamDelegate
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("Stream stopped: \(error)\n", stderr)
        exit(1)
    }

    // MARK: - Serve stdin via DispatchSource (non-blocking — never occupies fd 0)
    func startStdinSource() {
        var lineBuf = Data()
        let source = DispatchSource.makeReadSource(
            fileDescriptor: STDIN_FILENO, queue: DispatchQueue(label: "sck.stdin")
        )
        source.setEventHandler { [weak self] in
            guard let self else { return }
            let avail = source.data
            guard avail > 0 else { exit(0) }
            var chunk = Data(count: Int(avail))
            let n = chunk.withUnsafeMutableBytes {
                Foundation.read(STDIN_FILENO, $0.baseAddress!, Int(avail))
            }
            if n <= 0 { exit(0) }
            lineBuf.append(chunk.prefix(n))
            while let idx = lineBuf.firstIndex(of: UInt8(ascii: "\n")) {
                let lineData = lineBuf[lineBuf.startIndex ..< idx]
                let line = String(data: lineData, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespaces) ?? ""
                lineBuf = Data(lineBuf[lineBuf.index(after: idx)...])
                if line == "screenshot" { self.sendFrame() }
            }
        }
        source.setCancelHandler { exit(0) }
        source.resume()
    }

    func sendFrame() {
        lock.lock()
        let frame = latestFrame
        lock.unlock()
        guard let img = frame else { writeUInt32(0); return }
        let data = NSMutableData()
        if let dest = CGImageDestinationCreateWithData(
            data, UTType.png.identifier as CFString, 1, nil
        ) {
            CGImageDestinationAddImage(dest, img, nil)
            CGImageDestinationFinalize(dest)
        }
        writeUInt32(UInt32(data.length))
        FileHandle.standardOutput.write(data as Data)
    }

    private func writeUInt32(_ value: UInt32) {
        var big = value.bigEndian
        FileHandle.standardOutput.write(Data(bytes: &big, count: 4))
    }
}

// Initialize CGS session required by ScreenCaptureKit
_ = NSApplication.shared

Task {
    do {
        let content = try await SCShareableContent.current

        guard let display = content.displays.first(where: { $0.frame.origin == .zero })
                ?? content.displays.first else {
            fputs("No display found\n", stderr); exit(1)
        }

        // Find iPhone Mirroring main window (318×701 logical points)
        let win = content.windows.first(where: {
            $0.owningApplication?.applicationName.contains("iPhone") == true &&
            Int($0.frame.width) == 318 && Int($0.frame.height) == 701
        })
        if let win {
            fputs("Window: \(win.frame)\n", stderr)
        } else {
            fputs("iPhone window not found, capturing full display\n", stderr)
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        if let win {
            config.sourceRect = win.frame
        }
        config.width = 636   // 318 * 2 Retina
        config.height = 1402 // 701 * 2 Retina
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1) // 1fps — low CPU
        config.showsCursor = false

        let server = FrameServer()
        gServer = server   // keep alive
        let stream = SCStream(filter: filter, configuration: config, delegate: server)
        gStream = stream   // keep alive

        try stream.addStreamOutput(
            server, type: .screen,
            sampleHandlerQueue: DispatchQueue(label: "sck.frames", qos: .utility)
        )
        try await stream.startCapture()
        fputs("Stream started — ready\n", stderr)
        // Signal Python that the stream is up and accepting requests
        FileHandle.standardOutput.write("ready\n".data(using: .utf8)!)

        server.startStdinSource()

        // Background task: re-query iPhone window position every 500ms and update
        // sourceRect so the capture follows the window if it moves.
        Task {
            var lastRect = win?.frame ?? .zero
            while true {
                try await Task.sleep(nanoseconds: 500_000_000)
                guard let content = try? await SCShareableContent.current,
                      let w = content.windows.first(where: {
                          $0.owningApplication?.applicationName.contains("iPhone") == true &&
                          Int($0.frame.width) == 318 && Int($0.frame.height) == 701
                      }),
                      w.frame != lastRect else { continue }
                lastRect = w.frame
                fputs("Window moved to \(w.frame), updating sourceRect\n", stderr)
                let newConfig = SCStreamConfiguration()
                newConfig.sourceRect = w.frame
                newConfig.width = 636
                newConfig.height = 1402
                newConfig.minimumFrameInterval = CMTime(value: 1, timescale: 1)
                newConfig.showsCursor = false
                try? await stream.updateConfiguration(newConfig)
            }
        }

        // Keep the Task alive so local refs don't drop; globals also hold them.
        try await Task.sleep(nanoseconds: .max)
    } catch {
        fputs("Error: \(error)\n", stderr)
        exit(1)
    }
}

RunLoop.main.run()
