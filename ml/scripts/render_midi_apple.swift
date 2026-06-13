#!/usr/bin/env swift
/**
 * Render MIDI → WAV using macOS built-in GM acoustic grand piano (Apple DLS bank).
 * Plays MIDI at written pitch (no dashboard transpose). Matches how macOS renders
 * standard MIDI piano tracks and is the closest offline match to opening a .mid in
 * GarageBand without the web Salamander pitch-shifting artifacts.
 *
 * Usage: swift ml/scripts/render_midi_apple.swift input.mid output.wav
 */

import AVFoundation
import Foundation

let args = CommandLine.arguments
guard args.count == 3 else {
    fputs("Usage: render_midi_apple.swift input.mid output.wav\n", stderr)
    exit(1)
}

let midiURL = URL(fileURLWithPath: args[1])
let outURL = URL(fileURLWithPath: args[2])

let dlsURL = URL(
    fileURLWithPath: "/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls"
)

let sampleRate: Double = 44100
let tailSeconds: Double = 2.5

guard FileManager.default.fileExists(atPath: dlsURL.path) else {
    fputs("Apple DLS sound bank not found at \(dlsURL.path)\n", stderr)
    exit(1)
}

let engine = AVAudioEngine()
let sampler = AVAudioUnitSampler()
let reverb = AVAudioUnitReverb()

engine.attach(sampler)
engine.attach(reverb)
reverb.loadFactoryPreset(.largeHall)
reverb.wetDryMix = 18 // light hall, similar to default GarageBand piano patch send

engine.connect(sampler, to: reverb, format: nil)
engine.connect(reverb, to: engine.mainMixerNode, format: nil)

do {
    try sampler.loadSoundBankInstrument(
        at: dlsURL,
        program: 0,
        bankMSB: UInt8(kAUSampler_DefaultMelodicBankMSB),
        bankLSB: UInt8(kAUSampler_DefaultBankLSB)
    )
} catch {
    fputs("Failed to load Apple sound bank: \(error)\n", stderr)
    exit(1)
}

let sequencer = AVAudioSequencer(audioEngine: engine)
do {
    try sequencer.load(from: midiURL, options: [])
} catch {
    fputs("Failed to load MIDI: \(error)\n", stderr)
    exit(1)
}

let sequenceLength = sequencer.tracks.map(\.lengthInSeconds).max() ?? 0
let totalSeconds = max(sequenceLength + tailSeconds, 1.0)
let totalFrames = AVAudioFrameCount(totalSeconds * sampleRate)

guard let format = AVAudioFormat(
    commonFormat: .pcmFormatFloat32,
    sampleRate: sampleRate,
    channels: 2,
    interleaved: false
) else {
    fputs("Failed to create audio format\n", stderr)
    exit(1)
}

let wavSettings: [String: Any] = [
    AVFormatIDKey: kAudioFormatLinearPCM,
    AVSampleRateKey: sampleRate,
    AVNumberOfChannelsKey: 2,
    AVLinearPCMBitDepthKey: 16,
    AVLinearPCMIsFloatKey: false,
    AVLinearPCMIsBigEndianKey: false,
    AVLinearPCMIsNonInterleaved: false,
]

do {
    try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
    try engine.start()

    sequencer.currentPositionInSeconds = 0
    sequencer.prepareToPlay()
    try sequencer.start()

    let outputFile = try AVAudioFile(forWriting: outURL, settings: wavSettings)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4096) else {
        fputs("Failed to create render buffer\n", stderr)
        exit(1)
    }

    var renderedFrames: AVAudioFrameCount = 0
    while renderedFrames < totalFrames {
        let framesLeft = totalFrames - renderedFrames
        let block = min(framesLeft, engine.manualRenderingMaximumFrameCount)
        buffer.frameLength = block

        let status = try engine.renderOffline(block, to: buffer)
        switch status {
        case .success:
            try outputFile.write(from: buffer)
            renderedFrames += buffer.frameLength
        case .cannotDoInCurrentContext:
            continue
        case .insufficientDataFromInputNode:
            break
        case .error:
            fputs("Offline render error\n", stderr)
            exit(1)
        @unknown default:
            fputs("Unknown render status\n", stderr)
            exit(1)
        }
    }

    sequencer.stop()
    engine.stop()

    let seconds = Double(renderedFrames) / sampleRate
    print("Wrote \(outURL.path) (\(String(format: "%.2f", seconds))s)")
} catch {
    fputs("Render failed: \(error)\n", stderr)
    exit(1)
}
