#!/usr/bin/env node
/**
 * Render MIDI → WAV using the same playback path as web/src/App.tsx:
 *   ensurePianoReady()  — Salamander sampler + light reverb
 *   scheduleFromMidi()  — parse @tonejs/midi, transpose -7, Tone.Part
 *
 * Offline rendering requires Tone.getTransport().start(0) (live play() does this
 * via Tone.Transport.start()).
 *
 * Usage:
 *   node web/scripts/render_midi_dashboard.mjs in.mid out.wav [in2.mid out2.wav ...]
 */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import puppeteer from "puppeteer";

// Keep in sync with web/src/App.tsx
const SALAMANDER_URLS = {
  A0: "A0.mp3",
  C1: "C1.mp3",
  "D#1": "Ds1.mp3",
  "F#1": "Fs1.mp3",
  A1: "A1.mp3",
  C2: "C2.mp3",
  "D#2": "Ds2.mp3",
  "F#2": "Fs2.mp3",
  A2: "A2.mp3",
  C3: "C3.mp3",
  "D#3": "Ds3.mp3",
  "F#3": "Fs3.mp3",
  A3: "A3.mp3",
  C4: "C4.mp3",
  "D#4": "Ds4.mp3",
  "F#4": "Fs4.mp3",
  A4: "A4.mp3",
  C5: "C5.mp3",
  "D#5": "Ds5.mp3",
  "F#5": "Fs5.mp3",
  A5: "A5.mp3",
  C6: "C6.mp3",
  "D#6": "Ds6.mp3",
  "F#6": "Fs6.mp3",
  A6: "A6.mp3",
  C7: "C7.mp3",
  "D#7": "Ds7.mp3",
  "F#7": "Fs7.mp3",
  A7: "A7.mp3",
  C8: "C8.mp3",
};
const SALAMANDER_BASE_URL = "https://tonejs.github.io/audio/salamander/";
const TRANSPOSE_SEMITONES = -7;

const RENDER_PAGE = String.raw`<!DOCTYPE html>
<html><head><meta charset="utf-8"></head><body>
<script type="module">
import * as Tone from "https://esm.sh/tone@15.1.22";
import { Midi } from "https://esm.sh/@tonejs/midi@2.0.28";

const SALAMANDER_URLS = ${JSON.stringify(SALAMANDER_URLS)};
const SALAMANDER_BASE_URL = ${JSON.stringify(SALAMANDER_BASE_URL)};
const TRANSPOSE_SEMITONES = ${TRANSPOSE_SEMITONES};

function audioBufferToWav(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataLength = buffer.length * blockAlign;
  const headerLength = 44;
  const out = new Uint8Array(headerLength + dataLength);
  const view = new DataView(out.buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) out[off + i] = s.charCodeAt(i); };
  writeStr(0, "RIFF");
  view.setUint32(4, headerLength + dataLength - 8, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeStr(36, "data");
  view.setUint32(40, dataLength, true);
  const channels = [];
  for (let ch = 0; ch < numChannels; ch++) channels.push(buffer.getChannelData(ch));
  let offset = headerLength;
  for (let i = 0; i < buffer.length; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const sample = Math.max(-1, Math.min(1, channels[ch][i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return out;
}

function bufferPeak(buffer) {
  let peak = 0;
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i]));
  }
  return peak;
}

// Mirrors scheduleFromMidi() + ensurePianoReady() from App.tsx, rendered offline.
window.renderMidiBytes = async (bytes) => {
  const midi = new Midi(bytes);
  const bpm = midi.header.tempos?.[0]?.bpm;
  if (typeof bpm === "number" && Number.isFinite(bpm) && bpm > 0) {
    Tone.getTransport().bpm.value = bpm;
  }

  const events = [];
  for (const track of midi.tracks) {
    for (const n of track.notes) {
      events.push([n.time, { midi: n.midi, duration: n.duration, velocity: n.velocity }]);
    }
  }
  events.sort((a, b) => a[0] - b[0]);

  const duration = Math.max(1, (midi.duration ?? 16) + 3.0);

  const rendered = await Tone.Offline(async () => {
    const reverb = new Tone.Reverb({ decay: 2.2, preDelay: 0.02, wet: 0.18 }).toDestination();
    const sampler = new Tone.Sampler({
      urls: SALAMANDER_URLS,
      baseUrl: SALAMANDER_BASE_URL,
      release: 1.2,
    }).connect(reverb);
    await Tone.loaded();

    const part = new Tone.Part((time, value) => {
      const note = Tone.Frequency(value.midi + TRANSPOSE_SEMITONES, "midi").toNote();
      sampler.triggerAttackRelease(note, value.duration, time, value.velocity);
    }, events);
    part.start(0);
    // Required for offline rendering (live dashboard calls Tone.Transport.start() in play()).
    Tone.getTransport().start(0);
  }, duration);

  const buffer = rendered.get();
  const peak = bufferPeak(buffer);
  if (peak < 1e-4) {
    throw new Error("Rendered audio is silent (transport may not have started)");
  }

  const wav = audioBufferToWav(buffer);
  let binary = "";
  for (let i = 0; i < wav.length; i++) binary += String.fromCharCode(wav[i]);
  return btoa(binary);
};

window.__ready = true;
</script>
</body></html>`;

async function renderMany(pairs) {
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--autoplay-policy=no-user-gesture-required"],
  });
  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(300_000);
    await page.setContent(RENDER_PAGE, { waitUntil: "networkidle0", timeout: 120_000 });
    await page.waitForFunction("window.__ready === true", { timeout: 60_000 });

    for (const [midiPath, wavPath] of pairs) {
      const midiBytes = readFileSync(midiPath);
      const b64 = await page.evaluate(async (arr) => {
        const bytes = new Uint8Array(arr);
        return await window.renderMidiBytes(bytes);
      }, [...midiBytes]);
      writeFileSync(wavPath, Buffer.from(b64, "base64"));
      console.log("Wrote", wavPath);
    }
  } finally {
    await browser.close();
  }
}

const args = process.argv.slice(2);
if (args.length < 2 || args.length % 2 !== 0) {
  console.error("Usage: node web/scripts/render_midi_dashboard.mjs in.mid out.wav [in2.mid out2.wav ...]");
  process.exit(1);
}

const pairs = [];
for (let i = 0; i < args.length; i += 2) {
  pairs.push([resolve(args[i]), resolve(args[i + 1])]);
}

try {
  await renderMany(pairs);
} catch (err) {
  console.error(err);
  process.exit(1);
}
