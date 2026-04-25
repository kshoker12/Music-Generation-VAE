import { Midi } from "@tonejs/midi";
import { useEffect, useMemo, useRef, useState } from "react";
import * as Tone from "tone";

type ModelType = "plain" | "vae" | "simple_vae";
type PhaseMode = "one" | "two";
type RequestState = "idle" | "generating" | "ready" | "error";
type PlayerState = "stopped" | "playing" | "paused";

const ATTR_NAMES = ["polyphony", "intensity", "velocity", "density"] as const;

function GlassCard({
  title,
  subtitle,
  right,
  children,
}: {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/5 shadow-[0_20px_70px_-40px_rgba(0,0,0,0.7)] backdrop-blur-xl">
      <div className="absolute inset-0 opacity-60 [background:radial-gradient(700px_400px_at_20%_20%,rgba(255,255,255,0.08),transparent_55%)]" />
      <div className="relative p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-sm font-semibold tracking-wide text-white/90">{title}</div>
            {subtitle ? <div className="mt-1 text-xs text-white/60">{subtitle}</div> : null}
          </div>
          {right ? <div className="shrink-0">{right}</div> : null}
        </div>
        <div className="mt-4">{children}</div>
      </div>
    </div>
  );
}

function Spinner({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs text-white/70">
      <span className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-white/70" />
      <span>{label}</span>
    </div>
  );
}

function SliderRow({
  label,
  value,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  disabled?: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <div className="grid grid-cols-[1fr_auto] items-center gap-3">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-sm font-medium text-white/85">{label}</div>
          <div className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs font-semibold text-white/80">
            {value}
          </div>
        </div>
        <input
          className="w-full accent-indigo-400"
          type="range"
          min={0}
          max={7}
          step={1}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(parseInt(e.target.value, 10))}
        />
        <div className="flex justify-between text-[10px] text-white/35">
          {Array.from({ length: 8 }).map((_, i) => (
            <span key={i}>{i}</span>
          ))}
        </div>
      </div>
      <div className="hidden sm:block text-xs text-white/35">bin</div>
    </div>
  );
}

export default function App() {
  const [model, setModel] = useState<ModelType>("vae");
  const [phaseMode, setPhaseMode] = useState<PhaseMode>("two");
  const [seed, setSeed] = useState(123);
  const [temperature, setTemperature] = useState(1.0);
  const [topP, setTopP] = useState(0.9);

  const [reqState, setReqState] = useState<RequestState>("idle");
  const [status, setStatus] = useState<string>("Ready to generate.");

  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [midiArrayBuffer, setMidiArrayBuffer] = useState<ArrayBuffer | null>(null);
  const [playerState, setPlayerState] = useState<PlayerState>("stopped");

  const [phase1, setPhase1] = useState([3, 3, 3, 3]);
  const [phase2, setPhase2] = useState([3, 3, 3, 3]);

  const effectivePhase2 = phaseMode === "one" ? phase1 : phase2;

  const attributesPreview = useMemo(() => {
    const bars: number[][] = [];
    for (let b = 0; b < 8; b++) bars.push(b < 4 ? phase1 : effectivePhase2);
    return bars;
  }, [phase1, effectivePhase2]);

  const synthRef = useRef<Tone.PolySynth<Tone.Synth> | null>(null);
  const partRef = useRef<Tone.Part | null>(null);

  function stopPlayback() {
    Tone.Transport.stop();
    Tone.Transport.position = 0;
    partRef.current?.dispose();
    partRef.current = null;
    setPlayerState("stopped");
  }

  function scheduleFromMidi(buf: ArrayBuffer) {
    stopPlayback();

    const midi = new Midi(buf);
    if (!synthRef.current) {
      synthRef.current = new Tone.PolySynth(Tone.Synth, {
        volume: -10,
        oscillator: { type: "triangle" },
        envelope: { attack: 0.01, decay: 0.1, sustain: 0.6, release: 0.6 },
      }).toDestination();
    }

    const events: Array<[number, { note: string; duration: number; velocity: number }]> = [];
    for (const track of midi.tracks) {
      for (const n of track.notes) {
        events.push([n.time, { note: n.name, duration: n.duration, velocity: n.velocity }]);
      }
    }
    events.sort((a, b) => a[0] - b[0]);

    const part = new Tone.Part((time, value: { note: string; duration: number; velocity: number }) => {
      synthRef.current?.triggerAttackRelease(value.note, value.duration, time, value.velocity);
    }, events);
    part.start(0);
    partRef.current = part;

    Tone.Transport.seconds = 0;
  }

  async function play() {
    if (!midiArrayBuffer) return;
    await Tone.start();
    if (!partRef.current) scheduleFromMidi(midiArrayBuffer);
    Tone.Transport.start();
    setPlayerState("playing");
  }

  function pause() {
    Tone.Transport.pause();
    setPlayerState("paused");
  }

  useEffect(() => {
    return () => {
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      stopPlayback();
      synthRef.current?.dispose();
      synthRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const busy = reqState === "generating";

  return (
    <div className="min-h-full text-white">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between gap-6">
              <div>
                <h1 className="text-3xl font-semibold tracking-tight">MusicGen</h1>
                <p className="mt-1 text-sm text-white/60">
                  Generate a MIDI remotely and audition it instantly in your browser.
                </p>
              </div>
              <div className="hidden sm:flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs text-white/70 backdrop-blur">
                <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400" />
                Remote inference
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="lg:col-span-1">
              <GlassCard
                title="Generation"
                subtitle="Model + sampling controls"
                right={busy ? <Spinner label="Generating…" /> : null}
              >
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-white/60">Model</span>
                    <select
                      className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-white outline-none ring-0 backdrop-blur"
                      value={model}
                      disabled={busy}
                      onChange={(e) => setModel(e.target.value as ModelType)}
                    >
                      <option value="plain">plain</option>
                      <option value="vae">vae</option>
                      <option value="simple_vae">simple_vae</option>
                    </select>
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-white/60">Phases</span>
                    <select
                      className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-white outline-none ring-0 backdrop-blur"
                      value={phaseMode}
                      disabled={busy}
                      onChange={(e) => setPhaseMode(e.target.value as PhaseMode)}
                    >
                      <option value="one">1 phase</option>
                      <option value="two">2 phases</option>
                    </select>
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-white/60">Seed</span>
                    <input
                      className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-white outline-none backdrop-blur"
                      type="number"
                      value={seed}
                      disabled={busy}
                      onChange={(e) => setSeed(parseInt(e.target.value || "0", 10))}
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-white/60">Temperature</span>
                    <input
                      className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-white outline-none backdrop-blur"
                      type="number"
                      step="0.05"
                      value={temperature}
                      disabled={busy}
                      onChange={(e) => setTemperature(parseFloat(e.target.value || "1"))}
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-white/60">Top-p</span>
                    <input
                      className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-white outline-none backdrop-blur"
                      type="number"
                      step="0.05"
                      value={topP}
                      disabled={busy}
                      onChange={(e) => setTopP(parseFloat(e.target.value || "0.9"))}
                    />
                  </label>

                  <div className="flex items-end">
                    <button
                      type="button"
                      disabled={busy}
                      className={[
                        "w-full rounded-lg px-4 py-2 text-sm font-semibold shadow",
                        busy
                          ? "cursor-not-allowed bg-white/10 text-white/40"
                          : "bg-indigo-500/90 text-white hover:bg-indigo-500",
                      ].join(" ")}
                      onClick={async () => {
                        setReqState("generating");
                        setStatus("Calling API… this can take a bit.");

                        if (downloadUrl) URL.revokeObjectURL(downloadUrl);
                        setDownloadUrl(null);
                        setMidiArrayBuffer(null);
                        stopPlayback();

                        try {
                          const base = import.meta.env.VITE_API_BASE_URL as string | undefined;
                          const apiKey = import.meta.env.VITE_API_KEY as string | undefined;
                          if (!base) throw new Error("Missing VITE_API_BASE_URL (set at build time).");
                          if (!apiKey) throw new Error("Missing VITE_API_KEY (set at build time).");

                          const url = `${base.replace(/\/$/, "")}/generate`;
                          const controller = new AbortController();
                          const t = window.setTimeout(() => controller.abort(), 120_000);

                          let res: Response;
                          try {
                            res = await fetch(url, {
                              method: "POST",
                              headers: {
                                "Content-Type": "application/json",
                                "X-API-Key": apiKey,
                              },
                              body: JSON.stringify({
                                model_type: model,
                                attributes: attributesPreview,
                                seed,
                                temperature,
                                top_p: topP,
                                block_size: 1024,
                                bars_per_sample: 8,
                              }),
                              signal: controller.signal,
                            });
                          } finally {
                            window.clearTimeout(t);
                          }

                          if (!res.ok) {
                            const txt = await res.text().catch(() => "");
                            throw new Error(`API error ${res.status}: ${txt || res.statusText}`);
                          }

                          setStatus("Downloading MIDI…");
                          const blob = await res.blob();
                          const buf = await blob.arrayBuffer();
                          const dl = URL.createObjectURL(blob);

                          setMidiArrayBuffer(buf);
                          setDownloadUrl(dl);
                          setReqState("ready");
                          setStatus("Generated. You can play or download the MIDI.");
                        } catch (e) {
                          const msg =
                            e instanceof DOMException && e.name === "AbortError"
                              ? "Timed out waiting for the API response."
                              : e instanceof Error
                                ? e.message
                                : String(e);
                          setReqState("error");
                          setStatus(msg);
                        }
                      }}
                    >
                      {busy ? "Generating…" : "Generate"}
                    </button>
                  </div>
                </div>
              </GlassCard>
            </div>

            <div className="lg:col-span-2 grid grid-cols-1 gap-6">
              <GlassCard
                title="Attributes"
                subtitle="Drag sliders (0–7). Phase 2 controls bars 5–8 unless locked."
                right={
                  phaseMode === "one" ? (
                    <span className="rounded-full border border-white/10 bg-white/5 px-2 py-1 text-[11px] text-white/65">
                      Phase 2 locked
                    </span>
                  ) : null
                }
              >
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                  <div className="space-y-4">
                    <div className="text-xs font-semibold text-white/70">Phase 1 (bars 1–4)</div>
                    {ATTR_NAMES.map((name, j) => (
                      <SliderRow
                        key={`p1-${name}`}
                        label={name}
                        value={phase1[j]}
                        disabled={busy}
                        onChange={(v) => {
                          const next = [...phase1];
                          next[j] = v;
                          setPhase1(next);
                        }}
                      />
                    ))}
                  </div>

                  <div className="space-y-4">
                    <div className="text-xs font-semibold text-white/70">
                      Phase 2 (bars 5–8){phaseMode === "one" ? " (locked)" : ""}
                    </div>
                    {ATTR_NAMES.map((name, j) => (
                      <SliderRow
                        key={`p2-${name}`}
                        label={name}
                        value={effectivePhase2[j]}
                        disabled={busy || phaseMode === "one"}
                        onChange={(v) => {
                          if (phaseMode === "one") return;
                          const next = [...phase2];
                          next[j] = v;
                          setPhase2(next);
                        }}
                      />
                    ))}
                  </div>
                </div>
              </GlassCard>

              <GlassCard
                title="Output"
                subtitle="Play in-browser or download the .mid file"
                right={
                  reqState === "generating" ? (
                    <Spinner label="Working…" />
                  ) : downloadUrl ? (
                    <a
                      className="rounded-lg bg-emerald-500/90 px-3 py-2 text-xs font-semibold text-white hover:bg-emerald-500"
                      href={downloadUrl}
                      download={`generated_${model}_seed${seed}.mid`}
                    >
                      Download MIDI
                    </a>
                  ) : null
                }
              >
                <div className="flex flex-col gap-3">
                  <div className="flex items-center justify-between gap-4">
                    <div className="text-sm text-white/75">{status}</div>
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      disabled={!midiArrayBuffer || busy || playerState === "playing"}
                      className={[
                        "rounded-lg px-3 py-2 text-sm font-semibold",
                        !midiArrayBuffer || busy || playerState === "playing"
                          ? "cursor-not-allowed bg-white/10 text-white/35"
                          : "bg-white/10 text-white hover:bg-white/15",
                      ].join(" ")}
                      onClick={play}
                    >
                      Play
                    </button>
                    <button
                      type="button"
                      disabled={!midiArrayBuffer || busy || playerState !== "playing"}
                      className={[
                        "rounded-lg px-3 py-2 text-sm font-semibold",
                        !midiArrayBuffer || busy || playerState !== "playing"
                          ? "cursor-not-allowed bg-white/10 text-white/35"
                          : "bg-white/10 text-white hover:bg-white/15",
                      ].join(" ")}
                      onClick={pause}
                    >
                      Pause
                    </button>
                    <button
                      type="button"
                      disabled={!midiArrayBuffer || busy || playerState === "stopped"}
                      className={[
                        "rounded-lg px-3 py-2 text-sm font-semibold",
                        !midiArrayBuffer || busy || playerState === "stopped"
                          ? "cursor-not-allowed bg-white/10 text-white/35"
                          : "bg-white/10 text-white hover:bg-white/15",
                      ].join(" ")}
                      onClick={stopPlayback}
                    >
                      Stop
                    </button>

                    <div className="ml-auto text-xs text-white/45">
                      {midiArrayBuffer ? "Audio uses your device volume." : "Generate to enable playback."}
                    </div>
                  </div>

                  <details className="rounded-lg border border-white/10 bg-black/20 px-4 py-3">
                    <summary className="cursor-pointer text-xs font-semibold text-white/70">
                      Inspect request payload (attributes [8,4])
                    </summary>
                    <pre className="mt-3 overflow-auto rounded bg-black/40 p-3 text-xs text-white/80">
{JSON.stringify(attributesPreview, null, 2)}
                    </pre>
                  </details>
                </div>
              </GlassCard>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

