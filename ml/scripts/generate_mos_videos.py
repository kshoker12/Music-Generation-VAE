#!/usr/bin/env python3
"""Compile blind MOS evaluation MP4 videos from selected MIDI samples.

Audio engines:
  garageband (default for --out-dir videos/garageband) — macOS Apple DLS grand piano
    at written MIDI pitch + light hall reverb (ml/scripts/render_midi_apple.swift).
  dashboard — Tone.js Salamander engine used by the web dashboard
    (web/scripts/render_midi_dashboard.mjs).

Dependencies:
  pip install moviepy Pillow
  garageband engine: macOS + swift (built-in)
  dashboard engine: cd web && npm install

Usage (from repo root):
  python ml/scripts/generate_mos_videos.py --audio-engine garageband --out-dir artifacts/mos_evaluation/videos/garageband
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFont

Category = Literal["lullaby", "virtuoso", "crescendo"]
AudioEngine = Literal["garageband", "dashboard"]

ALL_CATEGORIES: list[Category] = ["lullaby", "virtuoso", "crescendo"]

# Subjective top-2 sample indices per model and category.
SELECTED_SAMPLES: dict[str, dict[str, list[int]]] = {
    "plain_transformer": {
        "lullaby": [10, 6],
        "virtuoso": [6, 8],
        "crescendo": [6, 9],
    },
    "simple_vae": {
        "lullaby": [2, 8],
        "virtuoso": [1, 9],
        "crescendo": [6, 4],
    },
    "hierarchical_vae": {
        "lullaby": [2, 10],
        "virtuoso": [4, 9],
        "crescendo": [7, 3],
    },
}

# Blind letter → model prefix (shuffled per category).
BLIND_MAPPING: dict[str, dict[str, str]] = {
    "lullaby": {"A": "plain_transformer", "B": "simple_vae", "C": "hierarchical_vae"},
    "virtuoso": {"A": "simple_vae", "B": "hierarchical_vae", "C": "plain_transformer"},
    "crescendo": {"A": "hierarchical_vae", "B": "plain_transformer", "C": "simple_vae"},
}

VIDEO_W, VIDEO_H = 1280, 720
BG_COLOR = (0x1E, 0x1E, 0x1E)
TEXT_COLOR = (0xF5, 0xF5, 0xF5)
FPS = 24

log = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _midi_path(mos_dir: Path, category: str, model_prefix: str, sample_idx: int) -> Path:
    return mos_dir / category / f"{model_prefix}_sample_{sample_idx:02d}.mid"


def _find_ffmpeg() -> str | None:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _dashboard_renderer_script(root: Path) -> Path:
    return root / "web/scripts/render_midi_dashboard.mjs"


def _apple_renderer_script(root: Path) -> Path:
    return root / "ml/scripts/render_midi_apple.swift"


def _preflight(*, root: Path, audio_engine: AudioEngine) -> None:
    missing: list[str] = []
    if _find_ffmpeg() is None:
        missing.append("ffmpeg (brew install ffmpeg, or pip install moviepy with imageio-ffmpeg)")
    if audio_engine == "garageband":
        if platform.system() != "Darwin":
            missing.append("garageband audio engine requires macOS")
        if shutil.which("swift") is None:
            missing.append("swift (Xcode Command Line Tools)")
        renderer = _apple_renderer_script(root)
        if not renderer.is_file():
            missing.append(f"Apple MIDI renderer not found: {renderer}")
        dls = Path(
            "/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls"
        )
        if not dls.is_file():
            missing.append(f"Apple DLS sound bank not found: {dls}")
    else:
        if shutil.which("node") is None:
            missing.append("node (https://nodejs.org/)")
        renderer = _dashboard_renderer_script(root)
        if not renderer.is_file():
            missing.append(f"Dashboard renderer not found: {renderer}")
        puppeteer = root / "web/node_modules/puppeteer"
        if not puppeteer.is_dir():
            missing.append("Run `cd web && npm install` to install puppeteer for dashboard audio")
    if missing:
        raise RuntimeError("Preflight failed:\n  - " + "\n  - ".join(missing))


def _render_one_wav(
    *,
    root: Path,
    midi: Path,
    wav: Path,
    audio_engine: AudioEngine,
) -> None:
    import subprocess

    wav.parent.mkdir(parents=True, exist_ok=True)
    if audio_engine == "garageband":
        result = subprocess.run(
            ["swift", str(_apple_renderer_script(root)), str(midi.resolve()), str(wav.resolve())],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        result = subprocess.run(
            [
                "node",
                str(_dashboard_renderer_script(root)),
                str(midi.resolve()),
                str(wav.resolve()),
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(root / "web"),
        )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"MIDI render failed for {midi.name} ({audio_engine}):\n{err}")
    if not wav.is_file() or wav.stat().st_size == 0:
        raise RuntimeError(f"Renderer produced no WAV: {wav}")
    log.info("Rendered WAV (%s) %s", audio_engine, wav)


def _render_wavs_for_category(
    *,
    root: Path,
    mos_dir: Path,
    category: Category,
    cache_dir: Path,
    skip_cache: bool,
    audio_engine: AudioEngine,
) -> None:
    mapping = BLIND_MAPPING[category]

    for letter in ("A", "B", "C"):
        model_prefix = mapping[letter]
        for sample_idx in SELECTED_SAMPLES[model_prefix][category]:
            midi = _midi_path(mos_dir, category, model_prefix, sample_idx)
            if not midi.is_file():
                raise FileNotFoundError(f"Missing MIDI: {midi}")
            wav = cache_dir / "wav" / category / f"{model_prefix}_sample_{sample_idx:02d}.wav"
            if wav.exists() and not skip_cache:
                continue
            _render_one_wav(root=root, midi=midi, wav=wav, audio_engine=audio_engine)


def _render_label_png(text: str, *, width: int, height: int, out_path: Path) -> None:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_size = 52
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), text, fill=TEXT_COLOR + (255,), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _midi_to_wav(
    midi_path: Path,
    wav_path: Path,
    *,
    root: Path,
    skip_cache: bool,
) -> Path:
    if not wav_path.is_file():
        raise FileNotFoundError(f"Expected pre-rendered WAV missing: {wav_path}")
    return wav_path


def _make_labeled_segment(
    *,
    label: str,
    wav_path: Path,
    cache_dir: Path,
):
    from moviepy import AudioFileClip, ColorClip, CompositeVideoClip, ImageClip

    audio = AudioFileClip(str(wav_path))
    duration = float(audio.duration)
    bg = ColorClip(size=(VIDEO_W, VIDEO_H), color=BG_COLOR, duration=duration)
    label_png = cache_dir / "labels" / f"{label.replace(' ', '_').replace('-', '_')}.png"
    _render_label_png(label, width=VIDEO_W, height=VIDEO_H, out_path=label_png)
    overlay = ImageClip(str(label_png), duration=duration).with_position("center")
    video = CompositeVideoClip([bg, overlay], size=(VIDEO_W, VIDEO_H))
    return video.with_audio(audio)


def _make_silence_segment(duration: float):
    import numpy as np
    from moviepy import AudioClip, ColorClip

    bg = ColorClip(size=(VIDEO_W, VIDEO_H), color=(0, 0, 0), duration=duration)
    silent = AudioClip(lambda t: np.zeros(2), duration=duration, fps=44100)
    return bg.with_audio(silent)


def _build_category_video(
    category: Category,
    *,
    root: Path,
    mos_dir: Path,
    out_path: Path,
    cache_dir: Path,
    transition_sec: float,
    skip_cache: bool,
    audio_engine: AudioEngine,
) -> dict[str, Any]:
    from moviepy import concatenate_videoclips

    mapping = BLIND_MAPPING[category]
    segments = []
    key_entries: list[dict[str, Any]] = []

    _render_wavs_for_category(
        root=root,
        mos_dir=mos_dir,
        category=category,
        cache_dir=cache_dir,
        skip_cache=skip_cache,
        audio_engine=audio_engine,
    )

    for letter in ("A", "B", "C"):
        model_prefix = mapping[letter]
        sample_indices = SELECTED_SAMPLES[model_prefix][category]
        for sample_num, sample_idx in enumerate(sample_indices, start=1):
            midi = _midi_path(mos_dir, category, model_prefix, sample_idx)
            if not midi.is_file():
                raise FileNotFoundError(f"Missing MIDI: {midi}")
            wav_rel = f"{category}/{model_prefix}_sample_{sample_idx:02d}.wav"
            wav = cache_dir / "wav" / wav_rel
            _midi_to_wav(midi, wav, root=root, skip_cache=skip_cache)
            label = f"Candidate {letter} - Sample {sample_num}"
            segments.append(
                _make_labeled_segment(label=label, wav_path=wav, cache_dir=cache_dir)
            )
            key_entries.append(
                {
                    "candidate": letter,
                    "sample_label": sample_num,
                    "model": model_prefix,
                    "sample_idx": sample_idx,
                    "midi_path": str(midi.relative_to(_repo_root())),
                    "on_screen_label": label,
                }
            )
        if letter != "C":
            segments.append(_make_silence_segment(transition_sec))

    final = concatenate_videoclips(segments, method="compose")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(
        str(out_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
        logger=None,
    )
    for seg in segments:
        seg.close()
    final.close()

    return {
        "category": category,
        "output_video": str(out_path.relative_to(_repo_root())),
        "blind_mapping": {k: v for k, v in mapping.items()},
        "clips": key_entries,
        "transition_sec": transition_sec,
        "audio_engine": audio_engine,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mos-dir", type=Path, default=None, help="MOS evaluation root (default: artifacts/mos_evaluation)")
    ap.add_argument("--out-dir", type=Path, default=None, help="MP4 output dir (default: artifacts/mos_evaluation/videos)")
    ap.add_argument("--transition-sec", type=float, default=1.5, help="Black gap between candidates (default: 1.5)")
    ap.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated subset: lullaby,virtuoso,crescendo",
    )
    ap.add_argument("--skip-cache", action="store_true", help="Force re-render all WAV files")
    ap.add_argument(
        "--audio-engine",
        choices=("garageband", "dashboard"),
        default="garageband",
        help="Audio renderer (default: garageband = macOS GM piano at written MIDI pitch)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = _repo_root()
    mos_dir = args.mos_dir or (root / "artifacts/mos_evaluation")
    if not mos_dir.is_absolute():
        mos_dir = root / mos_dir

    out_dir = args.out_dir or (mos_dir / "videos" / args.audio_engine)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    cache_dir = out_dir / "_cache"
    audio_engine: AudioEngine = args.audio_engine  # type: ignore[assignment]

    if args.categories:
        categories: list[Category] = [c.strip() for c in args.categories.split(",")]  # type: ignore[assignment]
        for c in categories:
            if c not in ALL_CATEGORIES:
                raise ValueError(f"Unknown category: {c!r}")
    else:
        categories = list(ALL_CATEGORIES)

    try:
        _preflight(root=root, audio_engine=audio_engine)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1

    blind_key: dict[str, Any] = {
        "description": "Researcher-only decode key for blind MOS videos. Do not share with judges.",
        "audio_engine": audio_engine,
        "categories": {},
    }

    for category in categories:
        out_mp4 = out_dir / f"{category}.mp4"
        log.info("Building %s -> %s", category, out_mp4)
        meta = _build_category_video(
            category,
            root=root,
            mos_dir=mos_dir,
            out_path=out_mp4,
            cache_dir=cache_dir,
            transition_sec=float(args.transition_sec),
            skip_cache=bool(args.skip_cache),
            audio_engine=audio_engine,
        )
        blind_key["categories"][category] = meta
        log.info("Wrote %s", out_mp4)

    key_path = out_dir / "blind_key.json"
    key_path.write_text(json.dumps(blind_key, indent=2), encoding="utf-8")
    log.info("Wrote blind decode key to %s", key_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
