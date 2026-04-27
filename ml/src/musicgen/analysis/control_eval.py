"""
Controllability + z-effect evaluation for the hierarchical VAE (vae_v3).

Two experiments are run:

1. Attribute sweep (fixed z, varied attribute)
   For each of the 4 attributes, generate one MIDI per bin (0..7) with all 8 bars
   set to the same bin (uniform exaggeration). The other 3 attributes are held
   at a neutral middle bin. The latent z_p is fixed to the same draw across all
   sweeps so any change in the output is attributable to the attribute knob.

2. z sweep (fixed attributes, varied z)
   For 1 or more attribute configs, generate several MIDIs with the same
   attributes but different z draws. This isolates the effect of z on content
   while attributes should remain respected.

For every generated piece we re-derive the four raw per-bar descriptors
(polyphony rate, rhythmic intensity, velocity dynamics, note density) directly
from the generated REMI tokens using the same logic that produced the training
labels (see compute_raw_attributes_per_bar_from_tokens in
ml/scripts/preprocess_all.py), then bin them with the saved global quantile
thresholds in artifacts/attributes/quantiles.json.

Metrics (the "handful for the report") are emitted as a JSON report, a LaTeX
table, and three matplotlib figures.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


# Public attribute order. Keep aligned with preprocess_all.ATTRIBUTE_NAMES and
# the layout of `attributes` matrices fed into the model: index k -> attribute.
ATTRIBUTE_NAMES = ["polyphony_rate", "rhythmic_intensity", "velocity_dynamics", "note_density"]

# UI-friendly labels used in figure captions and the LaTeX table.
ATTRIBUTE_DISPLAY = {
    "polyphony_rate": "Polyphony rate",
    "rhythmic_intensity": "Rhythmic intensity",
    "velocity_dynamics": "Velocity dynamics",
    "note_density": "Note density",
}

# Block size and bars matches preprocess + ckpt.
BLOCK_SIZE = 1024
BARS_PER_SAMPLE = 8


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parents[i] for i in range(0, 10)]:
        if (p / "ml" / "src" / "musicgen").is_dir():
            return p
    return Path.cwd()


def _resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (_repo_root() / pp).resolve()


def _json_safe(obj: Any) -> Any:
    """Recursively replace NaN floats with None so json.dumps is strict-compliant."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# Raw per-bar attribute extraction (mirrors ml/scripts/preprocess_all.py)
# --------------------------------------------------------------------------- #


def _parse_duration_beats(token: str) -> float:
    raw = token.split("_", 1)[1]
    try:
        return float(raw)
    except Exception:
        parts = raw.split(".")
        if len(parts) == 3:
            a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
            if c == 0:
                raise ValueError(f"Cannot parse duration token: {token}")
            return float(a + (b / c))
        raise ValueError(f"Cannot parse duration token: {token}")


def compute_raw_attributes_per_bar_from_tokens(
    x_tokens_1024: list[int],
    *,
    id_to_token: dict[int, str],
    bar_token_id: int,
    bars_per_sample: int = BARS_PER_SAMPLE,
) -> list[list[float]]:
    """Recompute the 4 raw per-bar descriptors from a 1024-token REMI sequence.

    Returns a list of length `bars_per_sample`, each row of length 4 with
    [polyphony_rate, rhythmic_intensity, velocity_dynamics, note_density].
    """
    if len(x_tokens_1024) != BLOCK_SIZE:
        raise ValueError(f"Expected {BLOCK_SIZE} tokens, got {len(x_tokens_1024)}")

    onsets = [0 for _ in range(bars_per_sample)]
    pos_has_onset: list[set[int]] = [set() for _ in range(bars_per_sample)]
    velocities: list[list[int]] = [[] for _ in range(bars_per_sample)]
    global_intervals: list[tuple[float, float]] = []

    cur_bar = -1
    cur_pos = 0

    def parse_pos(tok: str) -> int:
        return int(tok.split("_", 1)[1])

    def parse_vel(tok: str) -> int:
        return int(tok.split("_", 1)[1])

    i = 0
    while i < len(x_tokens_1024):
        tid = x_tokens_1024[i]
        tok = id_to_token.get(int(tid))
        if tok is None:
            i += 1
            continue
        typ = tok.split("_", 1)[0]

        if tid == bar_token_id or typ == "Bar":
            cur_bar += 1
            cur_pos = 0
            i += 1
            continue

        if cur_bar < 0:
            i += 1
            continue
        if cur_bar >= bars_per_sample:
            break

        if typ == "Position":
            cur_pos = parse_pos(tok)
            i += 1
            continue

        if typ == "Pitch" and i + 2 < len(x_tokens_1024):
            vtok = id_to_token.get(int(x_tokens_1024[i + 1]), "")
            dtok = id_to_token.get(int(x_tokens_1024[i + 2]), "")
            if vtok.startswith("Velocity_") and dtok.startswith("Duration_"):
                vel = parse_vel(vtok)
                dur = _parse_duration_beats(dtok)

                onsets[cur_bar] += 1
                pos_has_onset[cur_bar].add(cur_pos)
                velocities[cur_bar].append(vel)

                s = cur_bar * 4.0 + (cur_pos * 0.25)
                e = s + dur
                global_intervals.append((s, e))

                i += 3
                continue

        i += 1

    attrs: list[list[float]] = []
    for b in range(bars_per_sample):
        note_density = float(onsets[b])
        rhythmic_intensity = float(len(pos_has_onset[b]) / 16.0)
        vel_std = float(np.std(np.asarray(velocities[b], dtype=float), ddof=0)) if velocities[b] else 0.0

        poly_samples: list[int] = []
        for p in range(16):
            t = b * 4.0 + (p * 0.25)
            active = 0
            for s, e in global_intervals:
                if s <= t < e:
                    active += 1
            poly_samples.append(active)
        polyphony_rate = float(sum(poly_samples) / 16.0)

        attrs.append([polyphony_rate, rhythmic_intensity, vel_std, note_density])
    return attrs


def bin_attributes(
    raw_attrs_barsx4: list[list[float]],
    thresholds: dict[str, list[float]],
    bars_per_sample: int = BARS_PER_SAMPLE,
) -> list[list[int]]:
    """Quantize raw per-bar attributes to integer bins 0..7 using global thresholds."""
    arr = np.asarray(raw_attrs_barsx4, dtype=float)
    if arr.ndim != 2 or arr.shape != (bars_per_sample, 4):
        raise ValueError(f"Expected shape [{bars_per_sample},4], got {arr.shape}")
    out = np.zeros_like(arr, dtype=int)
    for j, name in enumerate(ATTRIBUTE_NAMES):
        th = np.asarray(thresholds[name], dtype=float)
        if th.shape != (7,):
            raise ValueError(f"Expected 7 thresholds for {name}, got shape {th.shape}")
        out[:, j] = np.digitize(arr[:, j], th, right=False)
    return out.astype(int).tolist()


def load_thresholds(quantiles_path: str | Path | None = None) -> dict[str, list[float]]:
    """Load global quantile thresholds saved during preprocessing."""
    p = Path(quantiles_path) if quantiles_path else _resolve_path("artifacts/attributes/quantiles.json")
    payload = json.loads(p.read_text(encoding="utf-8"))
    th = payload["thresholds"]
    for name in ATTRIBUTE_NAMES:
        if name not in th:
            raise ValueError(f"Threshold for {name!r} missing in {p}")
    return th


# --------------------------------------------------------------------------- #
# Model loader
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _LoadedModel:
    model: torch.nn.Module
    tok_bundle: Any
    z_dim: int


def _device_from_str(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_checkpoint(path: str | Path) -> dict:
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict.")
    return ckpt


def load_vae_v3(
    *,
    ckpt_path: str | Path = "ml/src/musicgen/runs/vae_v3/ckpt.pt",
    device: str | torch.device = "auto",
    block_size: int = BLOCK_SIZE,
    bars_per_sample: int = BARS_PER_SAMPLE,
) -> _LoadedModel:
    """Load the hierarchical VAE checkpoint and the matching REMI tokenizer."""
    from musicgen.inference.tokenizer import load_remi_tokenizer
    from musicgen.models.vae import MusicVAE, VAEConfig

    dev = _device_from_str(device) if isinstance(device, str) else device
    resolved = _resolve_path(ckpt_path)
    if not resolved.exists():
        raise FileNotFoundError(f"VAE v3 checkpoint not found: {resolved}")

    ckpt = _load_checkpoint(resolved)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))

    cfg_dict = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    cfg = VAEConfig(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else VAEConfig()
    cfg = VAEConfig(
        **{**cfg.__dict__, "vocab_size": 195, "block_size": int(block_size), "bars_per_sample": int(bars_per_sample)}
    )
    model = MusicVAE(cfg)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(dev)

    tok_bundle = load_remi_tokenizer()

    return _LoadedModel(model=model, tok_bundle=tok_bundle, z_dim=int(cfg.z_dim))


# --------------------------------------------------------------------------- #
# Attribute matrix builders
# --------------------------------------------------------------------------- #


def make_uniform_attribute_matrix(
    swept_attr_idx: int,
    bin_value: int,
    *,
    neutral_bin: int = 4,
    bars_per_sample: int = BARS_PER_SAMPLE,
) -> list[list[int]]:
    """Return [bars,4] with the swept attribute set uniformly to `bin_value` across all bars."""
    if not (0 <= swept_attr_idx < 4):
        raise ValueError("swept_attr_idx must be in [0,4).")
    if not (0 <= bin_value <= 7):
        raise ValueError("bin_value must be in [0,7].")
    rows: list[list[int]] = []
    for _ in range(bars_per_sample):
        row = [int(neutral_bin)] * 4
        row[swept_attr_idx] = int(bin_value)
        rows.append(row)
    return rows


def make_constant_attribute_matrix(
    bins: list[int],
    *,
    bars_per_sample: int = BARS_PER_SAMPLE,
) -> list[list[int]]:
    """Return [bars,4] with the same per-attribute bins repeated for every bar."""
    if len(bins) != 4 or not all(0 <= int(b) <= 7 for b in bins):
        raise ValueError("bins must be a length-4 list of ints in [0,7].")
    return [[int(b) for b in bins] for _ in range(bars_per_sample)]


def make_phased_attribute_matrix(
    phase1_bins: list[int],
    phase2_bins: list[int],
    *,
    split: int = 4,
    bars_per_sample: int = BARS_PER_SAMPLE,
) -> list[list[int]]:
    """Return [bars,4] with `split` bars of phase1 then `bars-split` bars of phase2."""
    if len(phase1_bins) != 4 or len(phase2_bins) != 4:
        raise ValueError("Each phase must be length 4.")
    if not (0 <= split <= bars_per_sample):
        raise ValueError("split must be in [0, bars_per_sample].")
    rows = [[int(b) for b in phase1_bins] for _ in range(split)]
    rows += [[int(b) for b in phase2_bins] for _ in range(bars_per_sample - split)]
    return rows


# --------------------------------------------------------------------------- #
# Latent + generation helpers
# --------------------------------------------------------------------------- #


def sample_fixed_z(seed: int, *, z_dim: int, device: torch.device) -> torch.Tensor:
    """Deterministic standard-normal draw of shape [1, z_dim]."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    z = torch.randn((1, int(z_dim)), generator=g).to(device=device, dtype=torch.float32)
    return z


def generate_one(
    *,
    loaded: _LoadedModel,
    attributes: list[list[int]],
    z_p: torch.Tensor,
    seed: int,
    temperature: float = 0.95,
    top_p: float = 0.9,
    block_size: int = BLOCK_SIZE,
    bars_per_sample: int = BARS_PER_SAMPLE,
    device: Optional[torch.device] = None,
) -> list[int]:
    """Generate REMI tokens for the VAE with a fixed z_p."""
    from musicgen.inference.generate import GenerationConfig, generate_remi_tokens

    dev = device if device is not None else next(loaded.model.parameters()).device

    attrs = torch.tensor(attributes, dtype=torch.long, device=dev).unsqueeze(0)
    if attrs.shape != (1, bars_per_sample, 4):
        raise ValueError(f"attributes must be [{bars_per_sample},4], got shape {tuple(attrs.shape)[1:]}")

    z_p_in = z_p.to(device=dev, dtype=torch.float32)
    if z_p_in.ndim != 2 or z_p_in.shape[0] != 1:
        raise ValueError(f"z_p must be [1,z_dim], got {tuple(z_p_in.shape)}")

    cfg = GenerationConfig(
        block_size=int(block_size),
        bars_per_sample=int(bars_per_sample),
        temperature=float(temperature),
        top_p=float(top_p),
    )

    with torch.no_grad():
        tokens = generate_remi_tokens(
            model_type="vae",
            model=loaded.model,
            attributes=attrs,
            pad_id=loaded.tok_bundle.pad_id,
            bos_id=loaded.tok_bundle.bos_id,
            bar_id=loaded.tok_bundle.bar_id,
            id_to_token=loaded.tok_bundle.id_to_token,
            cfg=cfg,
            device=dev,
            z_p=z_p_in,
            seed=int(seed),
        )
    return tokens


def measure_attributes(
    tokens: list[int],
    *,
    id_to_token: dict[int, str],
    bar_id: int,
    thresholds: dict[str, list[float]],
    bars_per_sample: int = BARS_PER_SAMPLE,
) -> dict[str, list[list[float]] | list[list[int]]]:
    """Compute raw + binned per-bar attributes from generated tokens."""
    raw = compute_raw_attributes_per_bar_from_tokens(
        tokens, id_to_token=id_to_token, bar_token_id=int(bar_id), bars_per_sample=bars_per_sample
    )
    binned = bin_attributes(raw, thresholds, bars_per_sample=bars_per_sample)
    return {"raw": raw, "binned": binned}


# --------------------------------------------------------------------------- #
# Per-piece artifact saving
# --------------------------------------------------------------------------- #


def save_piece_artifacts(
    *,
    out_dir: Path,
    tokens: list[int],
    requested_attrs: list[list[int]],
    measured: dict[str, list[list[float]] | list[list[int]]],
    tok_bundle: Any,
    extra_meta: Optional[dict[str, Any]] = None,
) -> None:
    """Persist generated.mid, tokens.json, attributes.txt, measured.json."""
    from musicgen.inference.render_midi import tokens_to_midi, write_midi

    out_dir.mkdir(parents=True, exist_ok=True)

    midi_obj = tokens_to_midi(tokens, tok_bundle.tok)
    write_midi(midi_obj, out_dir / "generated.mid")

    (out_dir / "attributes.txt").write_text(json.dumps(requested_attrs, indent=2), encoding="utf-8")
    (out_dir / "tokens.json").write_text(json.dumps([int(t) for t in tokens]), encoding="utf-8")

    payload: dict[str, Any] = {
        "raw": [[float(v) for v in row] for row in measured["raw"]],
        "binned": [[int(v) for v in row] for row in measured["binned"]],
        "attribute_names": ATTRIBUTE_NAMES,
    }
    if extra_meta:
        payload["meta"] = extra_meta
    (out_dir / "measured.json").write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Statistics (Pearson, Spearman, Kendall) -- pure numpy, no scipy dependency
# --------------------------------------------------------------------------- #


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2 or y.size != x.size:
        return float("nan")
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of values in `a` (1-indexed)."""
    a = np.asarray(a, dtype=float).ravel()
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    # Handle ties via average rank.
    sorted_a = a[order]
    i = 0
    n = a.size
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = float(np.mean(ranks[order[i:j]]))
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.asarray(x).size < 2:
        return float("nan")
    return _safe_pearson(_rankdata(x), _rankdata(y))


def _safe_kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b (handles ties). O(n^2), fine for n up to a few thousand."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if n < 2 or y.size != n:
        return float("nan")
    concordant = 0
    discordant = 0
    tx = 0
    ty = 0
    for i in range(n - 1):
        xd = x[i + 1 :] - x[i]
        yd = y[i + 1 :] - y[i]
        sx = np.sign(xd)
        sy = np.sign(yd)
        prod = sx * sy
        concordant += int(np.sum(prod > 0))
        discordant += int(np.sum(prod < 0))
        tx += int(np.sum((sx == 0) & (sy != 0)))
        ty += int(np.sum((sy == 0) & (sx != 0)))
    n0 = n * (n - 1) // 2
    denom = math.sqrt(max(n0 - tx, 0) * max(n0 - ty, 0))
    if denom <= 0:
        return float("nan")
    return float((concordant - discordant) / denom)


# --------------------------------------------------------------------------- #
# Attribute-sweep metrics
# --------------------------------------------------------------------------- #


@dataclass
class AttributeSweepRecord:
    swept_attr_idx: int
    swept_attr_name: str
    requested_bin: int
    requested_bins_8x4: list[list[int]]
    measured_raw_8x4: list[list[float]]
    measured_binned_8x4: list[list[int]]
    seed: int
    z_seed: int


def metrics_attribute_sweep(records: list[AttributeSweepRecord]) -> dict[str, Any]:
    """
    Compute control metrics for the fixed-z, varied-attribute sweep.

    For each swept attribute, we treat each bar as a sample point: requested
    bin (uniform across all 8 bars) is paired with the bar's measured raw value
    AND its measured bin. Pearson/Spearman/Kendall are computed against the
    measured raw value (continuous) which is the sharpest signal. MAE_bin is
    computed against the measured bin (integer 0..7).
    """
    by_attr: dict[str, list[AttributeSweepRecord]] = {n: [] for n in ATTRIBUTE_NAMES}
    for r in records:
        by_attr[r.swept_attr_name].append(r)

    per_attr_summary: dict[str, dict[str, Any]] = {}
    cross_pearson = np.full((4, 4), np.nan, dtype=float)
    cross_slope = np.full((4, 4), np.nan, dtype=float)

    for ai, name in enumerate(ATTRIBUTE_NAMES):
        rs = sorted(by_attr.get(name, []), key=lambda r: r.requested_bin)
        if not rs:
            per_attr_summary[name] = {"n_pieces": 0}
            continue

        requested = []
        measured_raw_self = []
        measured_bin_self = []
        # For cross-attribute leakage we also need measured raw of OTHER
        # attributes paired with the requested bin of `ai`.
        measured_raw_others: dict[int, list[float]] = {bi: [] for bi in range(4)}

        for r in rs:
            for bar_idx in range(BARS_PER_SAMPLE):
                requested.append(int(r.requested_bin))
                measured_raw_self.append(float(r.measured_raw_8x4[bar_idx][ai]))
                measured_bin_self.append(int(r.measured_binned_8x4[bar_idx][ai]))
                for bi in range(4):
                    measured_raw_others[bi].append(float(r.measured_raw_8x4[bar_idx][bi]))

        req = np.asarray(requested, dtype=float)
        meas_raw = np.asarray(measured_raw_self, dtype=float)
        meas_bin = np.asarray(measured_bin_self, dtype=float)

        pearson = _safe_pearson(req, meas_raw)
        spearman = _safe_spearman(req, meas_raw)
        kendall = _safe_kendall_tau_b(req, meas_raw)
        mae_bin = float(np.mean(np.abs(meas_bin - req)))

        # Dynamic range: mean raw at bin 7 minus mean raw at bin 0 (8 bars each).
        bin0 = [r.measured_raw_8x4[b][ai] for r in rs if r.requested_bin == 0 for b in range(BARS_PER_SAMPLE)]
        bin7 = [r.measured_raw_8x4[b][ai] for r in rs if r.requested_bin == 7 for b in range(BARS_PER_SAMPLE)]
        dyn_range = float("nan")
        if bin0 and bin7:
            dyn_range = float(np.mean(bin7) - np.mean(bin0))

        # Per-bin mean+std of measured raw (handy for plotting / table footnote).
        per_bin_stats = []
        for b in range(8):
            vals = [r.measured_raw_8x4[t][ai] for r in rs if r.requested_bin == b for t in range(BARS_PER_SAMPLE)]
            per_bin_stats.append({
                "bin": b,
                "n": len(vals),
                "mean_raw": float(np.mean(vals)) if vals else float("nan"),
                "std_raw": float(np.std(vals)) if vals else float("nan"),
            })

        per_attr_summary[name] = {
            "n_pieces": len(rs),
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "kendall_tau": kendall,
            "mae_bin": mae_bin,
            "dynamic_range_raw": dyn_range,
            "monotone": (kendall > 0.0) if not math.isnan(kendall) else None,
            "per_bin": per_bin_stats,
        }

        # Cross-attribute leakage: Pearson between requested bin of `ai` and
        # measured raw of every other attribute. Off-diagonal entries should be
        # small if controls are disentangled.
        for bi in range(4):
            other = np.asarray(measured_raw_others[bi], dtype=float)
            r_val = _safe_pearson(req, other)
            cross_pearson[ai, bi] = r_val
            # Also record least-squares slope (per-bin units), useful to express
            # "how many raw units of attribute B leak per +1 bin of attribute A".
            sx = float(np.std(req))
            sy = float(np.std(other))
            if sx <= 1e-12 or sy <= 1e-12 or math.isnan(r_val):
                cross_slope[ai, bi] = float("nan")
            else:
                cross_slope[ai, bi] = float(r_val * sy / sx)

    return {
        "attribute_names": list(ATTRIBUTE_NAMES),
        "per_attribute": per_attr_summary,
        "cross_attribute_pearson": cross_pearson.tolist(),
        "cross_attribute_slope_raw_per_bin": cross_slope.tolist(),
    }


# --------------------------------------------------------------------------- #
# Z-sweep metrics
# --------------------------------------------------------------------------- #


@dataclass
class ZSweepRecord:
    config_name: str
    z_seed: int
    sample_seed: int
    requested_bins_8x4: list[list[int]]
    measured_raw_8x4: list[list[float]]
    measured_binned_8x4: list[list[int]]
    tokens: list[int]


def _pitch_class_histogram(tokens: list[int], id_to_token: dict[int, str]) -> np.ndarray:
    """12-bin pitch-class histogram (normalized to sum=1) from REMI Pitch_* tokens."""
    counts = np.zeros(12, dtype=float)
    for tid in tokens:
        s = id_to_token.get(int(tid), "")
        if s.startswith("Pitch_"):
            try:
                p = int(s.split("_", 1)[1])
                counts[p % 12] += 1.0
            except Exception:
                continue
    total = float(counts.sum())
    if total <= 0.0:
        return counts
    return counts / total


def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu <= 1e-12 or nv <= 1e-12:
        return float("nan")
    cos = float(np.dot(u, v) / (nu * nv))
    return float(1.0 - cos)


def _ngram_counts(seq: list[int], n: int) -> dict[tuple[int, ...], int]:
    out: dict[tuple[int, ...], int] = {}
    for i in range(0, len(seq) - n + 1):
        key = tuple(int(x) for x in seq[i : i + n])
        out[key] = out.get(key, 0) + 1
    return out


def _bleu_n(candidate: list[int], references: list[list[int]], n: int = 3) -> float:
    """Self-BLEU-style geometric mean of clipped n-gram precisions for n=1..N.
    Returns 0..1; lower means more diverse."""
    weights = [1.0 / n] * n
    precisions: list[float] = []
    for k in range(1, n + 1):
        cand_counts = _ngram_counts(candidate, k)
        if not cand_counts:
            precisions.append(0.0)
            continue
        max_ref_counts: dict[tuple[int, ...], int] = {}
        for ref in references:
            for ng, c in _ngram_counts(ref, k).items():
                if c > max_ref_counts.get(ng, 0):
                    max_ref_counts[ng] = c
        clipped = 0
        total = 0
        for ng, c in cand_counts.items():
            clipped += min(c, max_ref_counts.get(ng, 0))
            total += c
        precisions.append(clipped / total if total > 0 else 0.0)
    if any(p <= 0.0 for p in precisions):
        return 0.0
    log_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
    return float(math.exp(log_sum))


def _strip_pad(tokens: list[int], pad_id: int) -> list[int]:
    out: list[int] = []
    for t in tokens:
        if int(t) == int(pad_id):
            break
        out.append(int(t))
    return out


def metrics_z_sweep(
    records: list[ZSweepRecord],
    *,
    id_to_token: dict[int, str],
    pad_id: int,
) -> dict[str, Any]:
    """Compute attribute-consistency + content-diversity metrics across z draws."""
    by_cfg: dict[str, list[ZSweepRecord]] = {}
    for r in records:
        by_cfg.setdefault(r.config_name, []).append(r)

    out: dict[str, Any] = {}
    for cfg_name, rs in by_cfg.items():
        if len(rs) < 2:
            out[cfg_name] = {"n_pieces": len(rs), "note": "need >=2 z draws for diversity metrics"}
            continue

        # Attribute consistency: per-attribute std of measured bin across z draws,
        # averaged over bars.
        bin_stack = np.asarray([r.measured_binned_8x4 for r in rs], dtype=float)  # [n_z, 8, 4]
        std_per_attr = bin_stack.std(axis=0).mean(axis=0)  # [4]
        mean_per_attr = bin_stack.mean(axis=(0, 1))  # [4]
        requested_constant = np.asarray(rs[0].requested_bins_8x4, dtype=float).mean(axis=0)  # [4]

        # Pitch-class diversity across z (pairwise cosine distance).
        hists = [_pitch_class_histogram(r.tokens, id_to_token) for r in rs]
        pairs: list[float] = []
        for i in range(len(hists)):
            for j in range(i + 1, len(hists)):
                pairs.append(_cosine_distance(hists[i], hists[j]))
        pairs_clean = [v for v in pairs if not math.isnan(v)]
        pc_mean = float(np.mean(pairs_clean)) if pairs_clean else float("nan")
        pc_max = float(np.max(pairs_clean)) if pairs_clean else float("nan")

        # Self-BLEU-3 averaged across all (i, others) pairs.
        cleaned = [_strip_pad(r.tokens, pad_id) for r in rs]
        bleu_vals: list[float] = []
        for i, cand in enumerate(cleaned):
            refs = [cleaned[j] for j in range(len(cleaned)) if j != i]
            if not refs or len(cand) < 4:
                continue
            bleu_vals.append(_bleu_n(cand, refs, n=3))
        self_bleu = float(np.mean(bleu_vals)) if bleu_vals else float("nan")

        out[cfg_name] = {
            "n_pieces": len(rs),
            "requested_mean_bin": [float(v) for v in requested_constant.tolist()],
            "measured_mean_bin": [float(v) for v in mean_per_attr.tolist()],
            "attribute_consistency_std_bin": [float(v) for v in std_per_attr.tolist()],
            "attribute_consistency_std_bin_mean": float(np.mean(std_per_attr)),
            "pitch_class_pairwise_cosine_distance_mean": pc_mean,
            "pitch_class_pairwise_cosine_distance_max": pc_max,
            "self_bleu_3": self_bleu,
            "z_seeds": [int(r.z_seed) for r in rs],
        }

    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _save_control_scatter(
    out_path: Path,
    records: list[AttributeSweepRecord],
    summary: dict[str, dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for ai, name in enumerate(ATTRIBUTE_NAMES):
        ax = axes[ai]
        rs = sorted([r for r in records if r.swept_attr_idx == ai], key=lambda r: r.requested_bin)
        xs: list[float] = []
        ys: list[float] = []
        for r in rs:
            for b in range(BARS_PER_SAMPLE):
                xs.append(float(r.requested_bin))
                ys.append(float(r.measured_raw_8x4[b][ai]))
        if xs:
            ax.scatter(xs, ys, s=18, alpha=0.45, label="per bar")
            # Mean per requested bin
            by_bin = {}
            for x, y in zip(xs, ys):
                by_bin.setdefault(int(x), []).append(y)
            mx = sorted(by_bin.keys())
            my = [float(np.mean(by_bin[k])) for k in mx]
            ax.plot(mx, my, "o-", color="C3", linewidth=2.0, label="per-bin mean")

            # Linear fit
            X = np.asarray(xs, dtype=float)
            Y = np.asarray(ys, dtype=float)
            if X.size >= 2 and float(np.std(X)) > 1e-12:
                m, b = np.polyfit(X, Y, 1)
                xx = np.linspace(0, 7, 100)
                ax.plot(xx, m * xx + b, "--", color="gray", alpha=0.7, label=f"fit (slope={m:.2g})")

        s = summary.get(name, {})
        title = (
            f"{ATTRIBUTE_DISPLAY[name]}\n"
            f"r={s.get('pearson_r', float('nan')):.2f}  "
            f"$\\rho$={s.get('spearman_rho', float('nan')):.2f}  "
            f"$\\tau$={s.get('kendall_tau', float('nan')):.2f}  "
            f"MAEbin={s.get('mae_bin', float('nan')):.2f}"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("requested bin (0..7)")
        ax.set_ylabel("measured raw value")
        ax.set_xticks(list(range(8)))
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Attribute control (fixed z, varied attribute)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    import matplotlib.pyplot as _plt  # noqa: E402

    _plt.close(fig)


def _save_cross_attr_heatmap(out_path: Path, attr_metrics: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    M = np.asarray(attr_metrics["cross_attribute_pearson"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            label = f"{v:.2f}" if not math.isnan(v) else "nan"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="black")
    short = ["Poly", "RhyInt", "Vel", "Density"]
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(short)
    ax.set_yticklabels(short)
    ax.set_xlabel("measured attribute")
    ax.set_ylabel("swept attribute (requested)")
    ax.set_title("Cross-attribute Pearson r\n(diagonal = control fidelity, off-diagonal = leakage)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_z_sweep_summary(out_path: Path, z_metrics: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    cfgs = list(z_metrics.keys())
    if not cfgs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: per-attribute std-of-bin across z (lower = controls dominate over z).
    ax = axes[0]
    width = 0.18
    x = np.arange(4)
    short = ["Poly", "RhyInt", "Vel", "Density"]
    for i, cfg in enumerate(cfgs):
        m = z_metrics[cfg]
        if "attribute_consistency_std_bin" not in m:
            continue
        ax.bar(x + (i - (len(cfgs) - 1) / 2.0) * width, m["attribute_consistency_std_bin"], width, label=cfg)
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("std of measured bin across z")
    ax.set_title("Held-attribute consistency vs z\n(lower = attributes dominate over z)", fontsize=10)
    ax.legend(fontsize=8)

    # Right: pitch-class pairwise distance + Self-BLEU-3.
    ax = axes[1]
    pc = [z_metrics[c].get("pitch_class_pairwise_cosine_distance_mean", float("nan")) for c in cfgs]
    sb = [z_metrics[c].get("self_bleu_3", float("nan")) for c in cfgs]
    bx = np.arange(len(cfgs))
    ax.bar(bx - 0.2, pc, 0.4, label="pitch-class cos dist (higher=more diverse)")
    ax.bar(bx + 0.2, sb, 0.4, label="Self-BLEU-3 (lower=more diverse)")
    ax.set_xticks(bx)
    ax.set_xticklabels(cfgs)
    ax.set_title("Content diversity from z\n(fixed attributes)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# LaTeX table
# --------------------------------------------------------------------------- #


def _fmt(x: Any, fmt: str = "{:.3f}") -> str:
    if x is None:
        return "--"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "--"
    except Exception:
        pass
    try:
        return fmt.format(float(x))
    except Exception:
        return str(x)


def _render_latex_table(
    out_path: Path,
    attr_metrics: dict[str, Any],
    z_metrics: dict[str, Any],
    *,
    n_seeds_per_bin: int,
    z_seeds_per_cfg: dict[str, int],
) -> None:
    lines: list[str] = []
    lines.append("% Auto-generated by ml/src/musicgen/analysis/control_eval.py.")
    lines.append("% Drop into report.tex with \\input{...}.")
    lines.append("\\begin{table}[t]")
    lines.append(
        "\\caption{Controllability of the hierarchical VAE (vae\\_v3). "
        "Top: each attribute is swept over bins 0--7 with a fixed $z_p$ and "
        f"{n_seeds_per_bin} seed(s) per bin (per-bar samples = $8\\times 8\\times \\text{{seeds}}$); "
        "Pearson $r$ / Spearman $\\rho$ / Kendall $\\tau$ are computed between requested bin and "
        "the per-bar measured raw value, MAE\\textsubscript{bin} between requested and measured bin, "
        "and dynamic range as raw at bin 7 minus raw at bin 0. Bottom: held-attribute "
        "$z$-sweep -- low std of measured bin across $z$ means attributes dominate the latent, "
        "while non-zero pitch-class distance and Self-BLEU-3 $<1$ confirm $z$ injects "
        "musical variety.}"
    )
    lines.append("\\label{tab:control_eval}")
    lines.append("\\centering")
    lines.append("\\small")

    # ---- Attribute sweep ----
    lines.append("\\begin{tabular}{l r r r r r}")
    lines.append("\\toprule")
    lines.append("Attribute & Pearson $r$ & Spearman $\\rho$ & Kendall $\\tau$ & MAE\\textsubscript{bin} & Dyn.\\ range \\\\")
    lines.append("\\midrule")
    for name in ATTRIBUTE_NAMES:
        s = attr_metrics["per_attribute"].get(name, {})
        lines.append(
            f"{ATTRIBUTE_DISPLAY[name]} & "
            f"{_fmt(s.get('pearson_r'))} & "
            f"{_fmt(s.get('spearman_rho'))} & "
            f"{_fmt(s.get('kendall_tau'))} & "
            f"{_fmt(s.get('mae_bin'), '{:.2f}')} & "
            f"{_fmt(s.get('dynamic_range_raw'), '{:.2f}')} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    # ---- Z sweep ----
    if z_metrics:
        lines.append("\\par\\medskip")
        lines.append("\\begin{tabular}{l r r r r}")
        lines.append("\\toprule")
        lines.append(
            "Config & $|z|$ & mean std(bin) over $z$ & PC cos dist & Self-BLEU-3 \\\\"
        )
        lines.append("\\midrule")
        for cfg, m in z_metrics.items():
            n_z = z_seeds_per_cfg.get(cfg, m.get("n_pieces", 0))
            std_bin = m.get("attribute_consistency_std_bin_mean")
            pc = m.get("pitch_class_pairwise_cosine_distance_mean")
            sb = m.get("self_bleu_3")
            lines.append(
                f"{cfg.replace('_', ' ')} & "
                f"{int(n_z)} & "
                f"{_fmt(std_bin, '{:.2f}')} & "
                f"{_fmt(pc, '{:.3f}')} & "
                f"{_fmt(sb, '{:.3f}')} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #


@dataclass
class ControlEvalConfig:
    out_dir: str = "artifacts/control_eval"
    run_id: Optional[str] = None  # default = timestamped

    ckpt_path: str = "ml/src/musicgen/runs/vae_v3/ckpt.pt"
    quantiles_path: str = "artifacts/attributes/quantiles.json"

    # Generation
    device: str = "auto"
    block_size: int = BLOCK_SIZE
    bars_per_sample: int = BARS_PER_SAMPLE
    temperature: float = 0.95
    top_p: float = 0.9

    # Attribute sweep
    attr_seeds: list[int] = field(default_factory=lambda: [2026])
    attr_z_seed: int = 2026
    neutral_bin: int = 4

    # z sweep
    z_seeds: list[int] = field(default_factory=lambda: [11, 42, 123, 256, 1024, 4096])
    z_attr_configs: dict[str, list[list[int]]] = field(
        default_factory=lambda: {
            "neutral": [[4, 4, 4, 4]] * BARS_PER_SAMPLE,
            "phased_low_to_high": (
                [[1, 1, 2, 1]] * 4 + [[6, 6, 6, 7]] * 4
            ),
        }
    )
    z_sample_seed_offset: int = 0  # use z_seed itself as sample seed by default


def run_control_eval(cfg: ControlEvalConfig) -> Path:
    """Run both sweeps and emit JSON + LaTeX + figures. Returns the run directory."""
    from datetime import datetime, timezone

    run_id = cfg.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = _resolve_path(cfg.out_dir) / str(run_id)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)

    device = _device_from_str(cfg.device)
    loaded = load_vae_v3(
        ckpt_path=cfg.ckpt_path,
        device=device,
        block_size=cfg.block_size,
        bars_per_sample=cfg.bars_per_sample,
    )
    thresholds = load_thresholds(cfg.quantiles_path)

    print(f"[control_eval] run dir: {out_root}")
    print(f"[control_eval] device: {device}")

    # ---- Attribute sweep (fixed z) ----
    z_fixed = sample_fixed_z(cfg.attr_z_seed, z_dim=loaded.z_dim, device=device)
    attr_records: list[AttributeSweepRecord] = []
    for ai, name in enumerate(ATTRIBUTE_NAMES):
        for b in range(8):
            for s_idx, seed in enumerate(cfg.attr_seeds):
                requested = make_uniform_attribute_matrix(ai, b, neutral_bin=cfg.neutral_bin)
                tokens = generate_one(
                    loaded=loaded,
                    attributes=requested,
                    z_p=z_fixed,
                    seed=int(seed),
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    block_size=cfg.block_size,
                    bars_per_sample=cfg.bars_per_sample,
                    device=device,
                )
                measured = measure_attributes(
                    tokens,
                    id_to_token=loaded.tok_bundle.id_to_token,
                    bar_id=loaded.tok_bundle.bar_id,
                    thresholds=thresholds,
                    bars_per_sample=cfg.bars_per_sample,
                )
                rec = AttributeSweepRecord(
                    swept_attr_idx=ai,
                    swept_attr_name=name,
                    requested_bin=b,
                    requested_bins_8x4=requested,
                    measured_raw_8x4=measured["raw"],  # type: ignore[arg-type]
                    measured_binned_8x4=measured["binned"],  # type: ignore[arg-type]
                    seed=int(seed),
                    z_seed=cfg.attr_z_seed,
                )
                attr_records.append(rec)

                piece_dir = out_root / "sweeps" / name / f"bin_{b}"
                if len(cfg.attr_seeds) > 1:
                    piece_dir = piece_dir / f"seed_{seed}"
                save_piece_artifacts(
                    out_dir=piece_dir,
                    tokens=tokens,
                    requested_attrs=requested,
                    measured=measured,
                    tok_bundle=loaded.tok_bundle,
                    extra_meta={
                        "swept_attr": name,
                        "requested_bin": b,
                        "neutral_bin": int(cfg.neutral_bin),
                        "seed": int(seed),
                        "z_seed": int(cfg.attr_z_seed),
                        "model": "vae_v3",
                    },
                )
                print(f"[attr-sweep] {name} bin={b} seed={seed} -> {piece_dir.relative_to(out_root)}")

    attr_metrics = metrics_attribute_sweep(attr_records)

    # ---- Z sweep (fixed attributes, varied z) ----
    z_records: list[ZSweepRecord] = []
    for cfg_name, attr_matrix in cfg.z_attr_configs.items():
        for z_seed in cfg.z_seeds:
            sample_seed = int(z_seed) + int(cfg.z_sample_seed_offset)
            z = sample_fixed_z(int(z_seed), z_dim=loaded.z_dim, device=device)
            tokens = generate_one(
                loaded=loaded,
                attributes=attr_matrix,
                z_p=z,
                seed=sample_seed,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                block_size=cfg.block_size,
                bars_per_sample=cfg.bars_per_sample,
                device=device,
            )
            measured = measure_attributes(
                tokens,
                id_to_token=loaded.tok_bundle.id_to_token,
                bar_id=loaded.tok_bundle.bar_id,
                thresholds=thresholds,
                bars_per_sample=cfg.bars_per_sample,
            )
            rec = ZSweepRecord(
                config_name=cfg_name,
                z_seed=int(z_seed),
                sample_seed=int(sample_seed),
                requested_bins_8x4=[list(map(int, row)) for row in attr_matrix],
                measured_raw_8x4=measured["raw"],  # type: ignore[arg-type]
                measured_binned_8x4=measured["binned"],  # type: ignore[arg-type]
                tokens=[int(t) for t in tokens],
            )
            z_records.append(rec)

            piece_dir = out_root / "z_sweep" / cfg_name / f"seed_{z_seed}"
            save_piece_artifacts(
                out_dir=piece_dir,
                tokens=tokens,
                requested_attrs=attr_matrix,
                measured=measured,
                tok_bundle=loaded.tok_bundle,
                extra_meta={
                    "config": cfg_name,
                    "z_seed": int(z_seed),
                    "sample_seed": int(sample_seed),
                    "model": "vae_v3",
                },
            )
            print(f"[z-sweep ] {cfg_name} z_seed={z_seed} -> {piece_dir.relative_to(out_root)}")

    z_metrics = metrics_z_sweep(
        z_records,
        id_to_token=loaded.tok_bundle.id_to_token,
        pad_id=loaded.tok_bundle.pad_id,
    )

    # ---- Persist metrics report ----
    metrics_report = {
        "run_id": str(run_id),
        "config": {
            "ckpt_path": str(_resolve_path(cfg.ckpt_path)),
            "quantiles_path": str(_resolve_path(cfg.quantiles_path)),
            "block_size": int(cfg.block_size),
            "bars_per_sample": int(cfg.bars_per_sample),
            "temperature": float(cfg.temperature),
            "top_p": float(cfg.top_p),
            "attr_seeds": list(cfg.attr_seeds),
            "attr_z_seed": int(cfg.attr_z_seed),
            "neutral_bin": int(cfg.neutral_bin),
            "z_seeds": list(cfg.z_seeds),
            "z_attr_configs": cfg.z_attr_configs,
        },
        "attribute_sweep": attr_metrics,
        "z_sweep": z_metrics,
    }
    (out_root / "metrics_report.json").write_text(
        json.dumps(_json_safe(metrics_report), indent=2, allow_nan=False), encoding="utf-8"
    )

    # ---- Render figures ----
    _save_control_scatter(out_root / "figures" / "control_scatter.png", attr_records, attr_metrics["per_attribute"])
    _save_cross_attr_heatmap(out_root / "figures" / "cross_attr_heatmap.png", attr_metrics)
    if z_metrics:
        _save_z_sweep_summary(out_root / "figures" / "z_sweep_summary.png", z_metrics)

    # ---- Render LaTeX table ----
    _render_latex_table(
        out_root / "control_eval_table.tex",
        attr_metrics,
        z_metrics,
        n_seeds_per_bin=len(cfg.attr_seeds),
        z_seeds_per_cfg={cfg_name: len(cfg.z_seeds) for cfg_name in cfg.z_attr_configs.keys()},
    )

    print(f"[control_eval] wrote {out_root / 'metrics_report.json'}")
    print(f"[control_eval] wrote {out_root / 'control_eval_table.tex'}")
    return out_root
