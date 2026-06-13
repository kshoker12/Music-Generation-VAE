#!/usr/bin/env python3
"""
Build a 3x2 (or 3x1) publication-style figure comparing MidiBERT feature-space clusters
to Simple VAE and Hierarchical VAE latent projections — same artifact layout as the
Yang & Lerch latent plots (`feature_pca.npz`, `feature_umap.npz`, `e3_latent_mu.npz`, etc.).

Expects two Kaggle output folders with **identical** `cluster_labels` in `feature_pca.npz`
(same seeds / pipeline), e.g.:
  artifacts/eval/latent_vis_simple_bert/
  artifacts/eval/latent_vis_vae_bert/

Example:
  python ml/scripts/plot_latent_organization_bert.py \\
    --simple artifacts/eval/latent_vis_simple_bert \\
    --vae artifacts/eval/latent_vis_vae_bert \\
    --out artifacts/eval/latent_organization_bert_k4.png
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_feature_bundle(folder: Path) -> dict[str, Any]:
    pca_path = folder / "feature_pca.npz"
    if not pca_path.exists():
        raise FileNotFoundError(f"Missing {pca_path}")
    z = np.load(pca_path)
    pca2 = np.asarray(z["pca2"], dtype=np.float64)
    labels = np.asarray(z["labels"], dtype=np.int64)
    umap2 = None
    u_path = folder / "feature_umap.npz"
    if u_path.exists():
        zu = np.load(u_path)
        umap2 = np.asarray(zu["umap2"], dtype=np.float64)
    return {"pca2": pca2, "umap2": umap2, "labels": labels}


def _load_latent_bundle(folder: Path) -> dict[str, Any]:
    mu_path = folder / "e3_latent_mu.npz"
    if not mu_path.exists():
        raise FileNotFoundError(f"Missing {mu_path}")
    zm = np.load(mu_path)
    pca2 = np.asarray(zm["pca2"], dtype=np.float64)
    umap2 = None
    u_path = folder / "e3_umap.npz"
    if u_path.exists():
        zu = np.load(u_path)
        umap2 = np.asarray(zu["umap2"], dtype=np.float64)
    return {"pca2": pca2, "umap2": umap2}


def _k_from_labels(labels: np.ndarray) -> int:
    return int(labels.max()) + 1 if labels.size else 0


# High-contrast qualitative sRGB hexes (Okabe–Ito–style first, then vivid extensions).
# For k larger than this list, remaining colors use golden-ratio–spaced hues in HSV.
_DISTINCT_HEX: list[str] = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#5D3A9A",
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00",
    "#A65628",
    "#F781BF",
    "#66C2A5",
    "#FC8D62",
    "#8DA0CB",
    "#E78AC3",
    "#A6D854",
    "#FFD92F",
    "#B3DE69",
    "#80B1D3",
    "#FB8072",
    "#BEBADA",
    "#BC80BD",
    "#CCEBC5",
    "#FFED6F",
]


def _distinct_colors(k: int) -> list[tuple[float, float, float]]:
    if k <= 0:
        return []
    hexes = [_DISTINCT_HEX[i] for i in range(min(k, len(_DISTINCT_HEX)))]
    colors = [mcolors.to_rgb(h) for h in hexes]
    if k > len(colors):
        start = len(colors)
        for j in range(k - start):
            h = ((start + j) * 0.618033988749895) % 1.0
            colors.append(colorsys.hsv_to_rgb(h, 0.90, 0.92))
    return colors[:k]


def _discrete_cmap(k: int) -> tuple[ListedColormap, BoundaryNorm]:
    colors = _distinct_colors(k)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, k + 0.5, 1.0), cmap.N)
    return cmap, norm


def _panel(
    ax,
    xy: np.ndarray,
    labels: np.ndarray,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    title: str,
    xlab: str,
    ylab: str,
    *,
    point_size: float = 2.5,
    alpha: float = 0.62,
) -> None:
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=labels,
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _purity_block(p: dict[str, Any]) -> str:
    lines: list[str] = []
    if p.get("knn_purity_pca") is not None:
        lines.append(f"kNN-PCA={float(p['knn_purity_pca']):.3f}")
    if p.get("knn_purity_umap") is not None:
        lines.append(f"kNN-UMAP={float(p['knn_purity_umap']):.3f}")
    lines.append(f"NMI={float(p['nmi']):.3f}")
    lines.append(f"ARI={float(p['ari']):.3f}")
    return "\n".join(lines)


def make_figure(
    *,
    simple_dir: Path,
    vae_dir: Path,
    out_path: Path,
    suptitle: str | None = None,
) -> None:
    simple_dir = simple_dir.resolve()
    vae_dir = vae_dir.resolve()

    feat_s = _load_feature_bundle(simple_dir)
    feat_v = _load_feature_bundle(vae_dir)
    lab_s, lab_v = feat_s["labels"], feat_v["labels"]
    if lab_s.shape != lab_v.shape or not np.array_equal(lab_s, lab_v):
        raise ValueError(
            "feature cluster labels differ between --simple and --vae folders "
            "(expected identical test order and k-means labels)."
        )
    feat_labels = lab_s
    k = _k_from_labels(feat_labels)
    if k <= 0:
        raise ValueError("empty or invalid cluster labels")

    pur_s = _load_json(simple_dir / "e3_knn_purity_clusters.json")
    pur_v = _load_json(vae_dir / "e3_knn_purity_clusters.json")
    k_json = int(pur_s.get("k_best", k))
    if int(pur_v.get("k_best", k)) != k_json:
        raise ValueError(f"k_best mismatch: simple={pur_s.get('k_best')} vae={pur_v.get('k_best')}")

    lat_s = _load_latent_bundle(simple_dir)
    lat_v = _load_latent_bundle(vae_dir)

    cmap, norm = _discrete_cmap(k)

    has_umap = (
        feat_s["umap2"] is not None
        and feat_v["umap2"] is not None
        and lat_s["umap2"] is not None
        and lat_v["umap2"] is not None
    )
    n_data_cols = 2 if has_umap else 1
    n_grid_cols = 1 + n_data_cols

    plt.rcParams["font.family"] = "DejaVu Sans"

    fig_w = 11.0 if has_umap else 7.0
    fig_h = 11.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    wr = [0.52] + [1.0] * n_data_cols
    gs = GridSpec(
        3,
        n_grid_cols,
        figure=fig,
        width_ratios=wr,
        wspace=0.28,
        hspace=0.50,
        left=0.08,
        right=0.86,
        top=0.90,
        bottom=0.07,
    )

    def add_label_row(row: int, text: str) -> None:
        axl = fig.add_subplot(gs[row, 0])
        axl.axis("off")
        axl.text(
            1.0,
            0.5,
            text,
            transform=axl.transAxes,
            va="center",
            ha="right",
            fontsize=9,
            linespacing=1.2,
        )

    # Row 0 — MidiBERT feature space
    add_label_row(0, "K-means clusters\non MidiBERT\nembeddings\n(ground truth)")
    ax_p0 = fig.add_subplot(gs[0, 1])
    _panel(
        ax_p0,
        feat_s["pca2"],
        feat_labels,
        cmap,
        norm,
        f"Feature space — PCA (k={k_json})",
        "PC1",
        "PC2",
    )
    if has_umap:
        ax_u0 = fig.add_subplot(gs[0, 2])
        _panel(
            ax_u0,
            feat_s["umap2"],
            feat_labels,
            cmap,
            norm,
            f"Feature space — UMAP (k={k_json})",
            "UMAP-1",
            "UMAP-2",
        )

    # Rows 1–2 — latents
    rows = [
        (1, "Simple VAE", lat_s, pur_s),
        (2, "Hierarchical VAE", lat_v, pur_v),
    ]
    for r, name, lat, pur in rows:
        add_label_row(r, f"{name}\n{_purity_block(pur)}")
        ax_p = fig.add_subplot(gs[r, 1])
        t_p = f"{name} latent — PCA"
        if pur.get("knn_purity_pca") is not None:
            t_p += f"\nkNN purity={float(pur['knn_purity_pca']):.3f}"
        _panel(ax_p, lat["pca2"], feat_labels, cmap, norm, t_p, "PC1", "PC2")
        if has_umap:
            ax_u = fig.add_subplot(gs[r, 2])
            t_u = f"{name} latent — UMAP"
            if pur.get("knn_purity_umap") is not None:
                t_u += f"\nkNN purity={float(pur['knn_purity_umap']):.3f}"
            _panel(ax_u, lat["umap2"], feat_labels, cmap, norm, t_u, "UMAP-1", "UMAP-2")

    cax = fig.add_axes([0.905, 0.12, 0.02, 0.72])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=np.arange(k))
    cb.set_label("Cluster id", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    st = suptitle or f"Latent organization vs feature-space clusters (k={k_json})"
    fig.suptitle(st, fontsize=12, fontweight="bold", y=0.97)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simple", type=Path, required=True, help="Folder with Simple VAE Kaggle outputs")
    p.add_argument("--vae", type=Path, required=True, help="Folder with Hierarchical VAE Kaggle outputs")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: parent of --simple / latent_organization_bert_k{k}.png)",
    )
    p.add_argument("--title", type=str, default=None, help="Override figure suptitle")
    args = p.parse_args()

    if args.out is None:
        k_hint = _k_from_labels(_load_feature_bundle(args.simple)["labels"])
        args.out = args.simple.resolve().parent / f"latent_organization_bert_k{k_hint}.png"

    make_figure(simple_dir=args.simple, vae_dir=args.vae, out_path=args.out, suptitle=args.title)


if __name__ == "__main__":
    main()
