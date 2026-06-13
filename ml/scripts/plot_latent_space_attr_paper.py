#!/usr/bin/env python3
"""Side-by-side UMAP of encoder μ colored by attribute-regime clusters (k=4)."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

_DISTINCT_HEX: list[str] = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#5D3A9A",
]


def _distinct_colors(k: int) -> list[tuple[float, float, float]]:
    colors = [mcolors.to_rgb(_DISTINCT_HEX[i]) for i in range(min(k, len(_DISTINCT_HEX)))]
    if k > len(colors):
        for j in range(k - len(colors)):
            h = ((len(colors) + j) * 0.618033988749895) % 1.0
            colors.append(colorsys.hsv_to_rgb(h, 0.90, 0.92))
    return colors[:k]


def _discrete_cmap(k: int) -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(_distinct_colors(k))
    norm = BoundaryNorm(np.arange(-0.5, k + 0.5, 1.0), cmap.N)
    return cmap, norm


def _load_umap(folder: Path) -> np.ndarray:
    path = folder / "e3_umap.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path)["umap2"], dtype=np.float64)


def _load_labels(folder: Path) -> np.ndarray:
    path = folder / "feature_pca.npz"
    labels = np.asarray(np.load(path)["labels"], dtype=np.int64)
    return labels


def make_figure(*, simple_dir: Path, vae_dir: Path, out_path: Path) -> None:
    simple_dir, vae_dir = simple_dir.resolve(), vae_dir.resolve()
    labels_s = _load_labels(simple_dir)
    labels_v = _load_labels(vae_dir)
    if labels_s.shape != labels_v.shape or not np.array_equal(labels_s, labels_v):
        raise ValueError("cluster labels differ between simple and vae folders")

    umap_s = _load_umap(simple_dir)
    umap_v = _load_umap(vae_dir)
    k = int(labels_s.max()) + 1
    cmap, norm = _discrete_cmap(k)

    pur_s = json.loads((simple_dir / "e3_knn_purity_clusters.json").read_text())
    pur_v = json.loads((vae_dir / "e3_knn_purity_clusters.json").read_text())

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.82, bottom=0.14, wspace=0.22)

    panels = [
        (axes[0], umap_s, "Simple VAE", pur_s),
        (axes[1], umap_v, "Hierarchical VAE", pur_v),
    ]
    for ax, xy, title, pur in panels:
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=labels_s,
            cmap=cmap,
            norm=norm,
            s=3.0,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
        subtitle = (
            f"kNN purity = {float(pur['knn_purity_umap']):.3f}  ·  "
            f"NMI = {float(pur['nmi']):.3f}"
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xlabel("UMAP-1", fontsize=9)
        ax.set_ylabel("UMAP-2", fontsize=9)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    cax = fig.add_axes([0.905, 0.16, 0.02, 0.62])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=np.arange(k))
    cb.set_label("Attribute-regime cluster", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Encoder mean $\\mu$ projected to UMAP, colored by attribute-regime clusters ($k={k}$)",
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", out_path)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--simple",
        type=Path,
        default=root / "artifacts/eval/latent_vis_simple_attr",
    )
    p.add_argument(
        "--vae",
        type=Path,
        default=root / "artifacts/eval/latent_vis_vae_attr",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "assets/piano-music-generative-models/figures/latent_space_attr_umap_k4.png",
    )
    args = p.parse_args()
    make_figure(simple_dir=args.simple, vae_dir=args.vae, out_path=args.out)


if __name__ == "__main__":
    main()
