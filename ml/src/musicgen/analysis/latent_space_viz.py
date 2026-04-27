from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class LatentVizConfig:
    test_split_dir: str = "artifacts/preprocessed_memmap/test"
    seed: int = 123
    n_samples: int = 2000
    batch_size: int = 64

    ckpt_vae_v3: str = "ml/src/musicgen/runs/vae_v3/ckpt.pt"
    ckpt_simple_v2: str = "ml/src/musicgen/runs/simple_v2/ckpt.pt"

    out_dir: str = "artifacts/latent_viz"
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # Embedding choice: "mu" is deterministic; "z" uses reparameterization noise.
    embedding: str = "mu"  # "mu" | "z"

    # Preprocessing / DR
    standardize: bool = True
    pca_dim: int = 50  # 0 disables PCA
    umap_neighbors: int = 30
    umap_min_dist: float = 0.05

    # Clustering
    kmeans_k: int = 12
    hdbscan_min_cluster_size: int = 25
    hdbscan_min_samples: Optional[int] = None


def _repo_root() -> Path:
    # repo_root/.../ml/src/musicgen/analysis/latent_space_viz.py
    here = Path(__file__).resolve()
    for p in [here.parents[i] for i in range(0, 10)]:
        if (p / "ml" / "src" / "musicgen").is_dir():
            return p
    return Path.cwd()


def _resolve_path(p: str) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((_repo_root() / pp).resolve())


def _load_checkpoint(path: str) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict.")
    return ckpt


def _extract_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict")))
    if state is None:
        if all(isinstance(k, str) for k in ckpt.keys()):
            # sometimes the dict itself is the state dict
            state = ckpt  # type: ignore[assignment]
        else:
            raise ValueError("Could not find model state dict in checkpoint.")
    if not isinstance(state, dict):
        raise ValueError("Model state must be a dict.")
    return state  # type: ignore[return-value]


def _device_from_cfg(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sample_indices(n_total: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = min(int(n_samples), int(n_total))
    return rng.choice(n_total, size=n, replace=False)


def _reduce_attributes(attrs: np.ndarray) -> np.ndarray:
    """
    attrs: uint8 [N,8,4]
    returns: int64 [N,4] mean-over-bars rounded to nearest int bin.
    """
    mean_bins = attrs.astype(np.float32).mean(axis=1)
    return np.rint(mean_bins).astype(np.int64)


@torch.no_grad()
def _encode_mu_logvar(model: torch.nn.Module, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    enc = getattr(model, "encoder", None)
    if enc is None:
        raise ValueError("Model has no .encoder attribute.")
    mu, logvar = enc(X)
    return mu, logvar


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def extract_latents_for_model(
    *,
    model_type: str,
    ckpt_path: str,
    X_np: np.ndarray,
    cfg: LatentVizConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    ckpt_path = _resolve_path(ckpt_path)
    ckpt = _load_checkpoint(ckpt_path)
    state = _extract_state_dict(ckpt)

    vocab_size = 195
    block_size = int(X_np.shape[1])
    bars = 8

    if model_type == "vae":
        from musicgen.models.vae import MusicVAE, VAEConfig

        cfg_dict = ckpt.get("cfg", {})
        mcfg = VAEConfig(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else VAEConfig()
        mcfg = VAEConfig(**{**mcfg.__dict__, "vocab_size": vocab_size, "block_size": block_size, "bars_per_sample": bars})
        model = MusicVAE(mcfg)
    elif model_type == "simple_vae":
        from musicgen.models.simple_vae import SimpleMusicVAE, SimpleVAEConfig

        cfg_dict = ckpt.get("cfg", {})
        mcfg = SimpleVAEConfig(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else SimpleVAEConfig()
        mcfg = SimpleVAEConfig(
            **{**mcfg.__dict__, "vocab_size": vocab_size, "block_size": block_size, "bars_per_sample": bars}
        )
        model = SimpleMusicVAE(mcfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    mu_all: list[np.ndarray] = []
    logvar_all: list[np.ndarray] = []
    z_all: list[np.ndarray] = []

    bs = int(cfg.batch_size)
    for i in range(0, X_np.shape[0], bs):
        xb = torch.from_numpy(X_np[i : i + bs].astype(np.int64, copy=False)).to(device=device)
        mu, logvar = _encode_mu_logvar(model, xb)
        mu_all.append(mu.detach().cpu().numpy())
        logvar_all.append(logvar.detach().cpu().numpy())
        if cfg.embedding == "z":
            z = _reparameterize(mu, logvar)
            z_all.append(z.detach().cpu().numpy())

    out: dict[str, np.ndarray] = {
        "mu": np.concatenate(mu_all, axis=0),
        "logvar": np.concatenate(logvar_all, axis=0),
    }
    if cfg.embedding == "z":
        out["z"] = np.concatenate(z_all, axis=0)
    return out


def run_latent_space_viz(cfg: LatentVizConfig) -> Path:
    from musicgen.data.memmap_dataset import MemmapMusicDataset

    device = _device_from_cfg(cfg.device)

    split_dir = Path(_resolve_path(cfg.test_split_dir))
    ds = MemmapMusicDataset(split_dir)

    idx = _sample_indices(len(ds), cfg.n_samples, cfg.seed)

    # Load selected items into contiguous arrays for fast batching
    X_list: list[np.ndarray] = []
    attrs_list: list[np.ndarray] = []
    for j in idx:
        x, _bar_idx, attrs = ds[int(j)]
        X_list.append(np.asarray(x))
        attrs_list.append(np.asarray(attrs))

    X_np = np.stack(X_list, axis=0).astype(np.int64, copy=False)  # [N,1024]
    attrs_np = np.stack(attrs_list, axis=0).astype(np.int64, copy=False)  # [N,8,4]
    attrs_mean = _reduce_attributes(attrs_np)  # [N,4]

    out_root = Path(_resolve_path(cfg.out_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Latent extraction ---
    vae = extract_latents_for_model(model_type="vae", ckpt_path=cfg.ckpt_vae_v3, X_np=X_np, cfg=cfg, device=device)
    simple = extract_latents_for_model(
        model_type="simple_vae", ckpt_path=cfg.ckpt_simple_v2, X_np=X_np, cfg=cfg, device=device
    )

    emb_key = "mu" if cfg.embedding == "mu" else "z"
    emb_vae = vae[emb_key]
    emb_simple = simple[emb_key]

    # --- Clustering / DR / plots ---
    # Fix a macOS/numba caching issue some environments hit when importing umap.
    os.environ.setdefault("NUMBA_CACHE_DIR", str((_repo_root() / "artifacts" / "numba_cache").resolve()))

    from sklearn.decomposition import PCA
    from sklearn.metrics import normalized_mutual_info_score, silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    import umap
    import hdbscan
    import matplotlib.pyplot as plt

    def preprocess(E: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        meta: dict[str, Any] = {}
        X = E.astype(np.float32, copy=False)
        if cfg.standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            meta["standardize"] = True
        if int(cfg.pca_dim) and int(cfg.pca_dim) > 0 and int(cfg.pca_dim) < X.shape[1]:
            pca = PCA(n_components=int(cfg.pca_dim), random_state=int(cfg.seed))
            X = pca.fit_transform(X)
            meta["pca_dim"] = int(cfg.pca_dim)
            meta["pca_explained_var"] = float(np.sum(pca.explained_variance_ratio_))
        return X, meta

    def cluster_and_embed(Xp: np.ndarray) -> dict[str, Any]:
        res: dict[str, Any] = {}

        # KMeans
        km = KMeans(n_clusters=int(cfg.kmeans_k), n_init="auto", random_state=int(cfg.seed))
        km_labels = km.fit_predict(Xp)
        res["kmeans"] = {"k": int(cfg.kmeans_k)}
        res["kmeans"]["silhouette"] = float(silhouette_score(Xp, km_labels)) if len(np.unique(km_labels)) > 1 else None

        # HDBSCAN
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=int(cfg.hdbscan_min_cluster_size),
            min_samples=None if cfg.hdbscan_min_samples is None else int(cfg.hdbscan_min_samples),
        )
        hdb_labels = hdb.fit_predict(Xp)
        non_noise = hdb_labels >= 0
        if non_noise.sum() >= 5 and len(np.unique(hdb_labels[non_noise])) > 1:
            sil = float(silhouette_score(Xp[non_noise], hdb_labels[non_noise]))
        else:
            sil = None
        res["hdbscan"] = {
            "min_cluster_size": int(cfg.hdbscan_min_cluster_size),
            "min_samples": None if cfg.hdbscan_min_samples is None else int(cfg.hdbscan_min_samples),
            "n_clusters": int(len(set(int(x) for x in hdb_labels.tolist() if x >= 0))),
            "noise_frac": float(np.mean(~non_noise)),
            "silhouette_non_noise": sil,
        }

        # UMAP
        reducer = umap.UMAP(
            n_neighbors=int(cfg.umap_neighbors),
            min_dist=float(cfg.umap_min_dist),
            n_components=2,
            metric="euclidean",
            random_state=int(cfg.seed),
        )
        um = reducer.fit_transform(Xp)
        res["umap"] = um.astype(np.float32)
        res["labels"] = {"kmeans": km_labels.astype(np.int32), "hdbscan": hdb_labels.astype(np.int32)}
        return res

    def save_plots(prefix: str, um: np.ndarray, labels: dict[str, np.ndarray]) -> None:
        # Cluster plots
        for lname, lab in labels.items():
            plt.figure(figsize=(7, 6))
            plt.scatter(um[:, 0], um[:, 1], c=lab, s=8, cmap="tab20", alpha=0.9)
            plt.title(f"{prefix}: UMAP colored by {lname}")
            plt.tight_layout()
            plt.savefig(out_root / f"{prefix}_umap_{lname}.png", dpi=200)
            plt.close()

        # Attribute plots (mean-over-bars)
        attr_names = ["polyphony", "rhythmic_intensity", "velocity", "note_density"]
        for ai, an in enumerate(attr_names):
            plt.figure(figsize=(7, 6))
            plt.scatter(um[:, 0], um[:, 1], c=attrs_mean[:, ai], s=8, cmap="viridis", alpha=0.9)
            plt.colorbar(label=f"{an}_bin")
            plt.title(f"{prefix}: UMAP colored by {an} (mean bin)")
            plt.tight_layout()
            plt.savefig(out_root / f"{prefix}_umap_attr_{an}.png", dpi=200)
            plt.close()

    def label_summary(cluster_labels: np.ndarray) -> dict[str, Any]:
        # cluster_labels: [N]
        # summarize attribute distributions per cluster
        clusters = sorted(set(int(c) for c in cluster_labels.tolist()))
        out: dict[str, Any] = {}
        for c in clusters:
            mask = cluster_labels == c
            if mask.sum() == 0:
                continue
            a = attrs_mean[mask]  # [n,4]
            # majority per attribute
            majority = []
            dist = []
            for k in range(a.shape[1]):
                counts = np.bincount(a[:, k].astype(np.int64), minlength=8)
                majority.append(int(np.argmax(counts)))
                dist.append((counts / max(int(mask.sum()), 1)).tolist())
            out[str(c)] = {"n": int(mask.sum()), "majority_bins": majority, "bin_distributions": dist}
        return out

    def nmi_per_attribute(cluster_labels: np.ndarray) -> dict[str, float | None]:
        attr_names = ["polyphony", "rhythmic_intensity", "velocity", "note_density"]
        out: dict[str, float | None] = {}
        for ai, an in enumerate(attr_names):
            try:
                out[an] = float(normalized_mutual_info_score(cluster_labels, attrs_mean[:, ai]))
            except Exception:
                out[an] = None
        return out

    # preprocess + run per model
    vae_p, vae_meta = preprocess(emb_vae)
    simple_p, simple_meta = preprocess(emb_simple)

    vae_res = cluster_and_embed(vae_p)
    simple_res = cluster_and_embed(simple_p)

    save_plots("vae_v3", vae_res["umap"], vae_res["labels"])
    save_plots("simple_v2", simple_res["umap"], simple_res["labels"])

    # --- Metrics report ---
    report: dict[str, Any] = {
        "config": asdict(cfg),
        "sample": {"n_total_test": int(len(ds)), "n_used": int(X_np.shape[0]), "seed": int(cfg.seed)},
        "preprocess": {"vae_v3": vae_meta, "simple_v2": simple_meta},
        "models": {
            "vae_v3": {"ckpt": _resolve_path(cfg.ckpt_vae_v3)},
            "simple_v2": {"ckpt": _resolve_path(cfg.ckpt_simple_v2)},
        },
        "kmeans": {
            "vae_v3": {
                "silhouette": vae_res["kmeans"]["silhouette"],
                "nmi_by_attr": nmi_per_attribute(vae_res["labels"]["kmeans"]),
                "cluster_attribute_summary": label_summary(vae_res["labels"]["kmeans"]),
            },
            "simple_v2": {
                "silhouette": simple_res["kmeans"]["silhouette"],
                "nmi_by_attr": nmi_per_attribute(simple_res["labels"]["kmeans"]),
                "cluster_attribute_summary": label_summary(simple_res["labels"]["kmeans"]),
            },
        },
        "hdbscan": {
            "vae_v3": {
                **{k: v for k, v in vae_res["hdbscan"].items() if k != "umap"},
                "nmi_by_attr": nmi_per_attribute(vae_res["labels"]["hdbscan"]),
                "cluster_attribute_summary": label_summary(vae_res["labels"]["hdbscan"]),
            },
            "simple_v2": {
                **{k: v for k, v in simple_res["hdbscan"].items() if k != "umap"},
                "nmi_by_attr": nmi_per_attribute(simple_res["labels"]["hdbscan"]),
                "cluster_attribute_summary": label_summary(simple_res["labels"]["hdbscan"]),
            },
        },
    }

    report_path = out_root / "latent_space_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Save embeddings for later reuse (optional but handy)
    np.savez_compressed(
        out_root / "latent_space_embeddings.npz",
        idx=idx.astype(np.int64),
        attrs_mean=attrs_mean.astype(np.int64),
        vae_mu=vae["mu"].astype(np.float32),
        vae_logvar=vae["logvar"].astype(np.float32),
        simple_mu=simple["mu"].astype(np.float32),
        simple_logvar=simple["logvar"].astype(np.float32),
    )

    return report_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Latent space visualization for VAE V3 vs Simple V2")
    p.add_argument("--n", type=int, default=2000, help="Number of test samples")
    p.add_argument("--seed", type=int, default=123, help="RNG seed")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for encoder forward")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--embedding", type=str, default="mu", choices=["mu", "z"])
    p.add_argument("--out-dir", type=str, default="artifacts/latent_viz")
    args = p.parse_args()

    cfg = LatentVizConfig(
        seed=int(args.seed),
        n_samples=int(args.n),
        batch_size=int(args.batch_size),
        device=str(args.device),
        embedding=str(args.embedding),
        out_dir=str(args.out_dir),
    )
    report_path = run_latent_space_viz(cfg)
    print(str(report_path))


if __name__ == "__main__":
    main()

