# Paper Claims Ledger

Maps headline claims in `piano-music-generative-models.md` to repo sources. Update this file when numbers or checkpoints change.

## Checkpoints used in E1 / E2

| Paper label | Kaggle dataset slug | Eval notebook |
|-------------|---------------------|---------------|
| Plain Transformer (`plain_v3`) | `plain_v3/ckpt.pt` | `kaggle/notebooks/08_eval_recon_kl.ipynb`, `08_eval_all_combined.ipynb` |
| Simple VAE (`simple_v2`) | `simple_v2/ckpt.pt` | same |
| Hierarchical VAE (`vae_v3`) | `vae_v3/ckpt.pt` | same |

## Dataset splits

| Split | Examples | Source |
|-------|----------|--------|
| Train | 1,430,251 | `artifacts/preprocessed_memmap/train/meta.json` |
| Val | 20,135 | `artifacts/preprocessed_memmap/val/meta.json` |
| Test | 21,105 | `artifacts/preprocessed_memmap/test/meta.json` |

Block size 1024, 8 bars per example, vocab 195. Train/val/test ≈ 80/10/10 after preprocessing (spec §Data).

## E1 — Reconstruction (test, teacher forcing)

Source: `artifacts/eval/original/e1_recon_kl.json` (200 batches, ~1.58M tokens).

| Model | CE | PPL | KL bits/sample |
|-------|-----|-----|----------------|
| plain_v3 | 1.693 | 5.512 | — |
| simple_v2 | 1.695 | 5.519 | 100.2 |
| vae_v3 | 1.678 | 5.427 | 110.0 |

Per-dim KL ≈ 0.78–0.86 bits (110/128).

## E2 — Controllability

Source: `artifacts/eval/original/e2_controllability.json`. Protocol: sweep one attribute 0–7, others at bin 3; N=96 generations per attribute.

| Model | Mean exact-bin | Mean ±1-bin | Mean Pearson r |
|-------|----------------|-------------|----------------|
| plain_v3 | 0.898 | 0.945 | 0.826 |
| simple_v2 | 0.894 | 0.966 | 0.853 |
| vae_v3 | 0.525 | 0.992 | 0.943 |

## E3 — Latent structure (attribute clusters, primary)

Source: `artifacts/eval/latent_vis_vae_attr/`, `latent_vis_simple_attr/`; notebooks `022`, `023`.

Ground truth: 4-D attribute mode (bins 0–7) + IsolationForest + K-means k=4 on 20K test examples.

| Model | kNN purity (PCA) | kNN purity (UMAP) | NMI | ARI | Perturbation preservation |
|-------|------------------|-------------------|-----|-----|---------------------------|
| vae_v3 | 0.374 | 0.382 | 0.206 | 0.112 | 0.813 (520/640) |
| simple_v2 | 0.293 | 0.363 | 0.047 | 0.044 | 0.742 (475/640) |

Random kNN purity baseline: 0.25 (k=4). Plain omitted (no encoder μ).

Figures: `assets/piano-music-generative-models/figures/latent_space_attr_umap_k4.png` (paper, side-by-side UMAP); `artifacts/eval/latent_organization_bert_attr_k4.png` (legacy 3-row).

## Architecture claims

| Claim | Source |
|-------|--------|
| Encoder: full sequence → CLS → μ, log σ² | `ml/src/musicgen/models/vae.py` `BiTransformerEncoder` |
| z_dim = 128, conductor GRU → z_k ∈ R^384 | `vae.py` `Conductor` |
| C_k = [z_k; A_k] ∈ R^512 (3:1 ratio) | `vae.py` `MusicVAE.forward` |
| FiLM: h ← h⊙(1+γ)+β before self-attention | `vae.py` `FiLMBlock.forward` L199–204 |
| KL free bits λ = 1.0 bit | `vae.py` `kl_divergence_free_bits` |
| β cyclical: max 0.1, cycle 10K steps, ramp 5K | `kaggle/notebooks/04_vae_dev.ipynb` |
| ~44.3M params hierarchical | `technical-specification.md` §Encoder/Conductor/Decoder |

## Training defaults

| Setting | Value | Source |
|---------|-------|--------|
| Optimizer | AdamW lr=3e-4, wd=0.01 | notebooks 03–05 |
| Micro-batch | 8, grad accum 2 → effective 16 | notebooks 03–05 |
| AMP | Yes on CUDA | notebooks 03–05 |
| Hardware | Kaggle T4 16GB | spec + notebooks |

## Do not claim (not implemented)

- Self-BLEU diversity metric
- Composer-coloured E3 in main eval
- Adversarial disentanglement (GRL) — future work only
- MCTS decoding — future work only
