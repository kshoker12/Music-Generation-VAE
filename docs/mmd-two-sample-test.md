# Deep-kernel MMD test: which model's samples are closest to real music

Status: run 1 done (2026-10-23, results below); run 2 adds the shared-kernel relative-similarity test. Notebook: `kaggle/notebooks/024_mmd_two_sample.ipynb`.

## Question

Which of the three models (plain decoder, simple VAE, hierarchical VAE) generates 8-bar piano excerpts whose distribution is hardest to tell apart from real music?

Method: the learned deep-kernel MMD test (MMD-D) of Liu et al. (2020), "Learning Deep Kernels for Non-Parametric Two-Sample Tests" (ICML), Algorithm 1. Only MMD-D is run. The baselines from that paper (MMD-O, C2ST, ME, SCF) are not replicated.

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Models | All three: `plain_v3`, `simple_v2`, `vae_v3` | Plain is the baseline for "does a latent make samples more realistic". If it is less diverse, the test measures that; excluding it would assume the answer. |
| Test | MMD-D only | The paper's best method; the question is a ranking, not a method comparison. |
| Sample size | Up to $N = 10000$ excerpts per model (run 1: 5000), floor 1000, set by a timing pilot to fit the 9 h session (see Compute) | Larger $N$ gives a better-trained kernel and a tighter ranking statistic. |
| Splits | 10 random train/test splits of the same pools | As in the paper's CIFAR-10.1 experiment (fixed pools, ten repetitions). |
| Ranking statistic | Run 2: relative-similarity test under one shared kernel (headline). Per-model held-out $\hat{J}$ kept as a secondary score. | Run 1 showed every model rejected and a near-tie in per-model $\hat J$ under separately trained kernels, which are not on one scale. The relative test compares models directly with a p-value, and the B-B term that clustering inflates cancels in it. |
| Features | CLaMP 2 music-encoder embeddings | Large pretrained symbolic-music encoder; the default extractor of Fréchet Music Distance. |
| Sampling | temperature 1.0, top-p 0.9, grammar mask | Same settings as the listening test and demo. Applied identically to all models. |
| Control | Real vs real, piece-disjoint (val vs test), plus a cross-kernel control per model | Checks the false-positive rate, including the inflation from several chunks per piece, under the kernel that scores each model (see Dependence). |

## Data

- Sources: the val and test split memmaps (`preprocessed-memmapval`: 20,135 chunks, 851 pieces; `preprocessed-memmaptest`: 21,105 chunks, 852 pieces). Neither was trained on (val was used only for checkpoint selection), and the 80/10/10 split is at piece level, so the two share no pieces. Test chunks average 502 non-pad tokens (median 477, 90th percentile 817); 0.02% reach 1024.
- From each memmap, draw $N$ chunks evenly spaced through the index. `meta.json` has no per-chunk piece id; chunks are assumed stored piece by piece, so spacing spreads the draw across pieces (about 6 chunks per piece at $N = 5000$). The grid is put in a fixed random order, so the first $N$ rows are a random subset of it for whatever $N$ the pilot picks.
- The two sets:
  - **A** (from val; conditions and control side): each chunk's attribute grid $a_i \in \{0,\dots,7\}^{8 \times 4}$ (bar by polyphony, rhythmic intensity, velocity dynamics, note density) conditions generation.
  - **B** (from test; real reference): compared against every model.
- A and B come from disjoint pieces by construction. A random chunk-level partition of one split would put chunks of the same piece on both sides of the control and make it too easy to pass.

## Generation

For each $i = 1,\dots,N$ and each model $m$: generate one excerpt $y_i^{(m)}$ conditioned on $a_i$.

- Generation starts from a `Bar` token at position 0, as every training chunk does. `BOS` never occurs in the training memmaps (checked: 0 of 20,000 sampled train chunks; all start with `Bar`). The repo's `generate_remi_tokens` starts from `BOS`, which feeds an untrained embedding and shifts every position by one; it also shifts bar conditioning by one, because `compute_bar_indices_prefix` maps `BOS` to bar 0 and the first `Bar` to bar 1 (see Known issue below).
- VAEs draw a fresh $z_p \sim \mathcal{N}(0, I_{128})$ per excerpt. Plain has no latent; its randomness is token sampling only.
- Stop at the 9th `Bar` token or 1024 tokens. Excerpts with fewer than 8 bars are dropped and counted per model.
- **Common size and alignment.** The estimators below are the paper's equal-size U-statistic ($X_i$, $Y_i$ indexed together in $H$). Excerpt $i$ of every model was conditioned on $A_i$, so A and every $Q_m$ stay index-aligned: the kept set is the intersection of valid indices over the three models, of size $n_{\text{all}}$, and B is subsampled to $n_{\text{all}}$ with a fixed seed. All four comparisons use the same B and the same size.
- Result: three generated sets $Q_m$ whose attribute distribution matches real music by construction, up to each model's controllability error.
- Compare $Q_m$ to **B**, not **A**: $y_i^{(m)}$ is paired with $A_i$ through its grid, and the permutation test needs independent samples.
- Seeds fixed per (model, batch); batches run round-robin over models so a hard stop leaves equal counts.

Implementation (`generate_batch` in the notebook's module):

1. **KV cache.** Each step feeds one token and reuses cached keys and values; exact because the decoders use causal attention with absolute positions and per-token FiLM conditioning. The notebook checks this before generating: on 4 real test chunks, cached step-by-step logits must match each model's own full forward (fp32, max abs difference below $10^{-3}$), and the bar-index rule must reproduce the memmap's `bar_indices`.
2. **Batching.** 128 excerpts per batch; finished rows are masked, the batch ends when all rows finish.
3. **Grammar table.** `allowed_next_ids` depends only on the previous token, so it is precomputed as a $195 \times 195$ boolean table from the repo function and indexed on GPU.
4. **Batched top-p.** Same rule as the repo's `sample_next_token` (keep tokens whose preceding cumulative mass is at most 0.9, renormalize).
5. **fp16 autocast** on GPU.

Also recorded: the recomputed attribute grid (repo `compute_raw_attributes_per_bar_from_tokens` + `bin_attributes`; checked on real A chunks against the stored bins) and token length.

## Features

Every real and generated excerpt goes through the same path: REMI tokens, then a MIDI file, then a CLaMP 2 embedding (768-dimensional).

- MIDI: a minimal REMI decoder written with `mido` in the notebook (4/4, 16 positions per bar, 480 ticks per beat, tempo tokens as tempo changes). One function writes both sides, so file format cannot differ between real and generated. A round-trip check asserts every note survives. This replaces miditok, so no tokenizer-version risk.
- Install: `pip install --no-deps frechet-music-distance` plus `mido abctoolkit unidecode` (Kaggle's torch and transformers untouched); full install as fallback. Extractor: `frechet_music_distance.models.CLaMP2Extractor`, called per file with `extract_feature` so order is fixed. It downloads the CLaMP 2 checkpoint (4.45 GB) and `xlm-roberta-base` from Hugging Face on first use; internet must be on.
- Embeddings are saved as `emb_<set>.npy`.

## Test (per comparison, per split)

Comparisons: $Q_{\text{plain}}$ vs B, $Q_{\text{simple}}$ vs B, $Q_{\text{hier}}$ vs B, and A vs B (control).

1. Split each side ($n_{\text{all}}$ points) about 50% test, 10% validation, 40% fit, **by block**: whole blocks of 500 consecutive memmap indices go to test, then validation, then fit, in random block order, so no piece (pieces are stored contiguously) has chunks on both sides of the split. Run 1 split by row, so the kernel trained on other chunks of the test pieces. Each part is trimmed at random to equal size on both sides (the estimators need equal $n$). A and all $Q_m$ use the same assignment (index-aligned, by A's memmap index), B its own; so no model's kernel trains on an excerpt conditioned on an A chunk in the control's test half. The paper's experiments use equal train and test sizes; its Corollary 14 says the power-optimal train share for a fixed pool is smaller (order $N^{3/4}$), but with constants that cannot be computed here, so the paper's practice is kept.
2. Standardize embeddings with mean and std from the pooled fit halves. The scaler is then frozen and applied unchanged to validation, test, and every permutation. Refitting it inside the permutation loop would leak.
3. Kernel (paper eq. 1):
   $$k_\omega(x,y) = \big[(1-\epsilon)\,\kappa(\phi_\omega(x),\phi_\omega(y)) + \epsilon\big]\,q(x,y)$$
   with $\kappa$, $q$ Gaussian. Implemented as in the official code: $k = (1-\epsilon)\exp(-D_\phi/\sigma_0 - D/\sigma) + \epsilon \exp(-D/\sigma)$ with squared distances $D_\phi$, $D$, and $\sigma = s^2$, $\sigma_0 = s_0^2$ learned through their square roots. $\sigma$ starts at $2d$ (official init); $\sigma_0$ starts at the median of $D_\phi$ on the fit set (the official fixed $0.1$ was set for width-$3d$ nets on 10-dimensional data). $\epsilon = \mathrm{sigmoid}(\epsilon')$, so $0 < \epsilon < 1$: the paper's text says log space, but its theory needs $0 \le \epsilon \le 1$ (for $\epsilon > 1$, $1-\epsilon < 0$ and $k_\omega$ may not be a kernel), and the official code does the same (`ep = exp(epsilonOPT) / (1 + exp(epsilonOPT))`, `Deep_Kernel_HDGM.py`). Initialize $\epsilon' = \log 10^{-10}$ as in that code. $\phi_\omega$: the official `ModelLatentF` (four linear layers, softplus between; the paper calls it five-layer counting the input), hidden and output width 256. The paper's \"$3d$ width\" rule would give 2304 for CLaMP 2's 768 dimensions, too wide for this $n$; 256 is wider than any network the paper trained on vector data (Higgs: $d=4$, width 20), so overfitting is checked (step 5).
4. Statistic: with $H_{ij} = k(X_i,X_j) + k(Y_i,Y_j) - k(X_i,Y_j) - k(Y_i,X_j)$,
   $$\widehat{\mathrm{MMD}}_u^2 = \frac{1}{n(n-1)}\sum_{i\neq j} H_{ij}, \qquad \hat\sigma^2_{\lambda} = \frac{4}{n^3}\sum_i\Big(\sum_j H_{ij}\Big)^2 - \frac{4}{n^4}\Big(\sum_{ij} H_{ij}\Big)^2 + \lambda,$$
   $$\hat J_\lambda = \frac{\widehat{\mathrm{MMD}}_u^2}{\hat\sigma_\lambda}, \qquad \hat\sigma_\lambda = \sqrt{\hat\sigma^2_\lambda}, \qquad \lambda = 10^{-8}.$$
   Divide by the standard deviation, not the variance (paper eq. 4; official code: `mmd_std_temp = torch.sqrt(TEMP[1] + 10**(-8))`).
5. Train: maximize $\hat J_\lambda$ on the fit sets, full batch, Adam, lr $10^{-4}$, up to 1000 steps. The paper's vector-data rates range from $10^{-5}$ (HDGM) to $10^{-3}$; on a synthetic pilot at this $n$ (1500 per side, 128 dimensions), $10^{-4}$ beat $5 \times 10^{-4}$ (rejections 4/4 vs 3/4 under a shift, 0/3 vs 0/3 under the null) and the best validation step landed at 30 to 40. Every 10 steps compute $\hat J_\lambda$ on the validation sets; keep the parameters with the best validation $\hat J$. Training stops early once validation $\hat J$ has not improved for 300 steps (it peaks early, so most of the 1000 steps would be wasted; on CPU the full test stage would otherwise take about 3.9 h, on a T4 minutes). This is early stopping, not the paper's procedure (the paper ran a one-time hyperparameter search on a validation set the size of the training set). It is safe for type I error because the test halves never enter it. Validation $\hat J$ at 10% of the split is noisy, so it is used only to pick the checkpoint; ranking uses held-out $\hat J$ only. If validation and held-out $\hat J$ diverge in the smoke run, train a fixed 1000 steps instead. Logged per run: fit, validation, and held-out $\hat J$.
6. Test: on the test halves, compute $\widehat{\mathrm{MMD}}_u^2$ and a permutation p-value with 200 permutations, $p = (1 + \#\{\text{perm} \ge \text{est}\}) / (1 + 200)$, reject at $p \le \alpha = 0.05$. (Algorithm 1 omits the $+1$; it keeps the test exactly valid.) Also compute held-out $\hat J_\lambda$ with the frozen kernel. The p-value is the paper's test; held-out $\hat J$ is an added score, not the paper's reported outcome (it reports rejection rates).

Total runs: 4 comparisons × 10 splits = 40 trained kernels, plus 30 cross-kernel control tests (no training; see Dependence).

## Relative similarity under a shared kernel (run 2)

Per split, after the per-model tests:

1. **Shared kernel.** Pool the generated fit sets: for each fit index $i$, take $Q_m[i]$ for a model $m$ drawn uniformly from the three. Train one deep kernel on B (fit) vs the pooled sample, same settings and early stopping as above.
2. **Pairs.** On the test halves, row $i$ of every generated set (all conditioned on grid $A_i$) is paired with B row $i$, so $(q^{(1)}_i, q^{(2)}_i, b_i)$ are iid triples. With $H^{(m)}_{ij} = k(q^{(m)}_i, q^{(m)}_j) + k(b_i, b_j) - k(q^{(m)}_i, b_j) - k(q^{(m)}_j, b_i)$,
   $$D = \widehat{\mathrm{MMD}}_u^2(Q_1, B) - \widehat{\mathrm{MMD}}_u^2(Q_2, B) = \frac{1}{n(n-1)}\sum_{i \ne j} \big(H^{(1)}_{ij} - H^{(2)}_{ij}\big),$$
   a U-statistic whose $k(b_i, b_j)$ terms cancel. $D < 0$: the first set is closer to real music. This follows the relative-similarity test of Bounliphone et al. (ICLR 2016), with a learned shared kernel in place of a fixed one.
3. **Standard errors.** Linearize: $D - \mathbb{E}D \approx \frac{2}{n}\sum_i e_i$, with $e_i$ the row mean of $H^{(1)} - H^{(2)}$ minus $D$. The rows are not independent: real chunks come several per piece (about 12 at $N = 10000$), and every generated row inherits the piece of its conditioning chunk. Pieces are stored contiguously in the memmaps (pitch-class cosine between chunks: 0.81 at index distance 1, 0.70 at 5, 0.61 at 25, 0.59 at 100, random pairs 0.585). So rows are grouped into blocks of 200 consecutive memmap indices on the conditioning side and on the real side, and $\mathrm{Var}(\sum_i e_i)$ is the two-way cluster-robust (Cameron, Gelbach, Miller) sum over blocks, floored at the larger one-way value. This is computed for block widths 200 and 1000 (long performances can span more than 200 indices); the decision SE is the largest of the two and the naive SE, and $z_{\text{cluster}} = D / \mathrm{se}$, two-sided normal p-value. With three model pairs, a strict claim uses Bonferroni ($p \le 0.05/3$), also reported. The naive (independent-row) $z$ and the design effect (clustered over naive variance) are reported next to it.
4. **Fixed-kernel check.** The same test with a plain Gaussian kernel on the standardized embeddings (median-distance bandwidth, no training), as in Bounliphone et al. The learned kernel is trained on all three models pooled, so a model whose flaws differ from the other two gets less of its attention; agreement between the two kernels guards against that.
5. **Calibration.** `plain2` is a second plain sample on the same grids with different seeds; plain vs plain2 has true $D = 0$ and should reject about 5% of the time (3 or more in 10 fails). This checks the implementation, not robustness to clustering: for two samples of one model the piece-level terms cancel, so it would pass even with naive SEs. Clustering is handled by step 3.
6. Also reported per split: MMD² and $J$ of every set vs B under the shared kernel (one scale), and the A vs B control under the shared kernel.

Pairs: plain-simple, plain-hier, simple-hier, plain-plain2.

Synthetic checks:
- Independent rows (64 dimensions, 600 test points per side, learned kernel, 40 null trials): $z$ mean $-0.02$, sd $0.97$, 0/40 rejections; one set truly closer: 10/10 rejections, correct sign.
- Piece-clustered rows (12 chunks per piece on both sides, two models equally far from B but differing piece by piece, fixed kernel, 300 null trials): naive $z$ sd 2.96 to 3.01, 54 to 55% false rejections; clustered $z$ (single width) sd 1.07, 6% rejections, design effect about 8; final decision rule (two widths, naive floor) sd 1.09, 6.3% rejections. One set truly closer: 30/30 rejections, correct sign.

Reading: a pair is resolved if it rejects (clustered; Bonferroni for a strict claim) in most splits, with a consistent sign, under both kernels, and the calibration passes. "Most of 10 splits" is a reading rule, not a formal test: the splits reuse one sample. Report $D$, $z_{\text{cluster}}$, and rejections; do not report per-model $\hat J$ gaps as a ranking. Effective sample size is closer to the number of pieces (about 850 per split) than to $N$, so going from 5000 to 10000 adds less than $\sqrt 2$ in precision.

## Ranking and reporting

Per comparison, over 10 splits:

- held-out $\hat J$: mean and sd. **Rank models by mean held-out $\hat J$; lower means this test has less asymptotic power against that model, i.e. its samples are harder to tell from real music with MMD-D.** Each model gets its own kernel, trained to make its own pair as separable as possible, so the score is detectability under a per-model best-found kernel. It is not a shared distance: $J(ck) = J(k)$ makes $\hat J$ scale-free for one kernel, but does not put three different kernels on one scale.
- held-out $\widehat{\mathrm{MMD}}_u^2$: mean and sd.
- rejections out of 10 and median p-value.

Control expectation: about 0.5 rejections in 10 at $\alpha = 0.05$. The 10 splits reuse one pool, so the count is not a clean binomial and 1 or 2 is not by itself a failure. Treat 3 or more as a failed control. Control held-out $\widehat{\mathrm{MMD}}_u^2$ fluctuates around 0 and can be negative, so control $\hat J$ is noisy around 0; its spread is the noise floor for reading gaps between models.

### Dependence

Real chunks come several per piece; generated excerpts are independent draws. Equal marginals with different dependence can still produce rejections, and permuting chunk labels does not recreate piece clusters, so the permutation null is too narrow when a side is clustered. The diagnostics below cannot separate this from a musical difference.

What the piece-disjoint control shows. Clustering inflates each within-sample term of $\widehat{\mathrm{MMD}}_u^2 = \bar K_{XX} + \bar K_{YY} - 2\bar K_{XY}$ (same-piece pairs are more similar than independent pairs), and both enter with a plus sign. In val vs test the cross term is not inflated (no shared pieces), so clustering biases the control upward by $\delta_{\text{val}} + \delta_{\text{test}}$, while a random permutation spreads same-piece pairs over all three terms and stays near 0. In $Q_m$ vs B the bias is $\delta_{\text{test}} + \delta_{Q_m}$, where $\delta_{Q_m}$ comes only through attribute grids shared by excerpts conditioned on chunks of the same val piece. So a clean control is evidence against clustering driving model rejections, under two assumptions: $\delta_{Q_m} \le \delta_{\text{val}}$, and the $\delta$s are measured under the same kernel. The second fails as stated, because each comparison trains its own kernel.

**Cross-kernel control.** For each model's frozen kernel $k_m$, also run the permutation test on the control pair's test halves (A vs B), which $k_m$ never trained on. A rejection rate near $\alpha$ there means $k_m$ does not pick up the piece clustering in B. That is the direct check for the kernel actually used to score model $m$. Cost: 3 extra permutation tests per split.

Reading: if the control and all three cross-kernel controls pass, report rejections and held-out $\hat J$. If the plain control fails, or a cross-kernel control for model $m$ fails, report model $m$ (or all models) only as held-out $\hat J$ relative to the control's. In every case, show the attribute-shift and length diagnostics next to the ranking.

Diagnostics per model:

- Attribute shift: mean of (recomputed bin minus requested bin) per attribute. The hierarchical VAE is known to undershoot by about one bin (exact-bin accuracy 0.525), so a high $\hat J$ for it may partly reflect that offset.
- Token-length distribution vs B; count of dropped excerpts.
- Fit, validation, and held-out $\hat J$ per run. Fit far above held-out means $\phi$ overfit; check this in the smoke run first.

## Compute

Kaggle limits: 9 h per GPU session (as observed on this account), about 30 GPU hours per week. Use **GPU T4 x2** (one GPU is used; with the KV cache, generation is no longer the bottleneck). Run with "Save Version" (background, browser can close).

Budget, enforced in the notebook (seconds since the first cell):

| Stage | Budget |
|---|---|
| Install, CLaMP 2 download, load, checks | about 15 min |
| Timing pilot (first batch per model + 24 embeddings) | about 5 min |
| Generation (4 sets incl. plain2) | 2.5 h (hard stop at 3.1 h) |
| MIDI + CLaMP 2 embedding of 6 sets | 2.5 h |
| Tests (40 per-model kernels, 10 shared kernels, controls) | 3 h reserved |
| Deadline | 8.25 h; any stage past it stops and results are written |

$N$ rule: after the pilot, $N = \min(5000,\ \text{gen budget}/\text{gen rate},\ \text{emb budget}/\text{emb rate},\ \text{time left before the test reserve}/(\text{gen}+\text{emb rate}))$, the same $N$ for all models, with a warning below the floor of 1000. During generation, a batch is skipped if it would push generation plus embedding of everything generated into the test reserve. `results.json` is rewritten after every split, so a run stopped by the deadline still has results.

## Notebook layout (`024_mmd_two_sample.ipynb`)

1. Mode (`MODE = 'dry' | 'full'`; outside Kaggle `--dry-run` or `DRY_RUN=1`), imports, timer.
2. Config; dry-run overrides.
3. Install `frechet-music-distance` and `mido`.
4. Locate datasets (`/kaggle/input/datasets/karandeepshoker/...`, as notebooks 04/05/08).
5. Inlined module: model, grammar and attribute code verbatim from notebook 08, plus the new code (generation, MIDI, MMD).
6. Tokenizer meta, thresholds, grammar table.
7. Load the three checkpoints.
8. Open memmaps; KV-cache equivalence check.
9. Draw A (val) and B (test).
10. CLaMP 2 load, MIDI round-trip check, embedding pilot.
11. Generation pilot; choose $N$; print the projected finish time.
12. Generate (tqdm with ETA).
13. Valid set, common size, attribute shift, lengths; save tokens.
14. MIDI + CLaMP 2 embeddings (tqdm with ETA).
15. MMD tests (tqdm with ETA).
16. Summary: ranking, controls, diagnostics.

Dry run: $N = 64$, batch 32, 1 split, 50 training steps (validation every 5), 50 permutations. About 15 min on Kaggle, most of it the CLaMP 2 download. It prints the measured seconds per excerpt and per file and the $N$ the full run would choose.

## Kaggle inputs

`plain_v3/ckpt.pt`, `simple_v2/ckpt.pt`, `vae_v3/ckpt.pt`, `preprocessed-memmapval/`, `preprocessed-memmaptest/`, `misc/` (`tokenizer_meta.json`, `quantiles.json`). Internet on.

## Outputs (`/kaggle/working/artifacts/mmd_test/`)

- `results.json`: config, chosen $N$, drops, checks, timings, attribute and length diagnostics, per-split statistics, summary.
- `summary.md`: the ranking table, controls, diagnostics.
- `tokens.npz`: tokens of A, B and each model (kept rows), requested attributes, memmap indices.
- `emb_<set>.npy` for A, B, plain, simple, hier.
- `midi_samples/<set>/`: 10 MIDIs per set for listening.

## Risks

- CLaMP 2 install or weight download fails on Kaggle. Found in the dry run; fallback is the MidiBERT-Piano embedder from notebook 011.
- Chunks from the same piece are correlated. The piece-disjoint and cross-kernel controls check for the effect but do not rule it out (see Dependence).
- Each $Q_m$ is conditioned on val grids and compared with test music. Val and test are random piece-level splits of the same corpus, so their attribute distributions match up to sampling noise; the control measures that noise.
- The ranking is in CLaMP 2's feature space: "closest to real music as CLaMP 2 represents it".
- Top-p 0.9 trims rare tokens, so part of any detected difference is the sampler. It is the same for all models, so the ranking stays fair.

## Known issue in the repo's generation path

`ml/src/musicgen/inference/generate.py` (and the copy inlined in notebook 08) starts generation with `BOS`, which never occurs in training data, and `compute_bar_indices_prefix` then assigns the first bar index 1: bars 1 to 7 are conditioned on attribute rows 1 to 7 (and, for the hierarchical VAE, Conductor steps 1 to 7), and bar 8 reuses row 7. Row 0 is never used. Constant-attribute sweeps are unaffected in $A_k$, but bar-varying prompts (the Crescendo listening prompt) and the Conductor's $z_k$ are shifted by one bar. This notebook starts from `Bar` and does not use that path. Not fixed in the repo.

## Run 1 results (2026-10-23, N = 5000, 1.06 h on a T4)

Measured: 0.056 s per excerpt per model, 0.083 s per file for MIDI + CLaMP 2. $n_{\text{all}} = 4926$ (dropped: plain 43, simple 12, hier 27); fit/val/test 1970/493/2463 per side.

| vs B | held-out $\hat J$ | held-out MMD² | rejections | fit / val $\hat J$ |
|---|---|---|---|---|
| plain | 0.870 ± 0.028 | 0.0897 | 10/10 | 1.45 / 0.90 |
| hier | 0.886 ± 0.022 | 0.1017 | 10/10 | 1.29 / 0.90 |
| simple | 0.911 ± 0.026 | 0.0909 | 10/10 | 1.47 / 0.91 |
| A (control) | 0.066 ± 0.008 | 0.0004 | 10/10 | 0.58 / 0.07 |

Cross-kernel controls: 10/10 for every model. Attribute shifts at most 0.16 bins for every model.

Reading: every model is clearly distinguishable from real music; the generated-vs-real effect is about 13 times the control's $\hat J$ and about 250 times its MMD². The control failed: two disjoint sets of real pieces (val, test) are distinguishable at this $n$, from piece clustering and from the two splits being different finite sets of about 850 pieces. Per the reading rule, only $\hat J$ relative to the control is reportable (plain 13.1, hier 13.4, simple 13.8 times). The models are a tie: gaps are about one sd (which understates uncertainty, since splits share samples), and the MMD² order differs from the $\hat J$ order. Run 2 adds the relative-similarity test above to compare the models directly.

## What this can and cannot claim

Can claim: at $n_{\text{all}}$ samples per side, under CLaMP 2 features, model X's excerpts are harder to distinguish from held-out real excerpts than model Y's, with the given $\hat J$ and rejection counts, and the control behaves as expected.

Cannot claim: perceptual quality. A significance statement for "X is closer than Y" comes only from the run 2 relative-similarity test, and only if its calibration passes.

## References

- Liu, Xu, Lu, Zhang, Gretton, Sutherland. Learning Deep Kernels for Non-Parametric Two-Sample Tests. ICML 2020. arXiv:2002.09116.
- Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation. arXiv:2412.07948. Library: https://github.com/jryban/frechet-music-distance
- CLaMP 2 (Wu et al., 2024): the symbolic music encoder used by the library above.
- Bounliphone, Belilovsky, Blaschko, Antonoglou, Gretton. A Test of Relative Similarity for Model Selection in Generative Models. ICLR 2016. arXiv:1511.04581 (verify before citing).
