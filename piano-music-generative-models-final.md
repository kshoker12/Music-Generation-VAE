# 🎵 Piano Music Generative Models
**Exploring Generative Models for Controllable Symbolic Piano Music Generation (Plain Transformer, VAE, & Hierarchical VAE)**

**Karandeep Shoker**  
[LinkedIn](https://www.linkedin.com/in/kshoker12/)

### Quick Access
- **[MusicGen Dashboard](https://kshoker12.github.io/Music-Generation-VAE/)** — Interactive UI to generate 8-bar Piano excerpts with Controllable Attributes  
- **[GitHub Repo](https://github.com/kshoker12/Music-Generation-VAE)** — Kaggle Training Notebooks, Model Definitions, and Inference Logic

<!--
[NARRATIVE ALIGNMENT - OVERALL TONE]
- Tone: Precise, intuitive, no fluff. Direct to the point.
- Strategy: "Math -> Meaning". Every statistical equation gets an intuitive, real-world translation.
- The Flex: Highlighting the engineering required to train a complex hierarchical model on a single 16GB Kaggle T4 GPU[cite: 1, 8].
-->

## Abstract

<!-- 
[NARRATIVE ALIGNMENT] 
- Hook: Symbolic music generation lacks long-range structure; autoregressive models wander aimlessly.
- Solution: Compare Plain Decoder, Simple VAE, and Hierarchical VAE[cite: 1].
- Core Technical Tension: Overcoming posterior collapse using cyclical beta-annealing, KL free bits, and FiLM[cite: 1, 8].
- The Engineering Flex: Achieved on strict hardware constraints (Kaggle T4) proving efficient system design[cite: 8].
--> 

## 1 Introduction

### 1.1 Motivation

<!-- 
[NARRATIVE ALIGNMENT] 
- Frame the project: Originating from CPSC 440, evolving into an exploration of sequence modeling.
- The "Resource Crucible": Emphasize that the 16GB Kaggle T4 limit wasn't a roadblock, but a catalyst for elegant architectural and data-pipeline design[cite: 8]. Focus on engineering efficiency over raw compute.
-->
Generative AI models have become a game-changer in recent years. Large Language Models (LLMs) generate textual responses, Image Generation Models turn prompts into highly detailed visuals, and Coding Agents produce executable software, enabling developers to reach new levels of productivity. Originating as my CPSC 440 project at the University of British Columbia (UBC), this work explores increasingly complex generative sequence modelling architectures, using classical piano music as an empirical testbed.

Music is uniquely challenging compared to text because it is bound by strict structural rules (like measures and beats) and relies on overarching themes that must evolve naturally over time. The central technical tension in modeling these sequences is **posterior collapse**. High-capacity autoregressive transformers are powerful, but short-sighted improvisers; they easily fit local next-token statistics while ignoring high-level global instructions (latent variables). To systematically explore and mitigate this issue, this paper evaluates three architectures:

1. **Plain Transformer Decoder:** A baseline conditioned strictly on explicit user-defined attributes.  
2. **Simple Variational Autoencoder (VAE):** Introduces a continuous global latent space to capture overarching musical characteristics.  
3. **Hierarchical VAE:** Introduces a recurrent temporal network to deterministically unroll global latents into bar-level instructions.

Furthermore, this research serves as a blueprint for efficient ML systems design. The entire training pipeline was strictly constrained to a 16GB Kaggle T4x2 GPU. Rather than relying on raw compute, this hardware ceiling is treated as a crucible that forces highly optimized, production-ready engineering, leveraging mixed-precision training, offline memory-mapped data streaming, and cyclical hyperparameter annealing to stabilize the learning process.

### 1.2 Musical Background

<!--
[NARRATIVE ALIGNMENT]
- Intuition first: Explain bars, 4/4 time, and the REMI token format simply. 
- Controllable Attributes: Define Polyphony Rate, Rhythmic Intensity, Velocity Dynamics, and Note Density. Frame these as the "control knobs" we want to hand to the user[cite: 8].
-->

This project generates symbolic classical piano music. To understand how the models process, structure, and control the music, it's important to define a few foundational musical concepts.

Bars and Time Signatures: Music is modelled in bars because they are the natural clock of Western piano music. Intuitively, if a song is a paragraph, a bar is a single sentence, it organizes individual notes into a cohesive musical idea. By strictly filtering our dataset to a \\(4/4\\) time signature, it's guaranteed that every bar contains exactly four beats (\\(16\\) discrete position slots). This enforces mathematical uniformity across the dataset, guaranteeing predictable sequence lengths and providing a stable temporal grid for our models to inject conditioning vectors. The models process and generate music in fixed \\(8\\)-bar excerpts, capturing a complete structural arc while keeping sequence lengths manageable for limited compute.

**Controllable Attributes:** To allow for user-directed generation, I utilize four mathematically measurable, bar-wise control signals, quantized into eight ordinal bins (\\(0-7\\)):

- **Polyphony Rate:** Controls the number of piano keys pressed simultaneously. Low polyphony leads to single-note melodies, and high polyphony leads to thick, multi-note chords.  
- **Rhythmic Intensity:** Controls how continuously the piano is played. Low rhythmic intensity leads to music with frequent pauses, and high rhythmic intensity leads to a constant, unbroken rhythm.  
- **Velocity Dynamics:** Controls how much loudness varies within a bar (measured by the standard deviation of velocity values). Low velocity leads to a more constant music volume, and high velocity leads to high variation in music volume. 
- **Note Density:** Controls the total number of notes played in a single bar. Low note density leads to sparse music with few notes, and high note density leads to fast music packed with notes.  

### 1.3 Dataset, Preprocessing, & Tokenization

<!-- 
[NARRATIVE ALIGNMENT]
- Data sources: MAESTRO[cite: 7] & GiantMIDI[cite: 8]. 
- Engineering Flex: Highlight the offline preprocessing. Mention the `uint16`/`uint8` memory-mapped arrays (memmap)[cite: 8]. Explain *why*: to bypass CPU RAM bottlenecks and keep the GPU saturated. This proves production-level system design.
-->

The models are trained on a curated corpus of classical piano performances from the MAESTRO and GiantMIDI-Piano datasets. Pieces are partitioned at the piece level into an \\(80/10/10\\) train/validation/test split. After filtering to \\(4/4\\) time signatures and applying \\(12\\)-key offline transposition on the training set, preprocessing yields \\(\\approx 1.43\\) million \\(8\\)-bar training chunks (\\(\\approx 1.46\\) billion tokens at \\(1024\\) tokens per chunk), alongside \\(20{,}135\\) validation and \\(21{,}105\\) test chunks, the latter reflecting piece-level splits without the same key augmentation.

To maximize the quality of the training signal, the raw MIDI undergoes a strict preprocessing pipeline. Sequences are trimmed of leading silence, sliced into fixed \\(8\\)-bar windows, and filtered to remove musically sparse or inactive chunks. Additionally, a quantization process occurs where the four controllable attributes from Section \\(1.2\\) are mathematically calculated for every bar and mapped into eight discrete bins (\\(0-7\\)) based on distributions derived strictly from the training set.

**Systems Optimization (Memmaps):** To bypass severe CPU RAM bottlenecks and keep the GPU fully saturated, monolithic PyTorch tensors are avoided. Instead, the final tokenized sequences and attribute labels are stored as memory-mapped binary arrays (`uint16` and `uint8` memmaps). Attributes are pre-binned offline; the DataLoader streams token slices directly from disk, with autoregressive targets derived at batch time (Section \\(2.4\\)).

**Tokenization:** The music is tokenized into REMI (Revamped MIDI) using the `miditok` library. Utilizing a highly optimized \\(195\\)-token vocabulary, REMI translates the notes into discrete sequences capped at \\(1024\\) tokens. Crucially, the vocabulary includes an explicit `Bar` token to mark bar boundaries. During autoregressive generation, encountering this hard boundary acts as a programmatic trigger, commanding the decoder to instantly switch its conditioning to the next bar's controllable attributes, allowing the music to evolve dynamically with the progression of bars. The complete end-to-end preprocessing pipeline, memory-mapping optimizations, and full REMI token vocabulary are provided in Appendix A. 

TODO: Fix appendix stuff once we get to appendix

### 1.4 Statistical Modelling Approach

<!--
[NARRATIVE ALIGNMENT]
- Introduce the "Composer, Conductor, Musician" metaphor here to ground the math.
- Plain Decoder (The Musician alone): Generates $X_{1:T}$ autoregressively conditioned only on attributes $A_{1:K}$.
- Simple VAE (The Composer & Musician): Introduces global latent $Z_p$ (The Composer's general vibe).
- Hierarchical VAE (Composer, Conductor, Musician): The Conductor unrolls the global $Z_p$ into bar-level instructions $Z_{1:K}$. This bridges abstract style with local coherence.
-->
The architectural progression systematically explores how to enforce global structure onto a localized generator.
- **The Plain Decoder (The Musician Alone):** Without latent variables, the model relies purely on local sequence history and explicit controllable attributes (polyphony, rhythmic intensity, velocity dynamics, note density). I introduce the autoregressive conditional likelihood \\(P_\theta(X|A) = \prod_{t=1}^T P_\theta(X_t | X_{<t}, C_{k(t)})\\). To achieve this, the four binned attributes are extracted for bar \\(k\\), embedded, and linearly projected to form a \\(512\\)-dimensional condition vector: \\(C_k = \text{Linear}(A_k) \in \mathbb{R}^{512}\\). Musically, this results in compositions that rigidly obey local attribute dials (like playing fast and loud when told to), but tend to wander aimlessly without a cohesive, long-term melody.
- **The Simple VAE (The Composer & Musician):** I introduce an approximate posterior \\(Q_\phi(Z_p|X)\\) to infer a single, continuous global latent \\(Z_p \sim \mathcal{N}(0, I)\\) (The Composer). This condenses the entire \\(8\\)-bar chunk's overarching musical identity into a \\(128\\)-dimensional bottleneck. \\(Z_p\\) is statically projected to \\(Z_g \in \mathbb{R}^{384}\\) and concatenated with attributes to form \\(C_k = [Z_g; A_k] \in \mathbb{R}^{512}\\). However, projecting a global latent as a static, uniform condition across all \\(8\\) bars makes it difficult to cleanly separate shifting local stylistic features from the global theme. Musically, this results in compositions that establish a consistent overarching mood or theme, but the local details often blur together.
- **The Hierarchical VAE (Composer, Conductor, Musician):** To bridge the compact global representation \\(Z_p\\) with the precise, bar-by-bar sequence variations required by the decoder, I introduce a deterministic \\(2\\)-layer GRU (The Conductor). It initializes from \\(Z_p\\) and unrolls learned bar-index embeddings into a sequence of bar-specific condition vectors \\(\\{Z_k\\}&#95;{k=1}^{8} \in \mathbb{R}^{384}\\). The condition becomes \\(C_k = [Z_k; A_k] \in \mathbb{R}^{512}\\). The decoder models \\(P_\theta(X|Z_p, A) = \prod_{t=1}^T P_\theta(X_t | X_{<t}, C_{k(t)})\\), ensuring long-range coherence without sacrificing local harmonic guidance. Musically, this results in highly dynamic compositions that maintain a clear, evolving long-term structure while simultaneously executing precise, bar-by-bar stylistic shifts.

<figure id="fig-plates-sec-1-4">
  <div class="plates-grid">
    <div class="plate-panel">
      <img src="assets/piano-music-generative-models/figures/plates-plain.svg" alt="Plate diagram: Plain decoder conditioned on per-bar attributes via C_k = Linear(A_k)" />
      <div class="plate-subcaption"><span>Left:</span> Plain Decoder</div>
    </div>
    <div class="plate-panel">
      <img src="assets/piano-music-generative-models/figures/plates-simple-vae.svg" alt="Plate diagram: Simple VAE with global latent Z_p projected to Z_g and concatenated with attributes" />
      <div class="plate-subcaption"><span>Middle:</span> Simple VAE</div>
    </div>
    <div class="plate-panel">
      <img src="assets/piano-music-generative-models/figures/plates-hier-vae.svg" alt="Plate diagram: Hierarchical VAE where Z_p is unrolled into bar-level Z_k for the decoder" />
      <div class="plate-subcaption"><span>Right:</span> Hierarchical VAE</div>
    </div>
  </div>
  <figcaption class="plates-caption">
    Graphical model (plate) view of the three architectures in Section 1.4. Shaded nodes are observed token variables; unshaded circles are latent or deterministic latents; double-bordered circles denote deterministic transformations. <strong>Left (Plain Decoder):</strong> the token sequence \(X = (X_1, \ldots, X_T)\) is generated autoregressively with per-bar attributes \(A_{1:K}\) mapped to \(C_k = \text{Linear}(A_k)\). <strong>Middle (Simple VAE):</strong> a global latent \(Z_p \sim \mathcal{N}(0, I)\) is inferred and projected to \(Z_g\); decoding uses \(C_k = [Z_g; A_k]\). <strong>Right (Hierarchical VAE):</strong> \(Z_p\) is deterministically unrolled by the Conductor into \(\{Z_k\}_{k=1}^{8}\); decoding uses \(C_k = [Z_k; A_k]\). In all models, \(k(t)\) indexes the bar of token \(t\).
  </figcaption>
</figure>

### 1.5 Related Work

<!--
[NARRATIVE ALIGNMENT]
- MusicVAE[cite: 2]: Acknowledge the hierarchical decoder to fight collapse, but note our shift to Transformers and explicit controls[cite: 1].
- MuseMorphose[cite: 5]: Acknowledge segment-level conditioning, but explicitly contrast their "in-attention" addition with our robust FiLM (Feature-wise Linear Modulation) multiplicative conditioning[cite: 1, 4].
- EmoMusicTV[cite: 3]: Connect to their emphasis on cross-bar coherence via hierarchical latents.
-->
This project builds upon a rich lineage of sequence modeling techniques, specifically targeting the intersection of latent variable models and symbolic music generation.

**MusicVAE (Hierarchical VAEs and Posterior Collapse):** Roberts et al. introduced MusicVAE [2], demonstrating that using a hierarchical recurrent decoder is highly effective at mitigating posterior collapse when generating long sequences. While I adopted their foundational principle of hierarchical decoding to maintain long-range structure, my architecture shifts the generative backbone from LSTMs to a causal Transformer to leverage superior self-attention mechanisms, and further augments the generation process with explicit user-defined controls [1].

**MuseMorphose (Segment-Level Conditioning):** My approach to bar-level control is closely related to MuseMorphose [5], a Transformer-based VAE that elegantly handles segment-level conditioning for musical attributes (like rhythmic intensity and polyphony). However, a critical architectural difference lies in the conditioning mechanism. While MuseMorphose injects its conditions into each self-attention layer of the transformer using an "in-attention" additive mechanism, my architecture utilizes Feature-wise Linear Modulation (FiLM) [1, 4]. FiLM is leveraged as a stronger multiplicative enforcer to strictly bind the decoder to the latent space, the mathematical mechanics of which are detailed in Section 2.1.

**EmoMusicTV (Connecting Musical Ideas Across Bars):** Finally, my use of a deterministic recurrent network (The Conductor) to translate global latents into sequential, bar-level instructions shares structural DNA with models like EmoMusicTV [3]. Their work heavily emphasized the use of hierarchical latents to maintain a cohesive, evolving musical narrative across multiple bars. This validates my approach to bridging abstract global themes (The Composer) with localized, bar-by-bar sequence execution.

### 1.6 Paper Overview

- Section 2 (Architecture) details the specific mechanisms of the generative models. The integration of FiLM is formally defined, followed by the specific structural breakdowns of the Plain Decoder, Simple VAE, and Hierarchical VAE. I also detail the modifications to the standard ELBO loss objective function (cyclical \\(\beta\\)-annealing and KL Free Bits) used to combat posterior collapse, and the resource-constrained training pipeline (Section \\(2.4\\)).

- Section 3 (Results) evaluates the models quantitatively across three dimensions: reconstruction fidelity (KL free bits and perplexity), attribute controllability (exact-bin accuracy vs. monotonic Pearson correlation), and latent space structure (UMAP disentanglement visualizations). Furthermore, I conduct a qualitative analysis (Section 3.4) of generated samples. 

- Section 4 (Conclusion) summarizes the findings, addresses the limitations of the fixed time-signature scope, and proposes future directions for latent disentanglement and generation decoding.

TODO: Potential reference to appendix for detailed breakdown of implementation details.
## 2 Architecture

### 2.1 Plain Transformer Decoder 
<!-- Math -> Meaning: Autoregressive generation $P_\theta(x_t | x_{<t}, A)$. Architecture breakdown. -->

The baseline architecture functions as the "Session Musician." It is a non-latent autoregressive model highly capable of generating token sequences that adhere perfectly to the structural rules provided by its conditioning. Given explicit, localized instructions, such as polyphony, rhythmic intensity, velocity dynamics, and note density, the Musician executes them with precision, processing the sheet music exactly one measure at a time.

**The Musician: Autoregressive Sequence Generation**

The core generative engine is a causal Transformer decoder [Vaswani et al., 2017] (\\(\approx 22.8\text{M}\\) parameters, \\(N=6\\) layers, \\(n_{\mathrm{embed}}=512\\), and a maximum sequence block size of \\(T=1024\\) tokens). To condition the generation process, the model relies strictly on explicit controllable attributes. For a given bar \\(k\\), four attribute variables (introduced in Section 1.2) quantized into 8 bins (\\(0\\)–\\(7\\)) are utilized: polyphony (\\(a_k^{\mathrm{poly}}\\)), rhythmic intensity (\\(a_k^{\mathrm{rhythm}}\\)), velocity dynamics (\\(a_k^{\mathrm{vel}}\\)), and note density (\\(a_k^{\mathrm{dens}}\\)). During training, these variables are computed directly from the raw MIDI data, whereas during inference, they serve as user-defined inputs.

To construct the unified attribute vector \\(A_k\\), these four quantized discrete variables are passed through learned embedding tables. Since each of the 8 bins possesses its own learned parameter weights, this operation maps each discrete integer into a continuous 32-dimensional vector. These embedded representations are then concatenated to form a 128-dimensional vector:

$$
A_k = [\mathrm{Embed}(a_k^{\mathrm{poly}}); \mathrm{Embed}(a_k^{\mathrm{rhythm}}); \mathrm{Embed}(a_k^{\mathrm{vel}}); \mathrm{Embed}(a_k^{\mathrm{dens}})] \in \mathbb{R}^{128}
$$

This attribute vector \\(A_k\\) is linearly projected to match the network's embedding dimension, forming the final condition vector \\(C_k\\):

$$
C_k = \mathrm{Linear}(A_k) \in \mathbb{R}^{512}
$$

To guarantee the decoder respects these attributes, \\(C_k\\) is dynamically injected into the generation of every token within bar \\(k\\) via a Feature-wise Linear Modulation (FiLM) mechanism (detailed below). With this conditioning in place, the model estimates the probability of the discrete musical sequence \\(X = (X_1, \ldots, X_T)\\), representing a flattened stream of MIDI events (e.g. notes, velocities, and time shifts), by factorizing the joint probability autoregressively:

$$
P_\theta(X \mid A) = \prod_{t=1}^{T} P_\theta(X_t \mid X_{1:t-1}, C_{k(t)})
$$

where \\(k(t)\\) denotes the specific bar index corresponding to timestep \\(t\\). During inference, the decoder utilizes this estimated conditional probability distribution \\(P_\theta(X \mid A)\\) to autoregressively generate the piece step-by-step. At each timestep, the probability distribution is evaluated over all possible musical tokens, paired with programmatic masking to filter out illegal token transitions (detailed in Appendix X) and top-\\(p\\) sampling to select the next note. Crucially, the vocabulary includes an explicit `Bar` token. When this token is generated, it serves as a hard structural boundary, dynamically switching the active FiLM condition vector to the next bar's attributes \\(C_{k+1}\\).

Since this architecture lacks a latent space, it is trained entirely using standard cross-entropy loss (\\(\mathcal{L}&#95;{\mathrm{CE}}\\)) with teacher-forcing, optimizing the network to maximize the log-likelihood of the ground truth tokens:

$$
\mathcal{L}&#95;{\mathrm{CE}} = -\mathbb{E}\left[ \sum_{t=1}^{T} \log P_\theta(X_t \mid X_{1:t-1}, C_{k(t)}) \right]
$$

**Feature-wise Linear Modulation (FiLM)**

A central challenge in conditional sequence modeling is ensuring that the high-capacity causal decoder actively utilizes the condition vector \\(C_k\\), rather than lazily defaulting to local token history. To enforce this, the architecture injects \\(C_k\\) directly into every Transformer layer using Feature-wise Linear Modulation (FiLM) [Perez et al., 2018].

<figure id="fig-film-pipeline">
  <img src="assets/piano-music-generative-models/figures/film-decoder-layers.svg" alt="FiLM conditioning pipeline: per-bar C_k broadcast into FiLM blocks in each decoder layer" />
  <figcaption>
    The FiLM conditioning pipeline. The per-bar condition vector \(C_k\) is globally broadcast and injected into the affine FiLM transformation block within every Transformer layer, modulating the normalized hidden states prior to self-attention.
  </figcaption>
</figure>

Given an intermediate hidden state \\(h_l^t \in \mathbb{R}^{d}\\) at layer \\(l\\) and timestep \\(t\\), the FiLM operation applies an element-wise affine transformation to the normalized activations:

$$
\mathrm{FiLM}(h_l^{t} \mid C_k) = (1 + \gamma_l(C_k)) \odot h_l^{t} + \beta_l(C_k)
$$

where \\(\gamma_l(C_k) = \mathrm{Linear}&#95;\gamma(C_k)\\) and \\(\beta_l(C_k) = \mathrm{Linear}&#95;\beta(C_k)\\) are learned linear projections mapping \\(C_k\\) to scale and shift vectors, and \\(\odot\\) denotes the Hadamard product. The \\((1 + \gamma_l)\\) form recovers the identity map when \\(\gamma_l \approx 0\\), while still permitting strong multiplicative modulation of normalized activations.

The structural necessity of this multiplicative scaling is best understood when contrasted with purely additive conditioning. Models such as MuseMorphose [Wu and Yang, 2021] utilize an "in-attention" additive mechanism, mathematically adding a projected condition directly to the hidden states (\\(\tilde{h}_l^t = h_l^t + W_C C_k\\)). While this shifts the mean of the activations, a purely additive signal is highly susceptible to being washed out by subsequent normalization layers, allowing the powerful decoder to easily bypass the conditioning.

FiLM circumvents this limitation by element-wise scaling of the hidden representations. Acting as a strict gating mechanism, it can amplify or mute specific feature maps based on \\(C_k\\). This operation inextricably links the generated sequence to the conditioning vector at every depth of the network, mathematically forcing the decoder to respect the controllable attributes. 

While this model achieves high accuracy in following the conditioning \\(C_k\\), it is fundamentally short-sighted. It adheres strictly to the explicit controllable attributes, frequently resulting in musical outputs that wander aimlessly without a cohesive, long-term melody.

**Architectural Overview**
- Components: Causal Transformer Decoder (\\(\approx 22.8\text{M}\\))

- Total Parameters: \\(\approx 22.8\text{M}\\)

- Implementation: [plain_transformer.py](https://github.com/kshoker12/Music-Generation-VAE/blob/main/ml/src/musicgen/models/plain_transformer.py)

<figure id="fig-plain-training" class="arch-diagram">
  <img src="music_vae_content/final_diagrams/plain_train_final.svg" width="537" height="316" alt="Plain Transformer Decoder training architecture: attribute embeddings to C_k, FiLM injection, autoregressive decoding, and cross-entropy loss" />
  <figcaption>
    Plain Transformer Decoder training architecture. Ground-truth tokens \(X\) and past tokens \(X_{\lt t}\) feed a causal Transformer decoder; per-bar attributes are embedded into \(A_k\), projected to \(C_k\), and injected via FiLM at every layer. The model is trained with teacher-forcing cross-entropy against the ground-truth sequence.
  </figcaption>
</figure>

<figure id="fig-plain-inference" class="arch-diagram">
  <img src="music_vae_content/final_diagrams/plain_inference_final.svg" width="378" height="316" alt="Plain Transformer Decoder inference architecture: user-defined attributes to C_k, FiLM injection, autoregressive sampling with past token history, and generated sequence X prime" />
  <figcaption>
    Plain Transformer Decoder inference architecture. User-defined bar attributes are embedded into \(A_k\), projected to \(C_k = \mathrm{Linear}(A_k)\), and injected via FiLM at every decoder layer. The causal decoder autoregressively generates \(X' = (X'_1, \ldots, X'_T)\), conditioning each step on past generated tokens \(X'_{\lt t}\) and bar control \(C_{k(t)}\).
  </figcaption>
</figure>

### 2.2 Simple Variational Auto-Encoder 
<!-- Math -> Meaning: Introduce $Q_\phi(z_p|x)$. Explain the reparameterization trick briefly and intuitively. -->

To give the Musician a sense of overarching direction, the architecture is expanded into a Simple Variational Autoencoder [Kingma and Welling, 2013]. This introduces the concept of a "Composer": an inference network that condenses the entire musical piece into a continuous global latent space ($Z_p$). The Composer provides an abstract structural blueprint, dictating the overarching stylistic theme and mood, which is then combined with the explicit bar-level attributes to guide the Musician throughout the generation process.

**The Composer: Global Latent Inference**

This architecture introduces an inference network to capture this global theme, paired with the generative decoder established in Section 2.1. The inference network is a bidirectional Transformer encoder (\\(\approx 19.7\text{M}\\) parameters, \\(N=6\\) layers, \\(n_{\mathrm{embed}}=512\\)). By utilizing unmasked self-attention across the entire 8-bar input sequence \\(X\\), the encoder effectively analyzes both past and future musical context. A dedicated `[CLS]` token is prepended to the input sequence, and its final hidden state serves as the pooled representation of the entire sequence. This `[CLS]` token is linearly projected to parameterize the mean \\(\mu \in \mathbb{R}^{128}\\) and log-variance \\(\log(\sigma^2) \in \mathbb{R}^{128}\\) of a 128-dimensional approximate posterior distribution:

$$
Q_\phi(Z_p \mid X) = \mathcal{N}(Z_p; \mu(X), \operatorname{diag}(\sigma^2(X)))
$$

By forcing the entire 8-bar sequence through this restricted 128-dimensional probability distribution, the model creates a strict informational bottleneck. This mathematically maps the highly complex musical piece \\(X\\) into a compressed latent representation \\(Z_p\\) that captures its fundamental global structure. Formulating a continuous latent space provides the model with a dense, organized topology of musical ideas. Rather than memorizing rigid sequences, the model learns a smooth landscape where nearby points represent stylistically similar compositions, allowing the decoder to interpret abstract coordinates as meaningful, overarching musical themes.

In the VAE framework, the model is optimized by maximizing the Evidence Lower Bound (ELBO):

$$
\mathcal{L}&#95;{\mathrm{ELBO}} = \mathbb{E}&#95;{Q_\phi}\left[ \log P_\theta(X \mid Z_p, A) \right] - D_{\mathrm{KL}}(Q_\phi(Z_p \mid X) \parallel P(Z))
$$

Intuitively, the ELBO acts as a statistical tug-of-war. The first term, the expected reconstruction loss (\\(\mathbb{E}&#95;{Q_\phi}\\)), forces the decoder to accurately reproduce the original music using the provided latent vector \\(Z_p\\) and explicit attributes \\(A\\). The second term, the Kullback-Leibler (KL) divergence, acts as a penalty that regularizes the approximate posterior toward a standard normal prior, \\(P(Z) = \mathcal{N}(0, I_{128})\\). This mathematical constraint is critical: by forcing the encoder to map real sequences into a standard normal distribution during training, it formally justifies sampling a 128-dimensional global latent vector \\(Z_p \sim \mathcal{N}(0, I_{128})\\) purely from noise during inference. Since the decoder has been trained to interpret these sampled standard normal vectors as meaningful global signals, this random noise acts as a unique structural blueprint to guide the generation process.

To maintain differentiability for backpropagation, the latent vector is sampled using the reparameterization trick:

$$
Z_p = \mu + \sigma \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I_{128})
$$

**Mitigating Posterior Collapse**

Coupling a VAE with a powerful causal Transformer introduces a severe optimization challenge known as posterior collapse. Since the autoregressive decoder is exceptionally strong at predicting the next token \\(X_t\\) based purely on sequence history \\(X_{1:t-1}\\), it naturally learns to ignore the global latent vector \\(Z_p\\). If this occurs, the KL term in the ELBO loss collapses to zero, rendering the latent space completely meaningless. Consequently, sampling \\(Z_p\\) during inference would have zero effect on the generated music.

To mathematically force the decoder to utilize the latent space, three specific mechanisms are integrated:

1. **Cyclical \\(\beta\\)-Annealing:** A scheduling parameter \\(\beta\\) scales the KL term during training. By cyclically ramping \\(\beta\\) from \\(0.0\\) to \\(0.1\\), the model optimizes a modified ELBO objective:
$$
\mathcal{L}&#95;{\beta\text{-ELBO}} = \mathbb{E}&#95;{Q_\phi}\left[ \log P_\theta(X \mid Z_p, A) \right] - \beta \ D_{\mathrm{KL}}(Q_\phi(Z_p \mid X) \parallel \mathcal{N}(0, I_{128}))
$$
When the penalty is near zero, the model is allowed to freely encode rich structural information into \\(Z_p\\). As the regularization pressure of \\(\beta\\) increases, the model is forced to organize this established latent space into a smooth, continuous normal distribution without destroying the encoded musical identity.

2. **KL Free Bits (Hinge Loss):** A threshold \\(\lambda\\) is applied per latent dimension (\\(\lambda = 1.0\\) bit). Instead of forcing the KL divergence strictly to zero, the penalty is only applied if the divergence exceeds \\(\lambda\\):
$$
\mathcal{L}&#95;{\mathrm{KL}} = \sum_{i=1}^{128} \max \left( \mathbb{E}&#95;{X \sim \mathrm{batch}} \left[ D_{\mathrm{KL}}(Q_\phi(Z_{p,i} \mid X) \parallel \mathcal{N}(0, 1)) \right], \lambda \right)
$$
This gives the model a minimum "information budget," preventing the regularization penalty from completely erasing the expressivity of the latent dimensions.

3. **Condition Vector Concatenation (\\(Z_p\\) and \\(A_k\\)):** Inspired by the conditioning structure of MuseMorphose [Wu and Yang, 2021], the final condition vector \\(C_k\\) is constructed by concatenating the latent space with the explicit controls. The global latent \\(Z_p\\) is linearly projected to 384 dimensions, while the 128-dimensional attribute vector \\(A_k\\) (from Section 2.1) is preserved:
$$
C_k = [\mathrm{Linear}_{Z}(Z_p); A_k] \in \mathbb{R}^{512}
$$
This \\(3{:}1\\) dimensional split allocates \\(75\\%\\) of the conditioning bandwidth to the latent projection, structurally incentivizing—but not guaranteeing—that the decoder route global structure through \\(Z_p\\). By allocating 384 dimensions to \\(Z_p\\), the architecture provides substantial bandwidth for the latent code to transmit structural information. Ultimately, when this concatenated \\(C_k\\) vector is dynamically injected into the decoder layers via FiLM, the high-capacity Transformer is encouraged to respect the latent space, acting as a critical safeguard against posterior collapse.

**The Static Conditioning Flaw**

While these mitigations are intended to prevent posterior collapse, a fundamental architectural limitation persists: static conditioning. Since \\(Z_p\\) is a single, fixed vector for the entire sequence, its linear projection, \\(\mathrm{Linear}_{Z}(Z_p)\\), remains identical across all 8 bars. Injecting this unchanging global signal at every step mathematically dilutes the uniqueness of the per-bar condition vector \\(C_k\\). This creates an inherent clash between the rigid global theme \\(Z_p\\) and the highly dynamic, localized attribute variations demanded by \\(A_k\\) across the 4 attributes and 8 discrete bins. Consequently, the model's capacity to generate complex, structurally varied pieces is constrained, because the decoder struggles to concurrently satisfy a static overarching style and shifting bar-level instructions.

**Architectural Overview**
- Components: Bidirectional Transformer Encoder (\\(\approx 19.7\text{M}\\)), Causal Transformer Decoder (\\(\approx 22.8\text{M}\\)), Attribute Embedder & \\(Z_p\\) Projection (\\(\approx 0.05\text{M}\\))

- Total Parameters: \\(\approx 42.5\text{M}\\)

- Implementation: [simple_vae.py](https://github.com/kshoker12/Music-Generation-VAE/blob/main/ml/src/musicgen/models/simple_vae.py)

<figure id="fig-simple-training" class="arch-diagram">
  <img src="music_vae_content/final_diagrams/simple_train_final.svg" width="542" height="524" alt="Simple VAE training architecture: encoder latent inference, attribute embeddings, projected global conditioning, FiLM decoder, and ELBO losses" />
  <figcaption>
    Simple VAE training architecture. Ground-truth tokens \(X\) and past tokens \(X_{\lt t}\) feed a bidirectional Transformer encoder that infers global latent \(Z_p\); per-bar attributes are embedded into \(A_k\), concatenated with projected global conditioning \(\mathrm{Linear}_Z(Z_p)\) to form \(C_k\), and injected via FiLM at every decoder layer. The model is trained with reconstruction cross-entropy against \(X\) and \(\beta\)-weighted KL regularization of \(Q_\phi(Z_p \mid X)\) toward \(\mathcal{N}(0, I_{128})\).
  </figcaption>
</figure>

<figure id="fig-simple-inference" class="arch-diagram">
  <img src="music_vae_content/final_diagrams/simple_inference_final.svg" width="383" height="405" alt="Simple VAE inference architecture: sampled Z_p, attribute embeddings, projected global conditioning, FiLM decoder, autoregressive sampling, and generated sequence X prime" />
  <figcaption>
    Simple VAE inference architecture. A global latent \(Z_p \sim \mathcal{N}(0, I_{128})\) is drawn via latent sampling, projected to \(\mathrm{Linear}_Z(Z_p)\), and concatenated with user-defined attribute embeddings \(A_k\) to form \(C_k\), which is injected via FiLM at every decoder layer. The causal decoder autoregressively generates \(X' = (X'_1, \ldots, X'_T)\), conditioning each step on past generated tokens \(X'_{\lt t}\) and bar control \(C_{k(t)}\).
  </figcaption>
</figure>

### 2.3 Hierarchical Variational Auto-Encoder
<!-- 
[NARRATIVE ALIGNMENT]
- The Conductor: Explain the 2-layer GRU. 
- The 3:1 Ratio Trick: Highlight that $C_k = [z_k; A_k]$ uses 384 dims for the latent and 128 for attributes[cite: 8]. Dimensional split incentivizes latent routing; do not overclaim guaranteed usage[cite: 1, 8].
-->
To resolve the static conditioning flaw of the Simple VAE, the architecture is upgraded to a Hierarchical VAE by introducing a deterministic recurrent network: the "Conductor". The Conductor acts as a temporal translator. Instead of forcing the Musician to interpret a single, rigid global theme, the Conductor smoothly unrolls the Composer's blueprint into a dynamic sequence of unique, bar-by-bar instructions ($Z_k$).

**The Conductor: Bar-Level Latent Unrolling**

The architectural setup for inferring the global latent \\(Z_p\\) remains identical to the Simple VAE: utilizing the bidirectional Transformer encoder (established in Section 2.2) to infer \\(Z_p \sim \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))\\) during training, and sampling \\(Z_p \sim \mathcal{N}(0, I_{128})\\) during inference.

However, instead of statically projecting this latent across the sequence, \\(Z_p\\) is projected to initialize the hidden states of a 2-layer Gated Recurrent Unit (GRU) [Cho et al., 2014] (\\(\approx 1.9\text{M}\\) parameters) Conductor. For each bar \\(k \in \\{1, \dots, 8\\}\\), a learned bar-index embedding is fed into the GRU, producing a unique bar-level latent vector \\(Z_k\\):

$$
Z_k = \mathrm{GRU}\big(\mathrm{BarEmbed}_k;\ \mathrm{init}(Z_p)\big), \quad k \in \{1,\ldots,8\}, \quad Z_k \in \mathbb{R}^{384}
$$

where \\(\mathrm{init}(Z_p)\\) maps the global latent to the GRU's initial hidden state and \\(\mathrm{BarEmbed}_k \in \mathbb{R}^{384}\\) is a learned embedding for bar index \\(k\\). This unrolling process allows \\(Z_k\\) to vary across bars while remaining anchored to the global theme encoded in \\(Z_p\\).

**Dynamic Condition Vectors**

The unrolled latent \\(Z_k\\) is then concatenated with the explicit attribute vector \\(A_k\\) to construct the dynamic FiLM condition vector:

$$
C_k = [Z_k \;\ A_k] \in \mathbb{R}^{512}
$$

Since \\(Z_k\\) updates iteratively, the final condition vector \\(C_k\\) is now entirely unique for every single bar. This is the critical mathematical advantage of the hierarchical formulation: it explicitly breaks the static conditioning bottleneck by feeding the decoder a continuously evolving stream of instructions. It ensures that each bar receives highly unique conditioning signals while still strictly adhering to the overarching global theme established by the Composer.

By generating these dynamic, bar-specific latent instructions, the Hierarchical VAE optimizes an identical modified ELBO objective to the Simple VAE (utilizing \\(\beta\\)-annealing and KL free bits to effectively mitigate posterior collapse), while successfully bypassing the static conditioning bottleneck. Ultimately, this architecture establishes a robust generative process that synthesizes highly creative and dynamic compositions. By resolving the clash between global and local instructions, it allows for the generation of music that maintains a clear, evolving long-term structure while seamlessly executing precise, bar-by-bar stylistic shifts.

**Architectural Overview**
- Components: Bidirectional Transformer Encoder (\\(\approx 19.7\text{M}\\)), 2-Layer GRU Conductor (\\(\approx 1.9\text{M}\\)), Causal Transformer Decoder (\\(\approx 22.8\text{M}\\)), Attribute Embedder (\\(\approx 0.05\text{M}\\))

- Total Parameters: \\(\approx 44.3\text{M}\\)

- Implementation: [vae.py](https://github.com/kshoker12/Music-Generation-VAE/blob/main/ml/src/musicgen/models/vae.py)

<figure id="fig-hierarchical-training" class="arch-diagram">
  <img src="music_vae_content/final_diagrams/hierarchical_train_final.svg" width="537" height="524" alt="Hierarchical VAE training architecture: encoder latent inference, GRU conductor unrolling, attribute embeddings, dynamic FiLM conditioning, and ELBO losses" />
  <figcaption>
    Hierarchical VAE training architecture. Ground-truth tokens \(X\) and past tokens \(X_{\lt t}\) feed a bidirectional Transformer encoder that infers global latent \(Z_p\); a 2-layer GRU Conductor initializes from \(Z_p\) and unrolls learned bar-index embeddings into bar-level latents \(\{Z_k\}_{k=1}^{8}\), which are concatenated with embedded attributes \(A_k\) to form dynamic condition vectors \(C_k = [Z_k; A_k]\) and injected via FiLM at every decoder layer. The model is trained with reconstruction cross-entropy against \(X\) and \(\beta\)-weighted KL regularization of \(Q_\phi(Z_p \mid X)\) toward \(\mathcal{N}(0, I_{128})\).
  </figcaption>
</figure>

<figure id="fig-hierarchical-inference" class="arch-diagram">
  <img src="music_vae_content/final_diagrams/hierarchical_inference_final.svg" width="400" height="419" alt="Hierarchical VAE inference architecture: sampled Z_p, GRU conductor unrolling, attribute embeddings, dynamic FiLM conditioning, autoregressive sampling, and generated sequence X prime" />
  <figcaption>
    Hierarchical VAE inference architecture. A global latent \(Z_p \sim \mathcal{N}(0, I_{128})\) is drawn via latent sampling; the GRU Conductor initializes from \(Z_p\) and unrolls learned bar-index embeddings into bar-level latents \(\{Z_k\}_{k=1}^{8}\). User-defined attribute embeddings \(A_k\) are concatenated to form \(C_k = [Z_k; A_k]\), which is injected via FiLM at every decoder layer. The causal decoder autoregressively generates \(X' = (X'_1, \ldots, X'_T)\), conditioning each step on past generated tokens \(X'_{\lt t}\) and bar control \(C_{k(t)}\).
  </figcaption>
</figure>

### 2.4 Resource-Constrained Optimization

<!--
[NARRATIVE ALIGNMENT]
- Engineering Flex: Treat the 16GB Kaggle T4 limit as a design catalyst, not a roadblock.
- Memmap streaming, collate-time Y-shift, mixed precision, and gradient accumulation.
-->

While the Hierarchical VAE elegantly solved the theoretical problem of global-to-local coherence, executing it introduced a severe practical bottleneck. Training this system is not merely an architectural problem, but a systems problem as well. The same ELBO objective that regularizes the Composer (\\(Z_p\\)) also requires streaming \\(1.43\\) million long-context sequences (\\(T=1024\\)) through a \\(\approx 44.3\text{M}\\)-parameter graph on a \\(16\\)GB Kaggle T4x2 GPU. The systems-level optimizations below allow the Composer (Encoder), Conductor (GRU), and Musician (Decoder) to be optimized jointly without hardware failure.

**Streaming and Target Shifting**

The first physical limit was data ingestion. As introduced in Section \\(1.3\\), token sequences stream from `uint16` memory-mapped arrays directly to VRAM. To protect I/O bandwidth, autoregressive target sequences \\(Y\\) are omitted from the saved dataset entirely. Instead, \\(Y\\) is derived at batch time in the DataLoader `collate_fn` by left-shifting the input tensor \\(X\\) and appending a terminal `PAD` token, then transferred to the GPU with the batch. This avoids materializing a redundant \\(\approx 1.46\\)-billion-token target tensor on disk while preserving standard next-token supervision.

**Memory and Gradient Stability**

The second bottleneck was activation memory. The bidirectional Transformer encoder (Composer, \\(\approx 19.7\text{M}\\) parameters) and GRU Conductor (\\(\approx 1.9\text{M}\\) parameters) add substantial capacity, but backpropagating through the causal decoder's self-attention over \\(T=1024\\) tokens dominates VRAM consumption. To prevent Out-of-Memory (OOM) failures, VRAM limits are decoupled from effective batch size. FP16 mixed-precision training is combined with gradient accumulation: micro-batches of \\(8\\) are processed and gradients are accumulated over \\(2\\) steps, simulating an effective batch size of \\(16\\). This stabilizes gradients for both reconstruction cross-entropy and KL divergence across the encoder, conductor, and decoder when optimizing the ELBO.

**Training Notebooks**

The core training implementations are documented in the [Kaggle notebooks](https://github.com/kshoker12/Music-Generation-VAE/tree/main/kaggle/notebooks) directory:

- Plain Transformer: [03_plain_transformer_dev.ipynb](https://github.com/kshoker12/Music-Generation-VAE/blob/main/kaggle/notebooks/03_plain_transformer_dev.ipynb)
- Simple VAE: [05_simple_vae_dev.ipynb](https://github.com/kshoker12/Music-Generation-VAE/blob/main/kaggle/notebooks/05_simple_vae_dev.ipynb)
- Hierarchical VAE: [04_vae_dev.ipynb](https://github.com/kshoker12/Music-Generation-VAE/blob/main/kaggle/notebooks/04_vae_dev.ipynb)

See Appendix A for exact hyperparameters, learning rate schedules, and token specifications.

## 3 Results

### 3.1 Reconstruction Fidelity and Latent Capacity
<!-- 
[NARRATIVE ALIGNMENT]
- E1 Eval: Highlight that the Hierarchical VAE achieved the lowest perplexity (5.427) while maintaining healthy KL bits (~110 bits/sample)[cite: 8]. Prove that posterior collapse was defeated.
-->
To evaluate how well the architectures capture the underlying musical distribution and utilize the variational latent bottleneck (\\(Z_p\\)), teacher-forced Cross-Entropy (CE) and Perplexity (PPL) were measured on a held-out test set of \\(\approx 1.58\\) million tokens. For the VAE variants, the total information bandwidth successfully passing from the encoder to the decoder was also tracked. Full evaluation script at [08_eval_recon_kl.ipynb](https://github.com/kshoker12/Music-Generation-VAE/blob/main/kaggle/notebooks/08_eval_recon_kl.ipynb).

<figure class="table-figure">
<table>
<thead>
<tr><th>Architecture</th><th>Parameters</th><th>Cross-Entropy (nats)</th><th>Perplexity (PPL)</th><th>Total KL (bits/sample)</th></tr>
</thead>
<tbody>
<tr><td>Plain (Baseline)</td><td>\\(\approx 22.8\text{M}\\)</td><td>\\(1.693\\)</td><td>\\(5.51\\)</td><td>—</td></tr>
<tr><td>Simple VAE</td><td>\\(\approx 42.5\text{M}\\)</td><td>\\(1.695\\)</td><td>\\(5.52\\)</td><td>\\(100.2\\)</td></tr>
<tr><td>Hierarchical VAE</td><td>\\(\approx 44.3\text{M}\\)</td><td>\\(\boldsymbol{1.678}\\)</td><td>\\(\boldsymbol{5.43}\\)</td><td>\\(\boldsymbol{110.0}\\)</td></tr>
</tbody>
</table>
<figcaption>Teacher-forced reconstruction and KL bandwidth on the held-out test set. Lower is better for cross-entropy and perplexity; higher is better for total KL.</figcaption>
</figure>

**Understanding the Metrics**

- **Cross-Entropy (nats) & Perplexity (PPL):** These metrics quantify the autoregressive accuracy of the decoder. A Perplexity of \\(5.43\\) means that at any given step, the model has narrowed down the correct next musical event to roughly \\(5\\) equally likely options out of the entire \\(195\\)-token vocabulary. All models performed exceptionally well, with the Hierarchical VAE taking the lead. Interestingly, simply introducing a global latent vector does not automatically improve reconstruction performance: the Simple VAE performs nearly identically to the Plain baseline (\\(5.52\\) vs. \\(5.51\\) PPL), suggesting that latent information alone is insufficient unless it can be presented to the decoder in a form that remains useful throughout the sequence.
- **Total KL (bits/sample) & Information Distribution:** This measures the total information "bandwidth" passed from the Composer (encoder) to the Musician (decoder). The Hierarchical VAE successfully transmits approximately \\(110\\) bits of information through the bottleneck, indicating that the encoder is communicating a substantial amount of information to the decoder rather than collapsing toward the prior. Crucially, with a Free Bits hinge of \\(\lambda=1.0\\) bit per dimension applied during training, the optimizer intelligently "smeared" these \\(110\\) bits evenly across the \\(128\\)-dimensional space (averaging \\(\approx 0.86\\) bits/dim). This indicates the regularization functioned as intended: the model built a continuous map of musical structure distributed softly across the entire latent space.
- **Posterior Collapse:** Neither VAE exhibits the hallmark signature of posterior collapse. Rather than driving the KL divergence toward zero and reducing the model to a standard autoregressive decoder, both architectures maintain substantial information flow through the latent bottleneck throughout training.

**Resolving the Static Conditioning Bottleneck**

These reconstruction metrics validate the architectural narrative from Sections \\(2.2\\) and \\(2.3\\). The Plain baseline conditions the decoder only on per-bar attribute vectors \\(A_k\\). The Simple VAE adds a global piece-level latent \\(Z_p\\), but that vector is held fixed for all \\(8\\) bars, so the Musician still receives the same global instruction at every step. Reconstruction reflects this constraint: Simple and Plain are nearly identical (\\(5.52\\) vs. \\(5.51\\) PPL) despite the Simple model's larger capacity and an active latent bottleneck. The Hierarchical VAE changes what crosses the interface by having the Conductor unroll \\(Z_p\\) into sequence-aware, per-bar latents \\(Z_k\\), so \\(C_k = [Z_k ; A_k]\\) can evolve bar by bar. That yields the best score overall (\\(5.43\\) PPL) with only \\(\approx 1.8\text{M}\\) additional parameters over the Simple VAE, unlikely to be explained by scale alone. In short, a global latent helps reconstruction only when it is translated into localized instructions the autoregressive decoder can use throughout the sequence.


### 3.2 Attribute Controllability
<!-- 
[NARRATIVE ALIGNMENT]
- E2 Eval: Discuss the trade-off. Plain/Simple models win on exact-bin accuracy (plain 0.898, simple 0.894), but the Hierarchical model wins on monotonic control (mean Pearson r 0.943; note density ~0.94 per-attribute)[cite: 8].
- Sample counts differ by model (plain 96/attr, simple 104, vae 120)—describe protocol, not uniform N.
-->

Section \\(3.1\\) measured reconstruction quality; this evaluation asks whether the user-facing controls actually work. For each of the four attributes, the requested bin was swept from \\(0\\) to \\(7\\) while the other three attributes were held fixed at bin \\(3\\). After autoregressive generation, raw bar-level attributes were recomputed from the output tokens and scored by exact-bin accuracy, \\(\pm 1\\)-bin accuracy, and Pearson \\(r\\). Sample counts per attribute differ slightly by architecture (Plain \\(96\\), Simple \\(104\\), Hierarchical \\(120\\)). Full evaluation script at [08_eval_all_combined.ipynb](https://github.com/kshoker12/Music-Generation-VAE/blob/main/kaggle/notebooks/08_eval_all_combined.ipynb).

<figure class="table-figure">
<table>
<thead>
<tr><th>Architecture</th><th>Exact-Bin Accuracy ↑</th><th>±1-Bin Accuracy ↑</th><th>Pearson \\(r\\) ↑</th></tr>
</thead>
<tbody>
<tr><td>Plain (Baseline)</td><td>\\(\boldsymbol{0.898}\\)</td><td>\\(0.945\\)</td><td>\\(0.826\\)</td></tr>
<tr><td>Simple VAE</td><td>\\(0.894\\)</td><td>\\(0.966\\)</td><td>\\(0.853\\)</td></tr>
<tr><td>Hierarchical VAE</td><td>\\(0.525\\)</td><td>\\(\boldsymbol{0.992}\\)</td><td>\\(\boldsymbol{0.943}\\)</td></tr>
</tbody>
</table>
<figcaption>Attribute controllability over a full \\(0 \rightarrow 7\\) bin sweep (means across four attributes). Higher is better for each metric.</figcaption>
</figure>

**Understanding the Metrics**

- **Exact-bin accuracy & \\(\pm 1\\)-bin accuracy:** Exact-bin accuracy is the fraction of runs where the measured attribute equals the requested bin; \\(\pm 1\\)-bin accuracy allows a one-bin tolerance, which is natural under \\(8\\)-level quantization. All architectures achieve \\(\pm 1\\)-bin accuracy \\(\ge 94\\%\\), supporting the FiLM design from Section \\(2.1\\): injected condition vectors (\\(C_k\\)) keep outputs near the requested setting. The Plain and Simple models also snap to the exact bin label most of the time (\\(\approx 90\\%\\)).
- **Pearson \\(r\\) (monotonic controllability):** Pearson \\(r\\) is the linear correlation between the requested bin index and the measured attribute level across the \\(0 \rightarrow 7\\) sweep. Values near \\(1\\) mean that turning the dial up reliably increases the measured attribute; values near \\(0\\) mean the output barely tracks the request. The Plain and Simple models show weaker monotonic scaling (\\(r \approx 0.83\\)–\\(0.85\\)), while the Hierarchical VAE achieves the highest mean \\(r\\) (\\(0.943\\)), including \\(0.94\\) on note density, the attribute where non-hierarchical models struggle most. In practice, the hierarchical model is less likely to hit the precise bin label, but far more likely to move the music in the correct direction when the dial is turned.

**Coherent Control, Not Just Bin Matching**

These results should be read against the problem the paper set out to solve. Exact-bin accuracy asks a narrow question: did the measured attribute land on the requested integer bin? Plain and Simple models answer that well (\\(\approx 0.90\\)), but that metric does not test whether a model can steer an entire musical excerpt coherently as the dial moves. That is the harder requirement behind Sections \\(2.2\\) and \\(2.3\\): user attributes alone give the Musician local instructions, not a durable evolving structure, and a static global latent cannot vary those instructions bar by bar.

Pearson \\(r\\) is the scorecard for that harder form of controllability. The Hierarchical VAE leads clearly (\\(r = 0.943\\) vs. \\(0.826\\) Plain and \\(0.853\\) Simple), with \\(\pm 1\\)-bin accuracy at \\(0.992\\). Lower exact-bin accuracy (\\(0.525\\)) is consistent with a stochastic global latent and bar-varying \\(C_k = [Z_k ; A_k]\\): the model trades sharp bin matching for smoother monotonic control (higher Pearson \\(r\\)). Plain and Simple architectures are better at stamping a target bin onto each individual bar: Plain relies on fixed attribute vectors alone, while Simple adds a static global \\(Z_p\\) but still cannot vary latent instructions bar by bar the way the Conductor does. Section \\(3.1\\) showed that hierarchical structure helps reconstruction; this evaluation shows it also enables the most reliable monotonic controllability in practice.

### 3.3 Latent Space Structure
<!-- 
[NARRATIVE ALIGNMENT]
- Inspect whether μ organizes by coarse attribute profiles; hierarchical > simple on kNN purity and perturbation stability.
- Visual: Simple = diffuse cloud; Hierarchical = clearer regional grouping. Soften full disentanglement claims.
-->

Section \\(3.2\\) confirmed that the models follow explicit attribute controls. The next question is how they organize musical ideas internally, inside the \\(128\\)-dimensional global latent space (\\(Z_p\\)). For every \\(8\\)-bar chunk in a held-out test set, the most frequent bin (mode) is taken for each of the four attributes, forming a compact profile of the piece's overall character (polyphony, rhythmic intensity, velocity dynamics, note density). Pieces with similar profiles are grouped into \\(k=4\\) musical regimes using K-Means clustering, with \\(k\\) chosen by majority consensus across Silhouette, Calinski-Harabasz, and Davies-Bouldin scores. The encoded global latent mean (\\(\mu\\)) is then extracted for \\(20{,}000\\) test excerpts and projected from \\(128\\) dimensions down to \\(2\\)D using UMAP, as an attempt to visualize and understand organization within the latent space. The resulting map is compared against these attribute-based regimes, even though the models were never trained to classify them. A perturbation test adds Gaussian noise to \\(\mu\\) before decoding to measure how stable each representation is under small latent shifts. The Plain Transformer is omitted here, as it has no encoder latent. Full evaluation scripts: [022_latent_space_vae_attr.ipynb](https://github.com/kshoker12/Music-Generation-VAE/blob/main/kaggle/notebooks/022_latent_space_vae_attr.ipynb), [023_latent_space_simple_attr.ipynb](https://github.com/kshoker12/Music-Generation-VAE/blob/main/kaggle/notebooks/023_latent_space_simple_attr.ipynb).

<figure class="table-figure">
<table>
<thead>
<tr><th>Architecture</th><th>kNN Purity (PCA) ↑</th><th>kNN Purity (UMAP) ↑</th><th>NMI ↑</th><th>Perturbation Preservation ↑</th></tr>
</thead>
<tbody>
<tr><td>Random baseline</td><td>\\(0.250\\)</td><td>\\(0.250\\)</td><td>—</td><td>—</td></tr>
<tr><td>Simple VAE</td><td>\\(0.293\\)</td><td>\\(0.363\\)</td><td>\\(0.047\\)</td><td>\\(0.742\\)</td></tr>
<tr><td>Hierarchical VAE</td><td>\\(\boldsymbol{0.374}\\)</td><td>\\(\boldsymbol{0.382}\\)</td><td>\\(\boldsymbol{0.206}\\)</td><td>\\(\boldsymbol{0.813}\\)</td></tr>
</tbody>
</table>
<figcaption>Latent organization on the held-out test set (\(k=4\) attribute-profile clusters, \(k_{\mathrm{NN}}=15\)). Higher is better for each metric.</figcaption>
</figure>

**Understanding the Metrics**

- **kNN Purity:** This measures local neighborhood consistency in latent space. If a piece is selected at random, what fraction of its \\(15\\) closest neighbors share the same attribute-profile cluster? A score of \\(0.25\\) is random chance for four clusters. The Hierarchical VAE reaches \\(0.38\\) in UMAP space, outperforming the Simple VAE (\\(0.36\\)) and indicating that nearby latent coordinates tend to represent pieces with a similar overall musical character.
- **NMI (global alignment):** Normalized Mutual Information scores how well a global partition of \\(\mu\\) aligns with the attribute-profile clusters (higher is better). See [Appendix B](#appendix-b-global-latent-alignment-nmi) for the full protocol and interpretation.
- **Neighborhood Preservation (Perturbation):** This tests structural stability. A piece's latent coordinate (\\(\mu\\)) is perturbed with Gaussian noise and decoded; the output is re-binned and checked against the original attribute-profile cluster. The Hierarchical VAE preserves its cluster label in \\(81.3\\%\\) of trials, compared to \\(74.2\\%\\) for the Simple VAE, evidence of a more continuous latent landscape where small moves produce musically coherent variation rather than abrupt stylistic jumps.

<figure id="fig-latent-space-attr" class="arch-diagram">
  <img src="assets/piano-music-generative-models/figures/latent_space_attr_umap_k4.png" alt="Side-by-side UMAP projections of encoder mean μ for Simple VAE and Hierarchical VAE, colored by attribute-regime clusters (k=4)" />
  <figcaption>
    UMAP projection of encoder mean \(\mu\), colored by attribute-profile cluster (\(k=4\)), as an attempt to visualize organization within the latent space. <strong>Left:</strong> Simple VAE. <strong>Right:</strong> Hierarchical VAE. Each color marks pieces that share a similar dominant attribute setting across all eight bars.
  </figcaption>
</figure>

**Visual Proof of the Conductor**

The UMAP projections above offer an exploratory view of latent organization; together with the table, they show a clear structural gap between the two VAE variants. In the Simple VAE, the latent map is a diffuse, heavily mixed cloud. With the same static global vector conditioning all eight bars, the encoder must compress shifting local detail and overarching style into one coordinate, and the resulting space stays close to random on global alignment scores (NMI \\(\approx 0.05\\)).

In the Hierarchical VAE, the same coloring scheme is far easier to read. The projection separates into a few distinct neighborhoods with visible gaps between them, rather than one tangled cloud where every color is interleaved. Pieces with similar attribute profiles tend to sit near one another, so cluster color concentrates by region. Those neighborhoods are not pure one-color patches (several labels can still appear in the same area), but the overlap stays local; in the Simple plot, by contrast, all four colors bleed through the same mass. By delegating bar-by-bar execution to the GRU Conductor, the global latent \\(Z_p\\) is freed from tracking localized note-level detail, allowing the Composer to carve the \\(128\\)-dimensional space into broader zones of overarching musical character. Higher kNN purity, stronger perturbation preservation (\\(81.3\\%\\)), and improved global NMI (\\(0.21\\) vs. \\(0.05\\); Appendix B) support that reading: the hierarchical model builds a latent space with reliable local structure, where nudging a coordinate produces a smooth variation of the current style rather than an unrelated excerpt. That closes the arc from Sections \\(2.3\\), \\(3.1\\), and \\(3.2\\): the Conductor improves reconstruction and monotonic control, and also yields a global latent topology that is measurably better organized.



### 3.4 Generated Samples 
<!-- Qualitative evaluation, linking back to the interactive dashboard. -->

## 4 Conclusion

### 4.1 Summary
<!-- Wrap up: How combining rigorous statistical methods (VAE, FiLM) with hierarchical temporal unrolling solved long-range piano generation. -->

### 4.2 Limitations
<!-- Transparency: Discuss the $4/4$ time signature filter[cite: 1, 8]. Frame it as a necessary scope for mathematical consistency, but acknowledge it limits rhythmic diversity. -->
- Lot of this depends on sampling and luck for subjective analysis since chunks can be from arbitrary moments of songs. However, listening to 10 songs is quite a solid sample size when doing comparison. 
- Since training data includes chunks of pieces, sometimes a generates 8 bar piece doesn't end the bar smoothly and transitions to something else.  

### 4.3 Future work
<!-- Adversarial Latent Disentanglement (GRL) and Monte Carlo Tree Search (MCTS) decoding for better global coherence[cite: 8]. -->

### 4.4 Self-Reflection
<!-- Portfolio close: Mastering complex architectures, combating posterior collapse, and engineering efficient data pipelines. -->

## References
<!-- Keep existing references intact -->

## Appendix
<!-- Keep existing appendices for detailed spec breakdown -->

### Appendix B: Global Latent Alignment (NMI)

Section \\(3.3\\) reports Normalized Mutual Information (NMI) as a **global** complement to kNN purity (local) and perturbation preservation (decode stability).

**Protocol.** Attribute-profile clusters are defined by K-means (\\(k=4\\)) on the 4-D mode vector for each test excerpt (same labels used to color the UMAP figures). Independently, K-means with the same \\(k=4\\) is fit on the full \\(128\\)-D encoder mean \\(\mu\\). NMI measures how much knowing a point's attribute-profile cluster reduces uncertainty about its \\(\mu\\)-cluster assignment. A score near \\(0\\) means the two partitionings are unrelated; \\(1\\) would mean perfect alignment.

| Architecture | NMI |
| --- | --- |
| Simple VAE | \\(0.047\\) |
| Hierarchical VAE | \\(\boldsymbol{0.206}\\) |

The Simple VAE score is near chance: cutting \\(\mu\\) into four groups does not recover the attribute-based regimes. The Hierarchical VAE score is modest in absolute terms but roughly \\(4\\times\\) higher, consistent with the spatially separated UMAP islands in Section \\(3.3\\)—where attribute-profile colors concentrate regionally even though no island maps cleanly to one color alone. Source notebooks: [022_latent_space_vae_attr.ipynb](https://github.com/kshoker12/Music-Generation-VAE/blob/main/kaggle/notebooks/022_latent_space_vae_attr.ipynb), [023_latent_space_simple_attr.ipynb](https://github.com/kshoker12/Music-Generation-VAE/blob/main/kaggle/notebooks/023_latent_space_simple_attr.ipynb).