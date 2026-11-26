# Steering SD 1.5 toward background blur: NA-RFM vs TFG, with BokehDiff as the reference (feasibility pilot)

Branch `q2-pareto48`. Overnight feasibility pilot, Nov 25-26, 2026. Question: **can either steering method change
background blur during Stable Diffusion generation without substantially changing the subject or the scene?** Not a
benchmark. No refocusing, no transparency, no extra layers, models or datasets beyond what is listed here.

## Design (frozen before any data was generated; `design.sha256` covers this section)

### Stage 0 audit: what this pilot fixes from earlier work

- **Earlier SD blur steering** (previous repo: `q2-sd-dof`, `q2-sd-bias`, `q2-sd-dof-direction`, `q2-sd-dof-masked`,
  `q2-sd-dof-timing`):
  - The direction data (svnfs/depth-of-field) are *different photos* for shallow and deep focus. Shallow-focus
    photos are mostly close-ups (84% vs 11% by confident CLIP labels), so the direction mixes blur with framing.
    A portrait-biased reference set enlarged people at matched (near-zero) blur.
  - The RFM edit lowered background and foreground sharpness alike and often changed objects and layout. Late-only
    steering kept the layout but gave almost no blur. Post-hoc blur beat every steering variant.
  - **Fix here:** the direction and the predictor are learned from **same-scene pairs** (CamEdit50K, below), so
    content and framing are held fixed within each pair. Scene change is measured against a **blur-matched**
    reference, so the intended blur does not count as scene change.
- **NA-RFM as implemented upstream** (`isotrivial/na-rfm` at `8e9619bc`): RFM direction (`DualRFM`, Laplace kernel)
  on noised activations at one timestep. Sampling has three parts:
  - an additive adaptive hook h ← h + w‖h‖d/‖d‖;
  - CFG at the x0 level between hook-on and hook-off (`RFMCFGWrapper`);
  - plus **noise alignment**: x0 += γ (W_class(x_t/√ᾱ, σ) − W_full(x_t/√ᾱ, σ)), where W are PCA Wiener denoisers
    (`pca/gaussian_denoiser.py`).

  The old SD scripts used the hook and CFG but **not** the noise-alignment term. They are mean-difference/RFM edits,
  not full NA-RFM. Here all three parts are used. The PCA filters are built in SD's latent space; upstream builds
  them in pixel space for pixel-space models.
- **TFG** (`YWolfeee/Training-Free-Guidance` at `f8d17f3e`): `methods/tfg.py` `TFGGuidance.guide_step` is used
  verbatim, including its DDIM update, ρ (x_t gradient through the UNet), μ (iterated x0 gradient) and σ (MC
  smoothing) schedules. Only the guider (our blur predictor on the VAE-decoded x0, as in TFG's own SD tasks) and the
  import plumbing are ours. TFG's task registry imports audio and molecule packages, so `tasks.base`, `tasks.utils`
  and `utils.configs` are replaced by stubs. `rescale_grad` is copied verbatim; for image tensors it is the
  identity.
- **Round 45 controller** (`bokeh/controller/results/rule45.json`): isotonic curve of delivered blur against K·gap.
  - It was fitted on RealVisXL images saved at 768 px, with BokehDiff's own depth (Depth Anything V2 Base) and mask
    (BiRefNet), fp32 PISA, targets in px at the 768-px scale.
  - SD 1.5 makes 512-px images, so they are upsampled to 768 (Lanczos) and prepared exactly as in round 45. The
    target is converted to the 768 scale (×1.5).
  - **Transfer limitation:** upsampled SD 1.5 images are softer than native 768-px RealVisXL images, so the curve may
    be off. This is reported, not corrected.
- **Compute and caches** (DSMLP): A30 (24 GB) requested with `-v a30`. Cached: SD 1.5, Depth Anything V2 Small/Base,
  CLIPSeg, DINOv2-base, CLIP-L, BiRefNet, BokehDiff weights and venv.
- **CamEdit50K focal-plane subset** (inspected Nov 25 from the parquet metadata of all shards and 26 pairs of
  shard 001):
  - **What is published:**
    - parquets for shards 001-009: 26,620 pairs, 15,479 unique inputs;
    - tars for 001-008 only (the README's 11 shards are not all published).
  - **Labels:** only a focal-plane value in [0, 1] and an instruction string. There are **no aperture, blur-radius
    or depth labels**.
  - **What the pairs are:**
    - the input is a sharp real photo;
    - the edited image is the same photo, pixel-aligned, with synthetic depth-dependent defocus;
    - pairs are therefore same-scene by construction.
  - **Focal-value convention (by inspection, not documented):**
    - a high value focuses near, so the far background is blurred (1.0: far trees blurred, near grass sharp);
    - a low value focuses far (0.1: near wall blurred);
    - at 0 the edit is nearly invisible in mostly-far scenes.
  - **Blur amount** varies by scene, so it is **measured** on each pair (below) rather than read from the label.

### Shared generation settings (both steering methods, every arm)

| setting | value |
|---|---|
| checkpoint | `stable-diffusion-v1-5/stable-diffusion-v1-5` (HF cache) |
| resolution | 512 × 512 |
| scheduler | diffusers `DDIMScheduler` from the checkpoint's config (leading spacing, offset 1), 50 steps, η = 0 |
| CFG | 7.5, empty negative prompt |
| dtypes | UNet fp16; VAE decode fp32 (final images and TFG's guidance decode); RFM, PCA and metrics fp32 |
| initial noise | `torch.randn(4, 64, 64, generator=CPU seed)` × init sigma, one seed per prompt |
| prompt suffix | ", highly detailed background" on every prompt |

Schedules (DDIM step index k of 50): **early** 0-16, **middle** 17-33, **late** 34-49, **all** 0-49.
Blur target: **T = 3 px** of added disk blur in the background at 512 px. Tolerance: TOL = max(1.33, 0.2 T) = 1.33 px,
round 40's max(2, 0.2 r) at 768 px converted to the 512 scale.

### Data and splits

- **Prompts** (`prompts48.py`): 24 prompts, each with a clear subject noun (used by the subject masker) and a
  textured background. **Dev:** 12 prompts, seeds 101-112. **Confirm:** 12 different prompts, seeds 201-212.
  - Confirmation outputs are not generated until the selection is frozen.
  - Confirmation references are generated only in Stage 6, so nobody can look at them while selecting.
- **CamEdit50K shard 001** (3,000 pairs; CC BY-NC 4.0, research use).
  - **Scene** = input image id. Split by `sha1(id) mod 20`: 0-13 **train**, 14-16 **val**, 17-19 **test**.
  - All pairs of a scene stay in one split. Both steering components fit on train, choose iterations on val, and
    report on test.
  - **Pair eligibility** (measured on the 512 center crop; depth = Depth Anything V2 Small on the sharp input; far =
    farthest 40% of pixels, near = nearest 30%):
    - focal ≥ 0.7;
    - far-region added blur (disk fit, below) in [1, 12] px on ≥ 2 tiles with R² ≥ 0.9;
    - near-region DoG band-energy ratio ≥ 0.75 (near region not blurred much).
  - One pair per scene (the highest focal value), at most 600 scenes.
- Evaluation scenes are SD generations from the prompts above, so they are disjoint from all CamEdit scenes by
  construction.

### NA-RFM

- **Candidate layers** (all three hooked in one collection pass):
  - **middle:** `mid_block` output, 1280×8×8;
  - **attention:** `up_blocks[1].attentions[1]` output (spatial transformer: self- and cross-attention), 1280×16×16;
  - **later:** `up_blocks[2].resnets[2]` output, 640×32×32.
- **Collection:**
  - each pair's sharp and blurred 512 crops are VAE-encoded (latent mean × 0.18215);
  - both are noised with the **same** ε (seeded by scene) at the timestep whose σ is closest to the paper's 0.344;
  - one UNet pass with the empty prompt, fp16 activations.
  - **Correspondence with sampling:** the direction learned at this single σ is applied at every steered step,
    scaled by the activation norm (w‖h‖), as upstream does. Each schedule's σ range is logged.
- **RFM:**
  - `DualRFM` from the NA-RFM code: Laplace kernel, bandwidth 5000·√(D/327,680) (the paper's SD bandwidth, scaled
    with feature dimension because features are standardised);
  - reg 1e-3, 5 iterations, upstream early stop;
  - fit on train-scene activations, iteration chosen on val scenes;
  - direction = `get_direction(topk=1)`;
  - reported: val and test AUC of the projection, against 10 random unit directions.
- **Noise alignment:**
  - PCA in latent space (16,384 dims) of train-scene latents; class = blurred latents, full = all latents;
  - k = min(N − 1, 512) components;
  - upstream `GaussianDenoiser` in Wiener mode.
- **Sampling** inside the schedule:
  - three UNet branches: uncond, cond, cond + hook (w = 0.7);
  - x0 = x0_cfg + s (x0_hook − x0_cond) + γ (W_class − W_full)(z_t/√ᾱ_t, σ_t);
  - ε is recomputed from x0, then a DDIM step;
  - no x0 clipping (latent space).
  - Outside the schedule, plain CFG.
- **Strengths:** multiplier m with s = 4m, γ = γ0·m, w = 0.7.
  - **γ0 = 1.** At smoke, ρ_n = mean over steered steps of ‖γ0·NA‖ / ‖s(x0_hook − x0_cond)‖ (m = 1, case 1). If
    ρ_n ∉ [0.1, 10], then γ0 ← γ0/ρ_n.
  - **Grid {0.5, 1, 2}** by default, adjusted from smoke (all steps, 2 smoke cases):
    - if m = 2 degenerates (image std < 0.5 × reference std, or non-finite), use {0.25, 0.5, 1};
    - otherwise, if m = 2 gives no response (background DoG ratio > 0.9 on both cases), use {1, 2, 4}.
- **Random control:** same norm, seeded Gaussian direction, everything else identical, including the NA term. It
  is run for the selected NA-RFM configurations on all 12 dev cases.

### TFG

- **Blur predictor P:**
  - **Model:** a small CNN, RGB 256×256 in [−1, 1] → per-pixel added-blur radius map, 64×64, px at 512 scale.
  - **Training data:**
    - train-scene sharp CamEdit inputs with synthetic depth-layered disk blur, rendered in linear light: 8 depth
      layers, random focus on the near half, maximum radius ~ U[0, 10];
    - augmentations: brightness/contrast jitter, label unchanged;
    - plus unblurred images (label 0).
  - L1 loss on the radius map; checkpoint chosen on val scenes.
  - Its target is **background-specific**: the guider uses P's mean over the reference's frozen background mask.
- **Predictor validation** (test scenes; hard gates marked ★):
  - ★ **blur response:** Spearman(true r, predicted mean) ≥ 0.9 over r ∈ {0, 1, 2, 4, 8}, and MAE ≤ 1.5 px for r ≤ 4;
  - ★ **contrast reduction** (×0.5, no blur): |Δ prediction| ≤ max(0.5 px, 25% of the r = 4 response);
  - **texture removal** (5×5 median filter): Δ ≤ the r = 2 response (reported);
  - **scene category** (CLIP zero-shot: indoor / urban / nature / other, unblurred images): spread of group means
    ≤ 1 px (reported);
  - **real-pair transfer:** Spearman between P's far-region mean on the edited image and the disk-fit radius of
    val/test CamEdit pairs (reported).
  - A failed ★ gate means TFG is not swept (diagnosis saved).
- **Guidance:**
  - logp(x0) = −(mean_bg P(decode(x0)) − T)²;
  - TFG's SD settings from `scripts/style_transfer.sh`: ρ 0.25, μ 2, σ 0.1, eps_bsz 1, recur_steps 1;
  - **iter_steps 4 → 2** (cost reduction, recorded now);
  - **η 1 → 0** (shared scheduler);
  - schedules ρ/μ 'increase', σ 'decrease';
  - ρ = μ = 0 outside the steering schedule.
  - The gradient path is: latent → fp32 VAE decode → avg-pool to 256 → P.
- **Strengths:** (ρ, μ) = m·m0·(0.25, 2), m ∈ {0.5, 1, 2}. m0 is the smallest value in {1, 4, 16} for which the
  smoke run (all steps, both smoke cases) raises P's background radius by ≥ 1 px and stays finite. If none does,
  m0 = 16 and TFG is flagged "weak at the tested strengths".
- **Independent evaluator:** TFG is scored only by the disk-fit estimator and the DoG band ratio below, never by P.
  P's own reading is logged for comparison.

### Measurements (regions frozen on each unsteered reference before any steered output exists)

- **Regions:**
  - **subject:** CLIPSeg(`a photo of a <noun>`) on the reference. Threshold 0.4, falling back to 0.3 then 0.2 if
    the area is under 0.5%. Largest components kept.
  - **interior:** subject eroded by 6 px.
  - **background:** pixels > 16 px from the subject and farther than the subject's median disparity minus 10% of
    the range (Depth Anything V2 Small).
  - **background tiles:** 64×64, stride 32, entirely in background, mean gradient ≥ 4/255, up to 8, most textured
    first (round 40 rule).
- **1. Background blur:**
  - **(a) primary:** added disk radius via round 40's estimator (`bokeh/common/blur40.py`):
    - fits reference ⊛ disk(r) + per-channel gain/offset to the output, in linear light;
    - tiles with R² ≥ 0.9 counted; median over ≥ 2 tiles, otherwise **unmeasurable**;
    - a scene change breaks the fit, which shows up as unmeasurable (reported, never imputed).
  - **(b) secondary, alignment-free:** DoG(σ 1 → 3) band-energy ratio, output/reference, over the background
    region.
- **2. Subject sharpness:** log DoG band-energy ratio in the interior.
- **3. Identity and layout,** each against the **blur-matched reference** (reference plus background disk blur at the
  output's measured radius, or at the radius matching its DoG ratio when unmeasurable, subject composited back with
  a feathered mask):
  - **scene change** = 1 − mean DINOv2 patch-token cosine (whole image) — the Pareto y-axis;
  - subject DINOv2 cosine (reference subject box);
  - subject-mask IoU and centroid shift (CLIPSeg with the same noun and threshold on the output);
  - depth Spearman correlation with the reference.
  Raw pixel difference is reported but never used alone.
- **4. Colour and cost:**
  - ΔE76 between Gaussian(σ = 8) low-passed Lab images, mean over the image;
  - wall seconds, UNet forwards, UNet backwards and VAE-decoder backwards per image.
- **Measurement validation** (Stage 1, the 12 dev references):
  - **(a)** background disk blur r ∈ {1, 2, 3, 4, 6}, subject composited back. Requires:
    - disk-fit within TOL(r) for ≥ 80% of measurable cases;
    - scene change ≤ 0.05, subject DINOv2 ≥ 0.95, IoU ≥ 0.9, ΔE ≤ 2.
  - **(b)** background contrast ×0.6: disk-fit ≤ 1 px. The DoG ratio's response is reported as its known confound.
  - **(c)** shift by 8 and 16 px: scene change ≥ 0.1, and centroid shift ≥ half the applied shift.
  - **(d)** global disk blur r = 2: subject sharpness log ratio ≤ −0.3.
  - If (a)'s disk-fit check fails, blur numbers are labelled **exploratory** and the pilot reports image grids,
    not pixel radii.
- **Per-image rules:**
  - **blur response:** disk-fit r ≥ 1 px, or (unmeasurable and DoG ratio ≤ 0.8);
  - **on target:** |r − T| ≤ 1.33 px;
  - **preserved:** subject DINOv2 ≥ 0.85, IoU ≥ 0.75, centroid shift ≤ 16 px, subject sharpness log ratio ≥ −0.35,
    ΔE ≤ 4 and scene change ≤ 0.15.
  - If validation (a) at r = 3 itself fails a preservation threshold, that threshold is set to the validation's
    90th percentile **before** steered outputs exist. The change is logged.

### Sequence and decision rules

1. **Smoke** (dev cases 1-2):
   - baseline run twice is identical;
   - NA-RFM with s = 0, γ = 0 (hook on, w = 0.7) is identical to the baseline;
   - NA-RFM with w = 0 (s = 4) matches within max |Δ| ≤ 2/255 and mean ≤ 0.1/255 (batch-of-3 kernels);
   - TFG with ρ = μ = 0 matches within the same tolerance (its own DDIM arithmetic);
   - hooks fire at the intended layers and only at scheduled steps (counted);
   - guidance gradients finite and nonzero;
   - UNet/VAE/P weights unchanged (checksum before and after);
   - dtypes logged; peak memory logged.

   Any failure stops dependent stages.
2. **NA-RFM layer screen:** dev cases 1-4, m = 1, middle schedule.
   - Only layers with val AUC ≥ 0.65 are eligible.
   - Score per layer = number of cases with blur response **and** subject DINOv2 ≥ 0.85 **and** IoU ≥ 0.75. Ties go
     to the lower mean scene change.
3. **Development sweep:** each valid method × 12 dev cases × 3 strengths × 4 schedules. NA-RFM random control at
   the selected configurations.
4. **Selection** (dev only), per method, among configurations with blur response in ≥ 8/12 cases:
   - **P (preservation):** most cases with response ∧ preserved, ties to the lower median scene change;
   - **B (stronger blur):** smallest median |r − T| among configurations with ≥ 6/12 preserved (else among all
     responding configurations). If B = P, only one is selected.

   Written to `selection48.json` and hashed **before** confirmation outputs exist.
5. **BokehDiff:**
   - every reference is upsampled to 768 and prepared with BokehDiff's own `prepare_data.py`, then round 40
     `regions40.py`;
   - gap = `fit42.gap`; K = `fit45.plan_K(rule45, 1.5 T, gap)`;
   - fp32 PISA render; output downsampled to 512; scored identically.
   - Scenes with < 2 behind tiles have no gap and are reported as controller-ineligible (no fallback).
6. **Confirmation** runs automatically for a method only if it passes the smoke controls, has a configuration with
   blur response in ≥ 8/12 dev cases, and has a configuration with ≥ 8/12 cases responding **and** preserved.
   - Selected configurations are run on the 12 confirm prompts, with unsteered, SD + BokehDiff and post-hoc disk
     blur at T.
   - Paired differences, bootstrap 95% intervals over the 12 cases.
   - No tuning on this set.
7. **Verdicts per method:**
   - **promising:** passes 6 on dev, and on confirm ≥ 8/12 cases respond ∧ are preserved;
   - **no advantage established:** fails 6, or fails the confirm bar, or is preserved only where it delivers less
     blur than BokehDiff at matched-blur comparison;
   - **unresolved:** controls or measurements were invalid.

   Comparisons with BokehDiff use matched delivered blur (both within TOL of T, or |r_a − r_b| ≤ 1 px). Unmatched
   cases are listed, never pooled.

### Budget (A30, one pod), and predeclared reductions

| item | estimate |
|---|---|
| CamEdit shard download + pair filtering | 0.4 h |
| activations (3 layers, ≤ 1,200 images) + RFM × 3 + PCA | 0.3 h |
| predictor training + validation | 0.4 h |
| references, regions, measurement validation | 0.2 h |
| smoke | 0.2 h |
| layer screen + NA-RFM sweep (12 + 144 + 24 images, ~3 s each) + metrics | 0.4 h |
| TFG sweep (144 images, ~25 steered steps × ~1.5 s) | 1.6 h |
| BokehDiff (24 renders) | 0.2 h |
| confirmation (≤ 4 configurations × 12 + references) | 0.5 h |
| total | **≈ 4.2 h** (cap 8 h) |

- **Storage** (workspace `outputs/pareto48/`, persistent): CamEdit tar 4.0 GB (deleted after extraction), crops
  ~0.4 GB, activations ~2.6 GB, images ~0.6 GB.
- **Reductions,** applied in this order if the smoke timing projects past 7 h, and recorded before any sweep
  output is viewed:
  - **R1:** TFG iter_steps 2 → 1;
  - **R2:** TFG sweep on dev cases 1-8 only;
  - **R3:** drop TFG strength m = 0.5.
- **Overnight driver** (`run48.py`):
  - stages run in order; each writes a manifest and is skipped only if the manifest and its outputs validate;
  - one OOM retry per stage with memory-saving settings (attention slicing; VAE decode under gradient
    checkpointing);
  - no other retries;
  - the pod is deleted at the end or on terminal failure.

## Status

Design frozen; code being written. Nothing has run yet.
