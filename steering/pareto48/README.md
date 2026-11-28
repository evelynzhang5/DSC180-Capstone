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

**Finished Nov 25, 2026, 23:28.**
- 3.46 GPU-hours on one pod (`p48`), on an A30 MIG slice.
- The pod was deleted after the run.
- The confirmation set was not generated, because no method passed the development gate (rule 6).

## Audit of round 48 (Nov 26, 2026): plan, frozen before it ran

Requested before any further steering sweep. Dev cases only; **the 12 confirmation prompts stay ungenerated.** Script:
`audit48.py`; outputs in `outputs/pareto48/audit/`; copied to `results/audit/`.

**A. Numerics under identical execution conditions** (10 eligible dev cases).
- **Per-pixel |Δ| statistics:** mean, p50, p90, p99, p99.9, max, fraction nonzero (0-255 scale).
- **Metric deltas:** scene change, disk-fit r, DoG ratio, subject DINOv2.
- **Pairs:**
  1. baseline rerun vs the saved round 48 reference (same code, new process);
  2. NA-RFM s = 0, γ = 0 (hook armed) vs baseline;
  3. NA-RFM w = 0, s = 4 with round 48's mixed batch shapes (hook-on batch-1 vs hook-off taken from the batch-2 CFG
     call) vs baseline;
  4. the same, but with the hook-off reference computed in a batch-1 call like the hook-on call (`same_batch`;
     upstream `RFMCFGWrapper` calls the model twice alike);
  5. baseline computed as two batch-1 calls (the floor) vs baseline;
  6. TFG ρ = μ = 0 vs baseline;
  7. TFG ρ = μ = 0 vs upstream TFG's own unguided `BaseGuidance.guide_step` (identical arithmetic);
  8. round 48's TFG late-third m = 1 outputs rerun vs the saved ones.
- **Sound if:**
  - 1, 2, 4, 7 and 8 are bit-identical;
  - in 3, 5 and 6, no metric moves by more than the decision resolution: scene change 0.01, r 0.25 px, DoG ratio
    0.02, DINOv2 0.01 (median over cases).
- **Is round 48 NA-RFM sensitive to the batch-shape choice?** NA-RFM m 0.5 late and m 1 all, mixed vs same_batch,
  compared on all measures.

**B. Method comparison on shared eligible scenes.** The 9 cases eligible for every arm (d10 has no BokehDiff render).
- per-arm counts;
- paired comparison with BokehDiff on "responds ∧ preserved" (exact two-sided sign test on discordant cases);
- paired scene-change differences (bootstrap 95%).

**C. TFG: offline prediction accuracy vs intervention effectiveness.**
- **Re-run with tracing:** TFG late m 1 and late m 2 (10 cases), and all-steps m 0.5 (4 cases: d01, d03, d05, d07),
  with round 48's settings. At every guided step, record:
  - the clean-image estimate before the x0 update (pre) and after it (post), decoded;
  - P(pre) and P(post), gradient norms, μ and ρ;
  - a finite-difference check of the guidance gradient (3 steps per case, first configuration).
  Upstream code runs unchanged; only wrappers record.
- **Independent measures on every pre/post pair** (aligned by construction): disk-fit added radius on the reference's
  frozen background tiles, background DoG ratio, and the fraction of the update's energy above 0.25 cycles/px.
- **Second predictor P2:**
  - same architecture, a different seed;
  - trained on the half of the train scenes with even `sha1` (P used all of them);
  - same validation gates.
  - P2 reads every pre/post pair and every final output.
- **Offline accuracy, separately:**
  - P and P2 on known background disk blur r ∈ {0, 1, 2, 3, 4, 6} applied to the dev references (SD domain) and to
    the last pre estimate of each traced run (in-trajectory domain);
  - slope and Spearman of reading against true r;
  - disk-fit recovery on the same images.
- **Classification:**
  - **E1, score gain without blur:** in ≥ 6/10 late-m1 or late-m2 cases, the final P reading exceeds the reference's
    by ≥ 1 px, while the disk-fit gain is ≤ 0.5 × the P gain (or unmeasurable) and the DoG ratio is ≥ 0.9.
  - **E2, P accurate offline in this domain:** on the known-blur controls, slope 0.7-1.3 and Spearman ≥ 0.9.
  - **E3, specific to P:** P2 reads ≤ 50% of P's gain on the same outputs (median over E1 cases).
  - **E4, sound guidance:** gradients finite; finite-difference relative error ≤ 10% (median).
- **Recommendation rule:**
  - **E1 ∧ E2 ∧ E3 ∧ E4 → predictor exploitation.** A small late-only TFG ensemble experiment is justified (proposed,
    not launched).
  - **E1 ∧ E4 but not E3** (an independently trained predictor is fooled too) → score gain without measurable blur
    is not specific to P. BokehDiff improvements are the more productive direction.
  - **Not E1** (late guidance barely moves P) → TFG at late steps is too weak. Same recommendation.
  - **Not E2** → P is inaccurate on SD images, an offline accuracy problem. Reported; no ensemble experiment.
  - **Not E4** → a guidance bug. Fix before any interpretation.

## Audit of round 48: results (Nov 26, 2026)

Full tables: `results/audit/report.md` (machine-readable: `audit.json`, `numerics.json`, `trace.json`, `p2.json`).
Traced images (456 MB) stay on DSMLP under `outputs/pareto48/audit/`. The confirmation prompts were not generated.

**Audit deviation:** run on a GTX 1080 Ti (11 GB), with `P48_MEMSAVE=1` (attention slicing + gradient checkpointing)
from the start, because A30 pods were repeatedly scheduled onto node n57, whose course-workspace mount failed (exit
32), and 2080 Ti pods could not be scheduled. Consequence: pairs 1 and 8 (rerun vs saved round 48 outputs) cannot be
bit-identical across GPUs and are not, so they test cross-hardware reproduction only. Every other pair compares runs
made inside this audit.

**A. Numerics.**
- Pairs 2, 4 and 7 are bit-identical in 10/10 cases. Zero strength is exactly a no-op once the execution path is
  identical.
- Pairs 3, 5 and 6 differ only at the numerical floor: mean |Δ| 0.58-0.61, p99 5-7, max 205-216 (a few unstable
  pixels), about 36% of pixels nonzero. Every metric delta is far below decision resolution: |Δ scene| ≤ 0.0004 and
  |Δ DoG| ≤ 0.003.
- The difference between the separate and joint reference calls is no larger than splitting the unsteered batch.
- Round 48's NA-RFM conclusions do not depend on batch shape. m0.5 late gives 5/10 both ways; m1 all gives 7 respond
  / 0 preserved both ways; median scene matches to 0.003.
- Pair 1 (cross-GPU): mean 1.21, p99 12. Pair 8: not identical (cross-GPU).

**B. Shared eligible scenes** (9 cases: d01, d03-d09, d11):

| arm | respond ∧ preserved | on target (\|r − 3\| ≤ 1.33) | sign test vs BokehDiff (unadjusted) | scene − BokehDiff (95%) |
|---|---|---|---|---|
| BokehDiff | 8/9 | 7/9 | – | – |
| post-hoc blur (see caveat) | 8/9 | 9/9 | p = 1.00 | −0.10 [−0.13, −0.08] |
| NA-RFM m0.5 late (best) | 5/9 | 0/9 | p = 0.38 | −0.03 [−0.05, −0.01] |
| TFG m0.5 late (best) | 4/9 | 1/9 | p = 0.13 | −0.05 [−0.08, −0.03] |

- "Responds and preserved" is not the same as reaching the 3 px target. The best steering arms respond in at most
  5/9 scenes and reach the target in at most 1/9. Their scene change is lower largely because they blur less.
- **Statistics:** 26 arms were each compared with BokehDiff by an exact sign test; **no multiple-comparison
  adjustment** was applied. With 9 scenes, the smallest attainable p is 0.0078. After a Bonferroni or Holm
  adjustment over 26 tests, no arm differs significantly from BokehDiff. The late-steering comparisons are
  statistically inconclusive. The practical finding is that they produce less, and less accurately sized, blur.
- **Caveat: the post-hoc baseline is circular under these metrics.** `metrics48.posthoc` disk-blurs the whole
  reference in linear light by exactly r, and composites the subject back through a feathered copy of the
  evaluator's own CLIPSeg subject mask. The same function, at the measured r, produces the "blur-matched reference"
  that scene change is computed against (`Scorer.matched_ref`). Its blur is also the disk model the radius
  measurement fits. So by construction its scene change is 0.000 and its r is 3.00 in every case, and the −0.10
  scene advantage and 9/9 on-target are not evidence. Its one failure (d09) is a centroid shift of 39.9 px. What
  post-hoc blur lacks can only be measured against references that are not post-hoc blur.

**C. TFG.**

*Offline accuracy*, known disk blur, slope / Spearman:

| images | P | P2 | disk fit |
|---|---|---|---|
| dev references | 0.92 / 0.96 | 0.86 / 0.96 | 1.01 / 0.98 |
| in-trajectory x0 estimates | 0.58 / 0.95 | 0.61 / 0.95 | 1.01 / 0.98 |

Both predictors rank blur correctly but under-read its size by about 40% on the intermediate estimates TFG actually
guides on.

*Intervention effectiveness*, finals, medians over 10 cases:

| config | P gain | P2 gain | disk-fit r | DoG ratio | scene |
|---|---|---|---|---|---|
| late m1 | +2.30 px | +1.41 px | 1.14 px (7 measurable) | 1.01 | 0.08 |
| late m2 | +2.44 px | +1.77 px | 1.76 px (5 measurable) | 1.17 | 0.13 |

- **Per guided step:** P +0.10-0.14 px, P2 +0.01-0.08 px, disk-fit +0.13 px, DoG ratio ≈ 1.00.
- **Update spectrum:** 62-78% of the update's energy lies above 0.25 cycles/px, against 73-92% for a known 1 px blur.

*Classification:*
- **E1 = True:** 7/10 (late m1) and 6/10 (late m2) cases gain ≥ 1 px on P with disk-fit gain ≤ half of it.
- **E2 = False:** in-trajectory slope 0.58.
- **E3 = False:** P2 reads 67% of P's gain.
- **E4 = True:** finite-difference relative error 0.18%.

**Recommendation (frozen rule): P is inaccurate on the in-trajectory images, an offline accuracy problem; no ensemble
experiment.**
- The E1 ∧ E4 ∧ ¬E3 branch gives the same answer. The two tested predictors share the failure, so these results do
  not justify the proposed ensemble experiment. P2 has the same architecture and training data source. Two
  predictors failing alike does not rule out differently trained ones, for example predictors trained on noisy
  x0 estimates.
- **Diagnosis:**
  - P is well calibrated on clean images (slope 0.92) but poorly calibrated on the intermediate x0 estimates it
    guides (0.58);
  - the gradients are correct (finite-difference error 0.18%), so TFG correctly optimizes an unreliable signal;
  - the predictor reading rises faster than independently measured blur.
- **Strongest evidence:** the independent disk-fit radius (1.1-1.8 px against a P gain of 2.3-2.4 px). Unchanged
  background contrast (DoG ≈ 1) does not by itself show an absence of blur.
- **Scope:** the tested NA-RFM and TFG configurations showed no advantage over BokehDiff. That is
  - one model (SD 1.5),
  - one NA-RFM layer (mid) with one training procedure,
  - one TFG objective with one predictor architecture,
  - 10 dev scenes.

  This does not rule out other layers, training procedures or guidance objectives.
- **Next direction:** close this pilot and return to BokehDiff improvements. No further steering sweep is proposed.

## Results

Files: `results/` (copied from `outputs/pareto48/` on DSMLP). Generated images, activations, `P.pt`, the RFM directions
and the noise-alignment filters are on DSMLP only, under
`/dsmlp/workspaces-fs05/DSC180A_FA26_001/home/yuz271/DSC180-Capstone/outputs/pareto48/`. All counts are over the 10
eligible dev cases (d02 and d12 excluded, deviation 2).

### 1. Implementation checks

| check | result |
|---|---|
| CamEdit pairs (shard 001) | 955 candidate scenes; **321 usable**: 221 train / 49 val / 51 test scenes. Median measured background blur 4.6 px. Rejected: near region also blurred (577), unmeasurable (193) |
| measurement validation (dev references) | **passed:** known background blur 1-6 px recovered within tolerance in 11/11 measurable cases; shifts and whole-image blur detected; contrast reduction not read as blur. Centroid threshold relaxed to 36.7 px (deviation 3) |
| NA-RFM directions (scene split) | all 3 layers: test AUC 1.00, paired accuracy 1.00; 10 random directions: median AUC ≈ 0.5 |
| TFG predictor P (1.07 M parameters) | **all gates passed:** blur Spearman 0.97; MAE 0.33 px (r ≤ 4); contrast ×0.5 → 0.01 px (vs 3.8 px for r = 4); median filter 1.2 px (< r = 2 response 1.6); category spread < 0.1 px |
| smoke | baseline reproducible; zero-strength NA-RFM bit-identical; hooks fire only at scheduled steps; TFG gradients finite and nonzero; weights unchanged; peak 10.3 / 11.75 GB |
| smoke tolerance | failed the frozen 2/255 tolerance; passed against the measured numerical floor (deviation 4). The floor: splitting the unsteered CFG batch alone changes pixels by up to 40-42/255 (mean 0.4-0.7) |
| bugs found on the pod and fixed before any steered output | validation key clash, NaN disk kernel for tiny radii in P's renderer, TFG device type |

### 2. Development findings (10 eligible dev cases)

**Reading the table:**
- **Blur response:** measured added background blur ≥ 1 px, or (unmeasurable and background detail ≤ 0.8 of the
  reference).
- **On target:** within 3 ± 1.33 px.
- **Preserved:** subject identity, mask, position, sharpness and colour kept, and scene change ≤ 0.15 against the
  blur-matched reference.
- **Gate for confirmation:** ≥ 7/10 responding **and** ≥ 7/10 responding ∧ preserved.

| arm | responds | on target | preserved | responds ∧ preserved | median added blur (px, measurable n) | median scene change | s / image |
|---|---|---|---|---|---|---|---|
| post-hoc 3 px disk blur (anchor) | 10 | 10 | 9 | 9 | 3.01 (10) | 0.000 | 0.1 |
| **SD + BokehDiff** (round 45 controller; d10 controller-ineligible) | 9/9 | 7/9 | 8/9 | **8/9** | 3.15 (9) | 0.102 | ~3 (render) |
| NA-RFM mid_block, m 0.5, late third (best ∧) | 5 | 0 | 10 | **5** | 1.02 (9) | 0.072 | 5 |
| NA-RFM, m 1, late third | 3 | 0 | 7 | 3 | 1.00 (6) | 0.116 | 5 |
| NA-RFM, m 1, all steps (selected P and B) | 7 | 0 | 0 | 0 | unmeasurable (0) | 0.470 | 7 |
| NA-RFM, m 2, all steps | 8 | 0 | 0 | 0 | unmeasurable (0) | 0.550 | 7 |
| random direction, m 1, all steps | 8 | 0 | 0 | 0 | 1.25 (3) | 0.251 | 7 |
| TFG, m 0.5, late third (best ∧) | 4 | 1 | 10 | **4** | 1.03 (7) | 0.052 | 37 |
| TFG, m 1, late third | 4 | 1 | 9 | 4 | 1.15 (7) | 0.084 | 37 |
| TFG, m 2, late third | 3 | 3 | 7 | 3 | 1.73 (5) | 0.129 | 37 |
| TFG, m 0.5-2, all steps | 3-6 | 0-1 | 0-1 | 0-1 | 1.1-2.0 (1) | 0.50-0.56 | 107 |

The full table of all 27 arms, with subject DINOv2, mask IoU, ΔE, depth correlation and UNet forward/backward counts,
is in `results/report.md`. Note that its counts include the two excluded cases.

- **Pareto plot:** `results/pareto_dev.png`. No steering configuration lies near BokehDiff or the post-hoc anchor.
  Every configuration either changes the scene a lot (early or all steps; scene change 0.23-0.65) or delivers about
  1 px instead of 3 (late third; on target in ≤ 3/10).
- **Grids:** `results/grid_dev_best.jpg` (best scene-preserving and selected configurations next to BokehDiff) and
  `results/grid_dev_*.jpg`.
- **NA-RFM:**
  - Layer screen: mid_block 2/4 > attention 1/4, late 1/4.
  - Only the all-steps configurations respond in ≥ 7/10, and they do it by **regenerating the scene**: the cat's pose
    changes, the woman becomes a different person, the dog is reframed. The disk fit cannot even align them to the
    reference (0/10 measurable).
  - **Random-direction control at the selected configuration:** it responds just as often (8/10 vs 7/10) with *less*
    scene change (0.25 vs 0.47). At this setting the learned blur direction does no better than a random one.
  - Late-third steering preserves the scene (10/10 at m 0.5) but adds only about 1 px, and 0/10 are on target.
- **TFG:**
  - No configuration responds in ≥ 7/10 ("no repeatable response").
  - Guidance drives **its own predictor** to the target (P's background reading 2.4-3.0 px), but the independent disk
    fit measures 1.0-1.9 px. On the late schedule background detail barely changes (DoG ratio 0.97-1.13; 1 = no
    change).
  - The guidance mostly satisfies the predictor rather than producing visible blur. This held even though P passed
    every validation gate on real photos.
  - TFG costs 37-108 s per image (32-100 UNet backward passes plus VAE-decoder backward passes) vs 5-7 s for NA-RFM.
- **Matched-blur comparison with BokehDiff:** 0 matched cases for the selected NA-RFM configuration (its outputs are
  not alignable). So no preservation comparison at equal blur is possible, and none is claimed.

### 3. Held-out confirmation

**Not run.** Neither method passed the development gate:
- **NA-RFM:** best responds ∧ preserved = 5/10, need 7.
- **TFG:** no configuration with a repeatable response.

The 12 confirmation prompts were never generated, so they remain untouched for a later test.

### 4. Verdicts

- **NA-RFM: no advantage established.** With same-scene training pairs and the full method (hook + x0 CFG + latent
  noise alignment), steering SD 1.5 either blurs by changing the scene or keeps the scene and adds about 1 px. The
  learned direction did not beat a random direction at the setting that responded.
- **TFG: no advantage established.** A predictor that responds to blur and not to contrast on real photos is still
  satisfied during sampling without visible blur. The guided image moves the predictor's reading, not the background.
- **Reference:** SD followed by BokehDiff with the frozen round 45 controller responded and kept the scene in 8/9 dev
  cases, on target in 7/9. That is despite the transfer limitation (upsampled 512-px SD 1.5 images vs the controller's
  native 768-px RealVisXL inputs).

### Limitations

- 10 dev cases, one seed each.
- One checkpoint (SD 1.5) and one shared sampler setting.
- The layer was chosen on 4 cases.
- MIG GPU slice.
- TFG's iter_steps was reduced to 2.
- Two smoke rules were revised after smoke (deviations 4, 5).
- "Preserved" uses automatic measures (CLIPSeg, DINOv2, Depth Anything); the grids agree by eye on the cases shown.

### Remaining questions

- Is TFG's failure predictor exploitation? It could be tested by guiding on one predictor and evaluating with an
  independently trained one.
- Would a late-third NA-RFM edit with a background-masked hook reach 3 px without scene change? Earlier SD rounds found
  masking helps selectivity but not strength.
- Would the random-direction result hold on the late schedule, where the scene is kept?

### Recommended next experiment

**Late-third TFG with exploitation controlled:**
- train 3 predictors on disjoint CamEdit train scenes;
- guide with their mean;
- evaluate with a fourth held-out predictor, plus the same disk-fit and DoG measures;
- strengths m ∈ {2, 4, 8}, on the same 10 dev cases.

**Why:** the late schedule is the only one that keeps the scene, and TFG reached the target on its own predictor
there. So the open question is whether it fails because of predictor exploitation or because late steps cannot create
blur. One run separates the two. If it still delivers ≤ 1.5 px, steering is not worth further effort for this task,
and the project should focus on improving BokehDiff (dead zone, transparency).

Every deviation from the Design is listed below with the moment it was decided.

### Deviations from the frozen design

1. **GPU.** The pod got an A30 **MIG 2g.12gb slice** (11.75 GB, about half an A30's compute), not a full 24 GB A30.
   Smoke peak memory was 10.3 GB, so no OOM retry was needed.
2. **Case eligibility** (decided after seeing the 12 dev references, before any steered output). Cases whose frozen
   regions cannot be scored are excluded from counts:
   - **rule:** subject < 0.5% or background < 2% of the image;
   - **excluded:**
     - **d02:** brick wall at the bicycle's depth, so there is no background region;
     - **d12:** SD drew no pineapple, so there is no subject.
   - **Effect on thresholds:** they scale with the eligible count (8/12 → ⌈2n/3⌉, 6/12 → ⌈n/2⌉; 7/10 for dev).
   - **Effect on smoke and the layer screen:** they use the first eligible dev cases (d01, d03 / d01, d03-d05).
3. **Validation threshold change, per the frozen rule.** Background-only blur at r = 3 moved the CLIPSeg centroid by up
   to 36.7 px on some cases, so the centroid-shift threshold became 36.7 px.
4. **Smoke tolerance** (decided after smoke, before any sweep).
   - **What failed:** the frozen 2/255 (max) / 0.1/255 (mean) tolerance failed for NA-RFM with w = 0 and for TFG at
     zero strength. Measured: mean 0.26-0.84/255, max 16-59/255.
   - **What passed:** zero-strength NA-RFM (s = 0, γ = 0) is bit-identical to the baseline.
   - **Why it differs:** the only differences are numerical — a batch-1 hooked UNet call amplified by s = 4, and TFG's
     own DDIM arithmetic.
   - **Revised tolerance:** measured from the unsteered sampler's own numerical floor (the same CFG computed as two
     batch-1 calls instead of one batch-2 call; `s3b_floor`). Pass if the max and mean differences are ≤ 2× the floor
     (and at least the original tolerance). The original failures are kept in `smoke.json` (`revision`).
5. **TFG strength m0** (decided after smoke, before any sweep).
   - **What went wrong:** the frozen rule (raise P's background reading by ≥ 1 px on both smoke cases) cannot be met
     when a reference already reads above T (d03 reads 4.4 px). Every probe failed, and the rule fell back to m0 = 16.
   - **Why that matters:** the smoke images at m0 = 4 and 16 destroy the subject (subject DINOv2 ≈ 0, mask IoU 0).
   - **Revised rule:** m0 = the smallest probe at which guidance brings P's reading to T ± 0.5 px on both smoke cases.
     The original outcome is kept in `smoke.json` (`revision`).
   - **Smoke observation, for the report:** at every m0, P reads ≈ 3.0 px while the independent DoG evaluator shows no
     background detail loss. The guidance satisfies its own predictor without producing blur the evaluator can see.

## Full layer sweep (Nov 28, 2026): plan, frozen before it ran

**Why.** The pilot gave the full sweep only to the layer winning a 4-scene screen. Mid block scored 2/4, the
attention and late layers 1/4: a one-scene margin. The attention layer, the kind used in related work, never got a
fair test. Dev prompts only; **the 12 confirmation prompts stay ungenerated.**

**What runs** (`sweep48b.py`, output `outputs/pareto48L/`, so the pilot's records stay untouched):
- **Layers:** `attn` (`up_blocks[1].attentions[1]`), `late` (`up_blocks[2].resnets[2]`) and `mid` (`mid_block`,
  rerun so all three share hardware).
- **Grid per layer:** strengths m ∈ {0.5, 1, 2} (the per-layer grids chosen by the pilot's smoke stage, identical
  for all three) × schedules {early, middle, late, all} × 12 dev prompts.
- **Unchanged from the pilot:** the CamEdit RFM directions, the noise-alignment filters and γ0. Nothing is
  retrained.
- **Random-direction control at every configuration:** the same norm-scaled additive hook with a seeded random
  direction of the layer's size (the pilot ran it at the selected configuration only).
- **Hardware floor:** `narfm|mid|m0|all` (s = 0, γ = 0, the unsteered baseline on this GPU), scored against the
  pilot's frozen references.
- **Order** (resumable after a pod limit): learned attn, late, mid; then random attn, late, mid.

**Scoring:** the pilot's frozen references, regions, metrics and rules (`rules48.json`, centroid 36.7 px), on the 10
eligible dev scenes.

**Decision, per layer.** These are the pilot's selection and gate rules, applied to each layer separately:
- **Repeatable response:** a configuration responds in ≥ ⌈2n/3⌉ = 7/10 cases.
- **Gate:** some configuration has response ≥ 7/10 **and** response ∧ preserved ≥ 7/10.
- **Reported, not gated:**
  - on target (|r − 3| ≤ 1.33);
  - learned minus random on "response ∧ preserved" at each configuration (exact sign test, unadjusted; the number
    of tests is stated);
  - median scene change;
  - number of scenes with measurable radius.
- **If a layer passes the gate:** that is a dev result only. Held-out confirmation would need its own frozen
  addendum.

**Figure:**
- x = measured radius minus the 3 px target, on scenes where the radius is measurable;
- y = scene change;
- median and IQR per configuration over the 10 eligible scenes;
- configurations with fewer than 5 measurable scenes in a separate strip, not on the axis;
- learned and random directions shown separately.
