# Findings so far

All numbers are from experiments with rules fixed before each run. Most use generated scenes; real-photo evidence is
noted where it exists.

## Steering Stable Diffusion toward depth of field (SD 1.5)

- **The NA-RFM paper's setup reproduces, but the blur is not selective.**
  - Background sharpness falls 36%, and foreground sharpness falls about as much.
  - Steered images often change objects and layout, not only blur.
- **Other directions:** no tested direction blurs the background more selectively at matched blur (mean difference,
  other RFM eigenvectors).
- **Masking the edit to the background** improves sharpness selectivity, but the scene still changes.
- **Timing:** steering only late in denoising keeps the layout but delivers little blur; steering early blurs but
  changes the scene.
- **Post-hoc blur** of the unsteered image beat every steering variant at both mild and visible blur.
- **Implication for the Pareto study:** expect a clear trade-off between blur and scene change, controlled by strength
  and by which steps are steered.

## Steering trade-off pilot (branch `q2-pareto48`, Nov 25, 2026)

- **Setup.**
  - SD 1.5: 512 px, DDIM 50, CFG 7.5.
  - Both methods are trained on 221 same-scene sharp/background-blurred CamEdit50K pairs, so blur is not confounded with
    content.
  - Target: 3 px of added background blur.
  - Scene change is measured against the reference blurred by the same amount.
  - 10 dev prompts; 12 held-out prompts kept untouched.
- **NA-RFM** (full method: RFM hook, x0 CFG, latent noise alignment; mid_block):
  - late-third steering keeps the scene (10/10) but adds ~1 px;
  - steering early or over all steps blurs by regenerating the scene (scene change 0.47-0.55);
  - a random direction does as well as the learned one there.
- **TFG** (verified upstream code; blur predictor passed its blur-vs-contrast tests): guidance drives the predictor's
  reading to 3 px while the independent measure sees 1-1.9 px and little loss of background detail.
- **SD + BokehDiff** (round 45 controller): blurs and keeps the scene in 8/9, on target in 7/9.
- **Verdict:** no advantage established for either steering method; the held-out confirmation was not run.
- **Next:** late-third TFG guided by an ensemble of predictors and judged by a held-out one, to tell predictor
  exploitation apart from late steps being unable to create blur.

## BokehDiff (one-step diffusion bokeh editor)

**Inputs and controls.**
- BokehDiff takes a sharp photo, a depth map, a focus depth and one blur setting K. Depth enters only through one
  attention block (PISA) inside the UNet.
- In BokehDiff's own setting, K is fixed per image by the user.

**Blur amount.**
- **Scene dependence:** the same K gives very different blur in different photos.
  - Delivered blur ≈ K × (depth gap between background and subject), with a scene-specific **dead zone**: almost no
    blur below a threshold K.
  - Hitting a requested blur size with one fixed K per level works 33% of the time on fresh scenes.
- **Linear rule from the depth gap:** 41%.
- **Monotone curve** (fitted on 8 development scenes, dead zone included): **73%** on 21 fresh scenes.
  - By requested size: 4 px 19% → 86%, 8 px 71% → 67%, 12 px 33% → 67%, against the linear rule.
  - Paired gain over the linear rule: +32 points [+17, +46].
  - No extra subject or colour change, same render time.

**Numerical precision.**
- BokehDiff's released code computes PISA attention in fp16, which makes weak blur jump in steps.
- fp32 attention smooths the low end slightly, but it was not the cause of the weak-blur misses.

**Measurement.**
- Lens blur must be compared in linear light. An sRGB reference made "texture-dependent blur shape" look much worse
  than it is.
- After correcting it, the remaining texture effect is small.

**Things that were not failures.** A suspected sharp halo around subjects was mostly background genuinely near the
subject's depth, plus mask errors.

**Transparent objects (glass, clear plastic, windows).**
- **The failure:** BokehDiff keeps the scene seen through glass sharp. Its depth map marks the glass as solid.
  - Telling it the glass region is far away blurs the background behind the glass but also the glass itself (rim,
    marks, liquid).
  - One depth per pixel cannot describe "sharp surface + far scene" at the same pixel.
- **Steering-style fix: no advantage.** Mixing two runs inside the model (after the PISA block) was 34% worse than
  blending the two finished images. The model's internal features are too coarse (about 16 px per cell) for thin glass
  detail.
- **Layer separation: works on familiar glass.** A small U-Net (4.9M parameters, trained on synthetic glass over about
  300 photos) estimates the glass surface, the background and the mixing fraction. BokehDiff then blurs only the
  background, and the sharp surface is composited back.
  - Error inside the glass: 8.5 (blend) → **5.5** (−36%), better on all 6 unseen backgrounds.
  - With the true mixing fraction given: 2.9.
  - Does **not** generalise yet: on unseen glass patterns (text, dots) it is worse than blending; on the one real
    photo (an acrylic sign) it finds almost no glass.

**Comparison with Bokehlicious (a non-diffusion bokeh renderer).**
- On generated scenes, both hit requested blur equally rarely with one fixed setting (24%).
- Bokehlicious keeps the subject sharp less often (71% vs 89%) and can't produce strong blur.
- On real photos (RealBokeh) it is closer to the real capture.
- About 60× faster.

## Overall pattern

Most problems first blamed on the model's internals came from its inputs (one setting for every photo, one depth per
pixel), its numerical precision, or the measurement. Fixes around the frozen model (calibration, layer separation)
helped; internal interventions we tested did not beat them.

## Where each result lives in the previous repository (`dsc180-diffusion-steering`)

| Topic | Branches |
|---|---|
| SD 1.5 depth-of-field steering, directions, masking, timing | `q2-sd-dof`, `q2-sd-dof-direction`, `q2-sd-dof-masked`, `q2-diagnostics`, `q2-sd-dof-timing` |
| BokehDiff setup and blur calibration history | `q2-bokehdiff-smoke` … `q2-confirm-fp32` |
| Internal interventions on BokehDiff (attention, feature edits, LoRA, steering) | `q2-attn-content` … `q2-norm-freeze` |
| Linear-light measurement correction | `q2-edge-diag` |
| BokehDiff vs Bokehlicious | `q2-compare40` |
| Depth-aware control, precision check, fresh-scene confirmation | `q2-control42`, `q2-precision44`, `q2-confirm45` |
| Transparent objects | `q2-transparent43`, `q2-feature46`, `q2-separate47` |
| Steering trade-off pilot (this repository) | `q2-pareto48` |
