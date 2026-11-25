# DSC180 Capstone: controllable depth of field with diffusion models

**Question.** When you change how blurred a photo's background is, how much of the scene changes along with it? And
where should that control live:
- steering a diffusion model while it generates;
- a dedicated diffusion-based bokeh editor (BokehDiff);
- or fixes around a frozen editor (calibrating its inputs, decomposing and compositing its outputs)?

**Plan** (following our mentors' advice):
1. **Steering trade-off (main).** Steer Stable Diffusion toward background blur with NA-RFM and TFG, sweeping the
   strength and which denoising steps are steered. Plot blur achieved against scene change (a Pareto front), and
   compare with BokehDiff applied to the unsteered image.
2. **Improving BokehDiff.** Blur calibration and transparent-layer separation (below).
3. **Steering versions of those fixes** (lower priority).

## What we already have

| Finding | Result | Code |
|---|---|---|
| Blur calibration: a depth-aware monotone curve picks BokehDiff's blur setting K per image | Target attainment 33% (BokehDiff, one fixed K) → 41% (linear rule) → **73%** on 21 fresh generated scenes, same render cost | `bokeh/controller/` |
| Transparent-layer separation: a small CNN splits glass surface / background / mixing fraction, then frozen BokehDiff blurs only the background | **−36%** error vs blending two renders on unseen synthetic backgrounds; does not yet generalise to new glass patterns or real glass | `bokeh/transparency/` |
| SD 1.5 depth-of-field steering (RFM / NA-RFM directions, masking, timing) | Steering softens foreground and background alike; masking helps sharpness selectivity but the scene still changes; late steering keeps the layout but gives little blur | `steering/` |

Details and numbers: [docs/findings.md](docs/findings.md). Figures: `figures/`.

## Layout

| Folder | Contents |
|---|---|
| `steering/` | SD 1.5 depth-of-field steering: NA-RFM paper setup and RFM direction (`sd_dof/`), alternative directions, masked edits, timing windows, shared diagnostics. Starting point for the Pareto-front experiment |
| `bokeh/common/` | Shared BokehDiff tools: lens-blur estimator (linear light), measurement regions, preservation measures, scene generation |
| `bokeh/controller/` | Blur calibration: depth gap, linear and monotone rules (`results/rule45.json` is the frozen monotone curve), fresh-scene test |
| `bokeh/transparency/` | Transparent-layer separation: glass synthesis, U-Net separator, composites, BokehDiff renders, scoring |
| `docs/` | Findings, DSMLP workflow, BokehDiff setup notes |
| `scripts/` | `setup_narfm.sh`: clones NA-RFM and TFG at fixed commits |

## Running

- All experiments run on a DSMLP GPU pod (see [docs/dsmlp.md](docs/dsmlp.md)); locally, only CPU checks and analysis of
  saved outputs.
- BokehDiff scripts run inside BokehDiff's own repo and environment (see [docs/bokehdiff.md](docs/bokehdiff.md)).
- Steering scripts need `bash scripts/setup_narfm.sh` first.

History: the full experiment record (every test, pre-set decision rules, results) is in the previous repository,
`dsc180-diffusion-steering`, one branch per experiment.
