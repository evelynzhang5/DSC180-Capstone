# Steering Stable Diffusion toward depth of field

Code carried over from our SD 1.5 depth-of-field experiments. It is the starting point for the Pareto-front study:
blur achieved vs scene change, as strength and the steered denoising steps vary, for NA-RFM and TFG, compared with
BokehDiff on the unsteered image.

| Folder | What it does |
|---|---|
| `sd_dof/sd_dof.py` | NA-RFM paper's SD 1.5 depth-of-field setup: collects activations of `unet.down_blocks[2].resnets[-1]` on a shallow/deep depth-of-field photo set, trains the RFM direction (paper code), steers with h ← h + w‖h‖d/‖d‖ inside a sigma window, 50-step DDIM, CFG 7.5; compares RFM, mean-difference and prompt steering |
| `sd_dof_direction/` | Alternative directions at matched background blur |
| `sd_dof_masked/` | Edit restricted to the background (subject mask) |
| `direction_pilots/`, `diagnostics/` | Shared helpers: subject segmentation, preservation scorer (subject RMS, SSIM, DINOv2), post-hoc blur comparison |
| `sd_dof_timing/timing.py` | Same edit applied only in early / middle / late steps, strength tuned to matched background change |

**Setup.**
- `bash scripts/setup_narfm.sh` (NA-RFM and TFG code into `downloads/`).
- Scripts write to `outputs/`. Some later scripts load directions saved by earlier ones (e.g. `sd_dof_masked` expects
  `outputs/q2_sd_dof_direction/directions.pt`), so run them in order or copy those files from the previous repo's
  outputs on DSMLP.

**Measuring the trade-off.**
- **Blur:** `bokeh/common/blur40.py` measures the delivered background blur (disk fit in linear light) against the
  unsteered image.
- **Scene change:** what the blur fit cannot explain, plus change on the subject (`steering/diagnostics`,
  `bokeh/common/measures40.py`).
