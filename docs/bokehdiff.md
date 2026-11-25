# BokehDiff notes

**Code and environment.**
- BokehDiff code: <https://github.com/FreeButUselessSoul/bokehdiff>, used at commit
  `38ff9372ab146ab20468832a0424fa0b2888dffd`. Weights: `zcx65535/bokehdiff` (small) on Hugging Face.
- On DSMLP it lives in a venv at `$HOME/bokehdiff-smoke/venv` with the repo at `$HOME/bokehdiff-smoke/repo`.

**How our drivers run it.**
- Our scripts (`bokeh/*/bd*.py`) run from inside BokehDiff's repo.
- They replace `inference_hf.log_validation`, so BokehDiff's `main()` builds the pipeline once, then our code renders
  many images.
- `--test_data_dir` must point at an image that has a depth file: BokehDiff's startup dataset loads it even though we
  don't use it.

**Inputs.**
- A sharp photo. Processing size: short side ≥ 1024, rounded to a multiple of 64.
- `disp_coc` = [disparity, |disparity − focus| × K / 10].
- Disparity: BokehDiff's preprocessing of Depth Anything V2 output (min–max, 3×3 erode/dilate clip, then flipped to the
  training convention, larger = nearer).
- Focus: the median disparity inside the subject mask.

**Internals.**
- Depth enters only through the PISA attention processor in `unet.down_blocks[1].attentions[0]`.
- The pipeline denoises in overlapping 64-latent tiles, one step, then decodes.

**Precision.**
- The released code computes PISA in fp16.
- Our drivers use an fp32 wrapper (inputs cast to float32 with autocast off) and assert the result is float32 inside the
  call.

**Measuring blur.**
- `bokeh/common/blur40.py` fits a disk blur (radius 0–24 px, linear light, gain and offset allowed) of the input to the
  output on background tiles.
- Delivered blur is the median over tiles with fit R² ≥ 0.9.
