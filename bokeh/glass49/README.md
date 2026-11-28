# Real-glass inventory for the next transparency round (in progress)

CPU-only; no training, no GPU job. Purpose: decide whether RealBokeh holds enough **independent** real transparent
scenes for a quantitative test, or only a descriptive one.

- `inv49.py screen`:
  - CLIP ViT-L/14 glass score for all 4,400 RealBokeh f/22 sources (3,960 train / 220 validation / 220 test);
  - CLIP embeddings;
  - near-duplicate pairs (cosine ≥ 0.93);
  - contact sheets.
- `nnsheet.py`: each hand-picked candidate next to its 3 nearest test-split scenes that round 47 trained on.
- `inv49.py align`: phase-correlation shift and gradient correlation of every aperture against the f/22 source.
- `results/`: `screen.json`, `nn.json`, `cand49.json` (26 candidates), `align.json`. Embeddings and sheets stay on
  DSMLP under `outputs/glass49/`.

**Findings so far (not final):**
- 4,400 scenes collapse to about 2,830 distinct views at cosine 0.93.
- Near-identical pairs: 193 train–test and 18 validation–test.
- **Round 47 trained on every test-split f/22 source except ids 48, 49, 51, 116, 129, 131.** No list was saved;
  this comes from the glob in `train47.background_paths`.
- **Most real glass is at locations round 47 already trained on:**
  - the cinema bar (test 44, 75, 111, 189 and the Coca-Cola sign 48/129);
  - the barn with dark tables (test 161, 178);
  - the mannequin prop (test 12, 132, 174).
- **Plausibly new locations:** the kitchen (water-filter jug, dish rack), a cut-glass vase, a red glass vase, a
  printed glass panel, the glass lift, and campus windows. That is roughly 6–8 location groups, several of them
  reflection-dominated.
- **Still to do:**
  - read `align.json`;
  - check whether each candidate's focus falls on the glass, and whether both surface and background are visible;
  - write the inventory table;
  - decide quantitative vs descriptive.
