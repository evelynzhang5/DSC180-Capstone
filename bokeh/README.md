# BokehDiff tools

| Folder | Contents |
|---|---|
| `common/` | `blur40.py` (disk-blur estimator, linear light), `measures40.py` (subject preservation, boundary measures), `regions40.py` (frozen background tiles), `common40.py` (shared constants and I/O), `measure18.py` (blur kernels), `gen40.py`/`prompts40.py` (scene generation with RealVisXL), `review40.json`, `results/stage3/mapping.json` (development responses) |
| `controller/` | Blur calibration. `fit42.py` (depth gap; linear rule `results/rule42.json`), `fit45.py` (monotone curve from fp32 development curves `results/score44_dev_curves.json` → `results/rule45.json`, frozen), `gen45.py`/`prompts45.py` (fresh scenes), `plan45.py` (K per arm), `bd45.py` (fp32 renders), `score45.py`, `dsmlp45.sh` |
| `transparency/` | Glass layer separation. `compose46.py` (benchmark composites), `synth47.py` (procedural glass), `train47.py` (U-Net separator), `build47.py`, `infer47.py`, `bd47.py`, `score47.py`, `dsmlp47.sh` |

**Folder paths and filenames.**
- Folder paths were updated for this layout; filenames keep their original numbers.
- `transparency/dsmlp47.sh` reads benchmark data from the previous repo's outputs on DSMLP (`OLD_REPO`, default
  `$HOME/dsc180-diffusion-steering`).
