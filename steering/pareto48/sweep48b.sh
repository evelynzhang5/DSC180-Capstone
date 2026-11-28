#!/bin/bash
# Full layer sweep on one DSMLP GPU pod, from the repo root. Resumable: rerun after a pod limit.
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export P48_OUT=$PWD/outputs/pareto48L
nvidia-smi -L
python -m pip install --user -q -r steering/sd_dof/requirements.txt pyarrow || { stamp "pip FAILED"; exit 1; }
bash scripts/setup_narfm.sh || exit 1
python steering/pareto48/sweep48b.py || { stamp "SWEEP FAILED"; exit 1; }
stamp "##### ALL DONE"
