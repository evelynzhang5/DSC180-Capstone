#!/bin/bash
# Audit of round 48 on one DSMLP GPU pod, from the repo root: bash steering/pareto48/audit48.sh
# Dev cases only; the confirmation prompts are never generated. Resumable (stage manifests audit_*).
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export P48_OUT=$PWD/outputs/pareto48
nvidia-smi -L
python -m pip install --user -q -r steering/sd_dof/requirements.txt pyarrow || { stamp "pip FAILED"; exit 1; }
bash scripts/setup_narfm.sh || exit 1
python steering/pareto48/audit48.py numerics p2 trace analyze || { stamp "AUDIT FAILED"; exit 1; }
stamp "##### ALL DONE"
