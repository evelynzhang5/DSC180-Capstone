#!/bin/bash
# Pareto pilot on one DSMLP GPU pod, from the repo root:  bash steering/pareto48/dsmlp48.sh
# Resumable: every stage is skipped if its manifest validates. Outputs persist in outputs/pareto48 (workspace).
# Order: environment, data, references and validation, components, smoke (stops dependents on failure), layer screen,
# sweeps, selection (frozen), BokehDiff on dev, confirmation gate, confirmation (+ BokehDiff), report.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export P48_OUT=$REPO/outputs/pareto48
W17=$HOME/bokehdiff-smoke
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
D=$REPO/steering/pareto48
mkdir -p "$P48_OUT"
nvidia-smi -L; nvidia-smi --query-gpu=memory.total --format=csv
python -m pip install --user -q -r steering/sd_dof/requirements.txt pyarrow || { stamp "pip FAILED"; exit 1; }
bash scripts/setup_narfm.sh || { stamp "setup_narfm FAILED"; exit 1; }

run() { python "$D/run48.py" "$@"; }

bokeh() {   # $1 = dev | confirm
  local S=$1 DATA=$P48_OUT/bokeh/$1/data OUTB=$P48_OUT/bokeh/$1/run
  if [ -f "$OUTB/bd48.json" ]; then stamp "BokehDiff $S: already rendered"; return 0; fi
  mkdir -p "$OUTB"
  source "$W17/venv/bin/activate"
  cd "$W17/repo" && git checkout -q $COMMIT
  stamp "BokehDiff $S: prepare_data (Depth Anything V2 Base + BiRefNet)"
  python prepare_data.py --root "$DATA" 2>&1 | grep -v -i -e warn -e "it/s" | tail -3
  python "$REPO/bokeh/common/regions40.py" "$DATA" 2>&1 | tail -3
  FIRST=$(ls "$DATA/input" | head -1)
  B48_DATA=$DATA B48_OUT=$OUTB B48_T=3 python "$D/bd48.py" --test_data_dir "$DATA/input/$FIRST" \
      --output_dir "$OUTB/unused" --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 \
      | grep -v -i -e warn -e "it/s" | tee "$OUTB/log_bd48.txt" | tail -4
  deactivate; cd "$REPO"
  [ -f "$OUTB/bd48.json" ] || { stamp "BokehDiff $S FAILED"; return 1; }
}

stamp "stages 0-4"
run s0_env s1_data s1_refs s1_validate || { stamp "TERMINAL: data, references or measurement validation failed"; run s7_report; exit 1; }
run s2_narfm || stamp "NA-RFM component stage crashed (TFG continues; smoke will mark NA-RFM invalid)"
run s2_tfg || stamp "TFG component stage crashed (NA-RFM continues; smoke will mark TFG invalid)"
run s3_smoke || { stamp "TERMINAL: smoke stage crashed"; run s7_report; exit 1; }
run s4a_layers s4b_sweep s4c_select bprep_dev || { stamp "TERMINAL: sweep or selection failed"; run s7_report; exit 1; }
bokeh dev || stamp "BokehDiff dev failed (comparison will be reported missing)"
run bscore_dev || stamp "BokehDiff dev scoring failed"
run s6_gate s6_confirm || stamp "confirmation stage failed"
if [ -f "$P48_OUT/confirm_go" ]; then
  run bprep_confirm && bokeh confirm && run bscore_confirm || stamp "BokehDiff confirm failed"
fi
run s7_report
stamp "##### ALL DONE"
