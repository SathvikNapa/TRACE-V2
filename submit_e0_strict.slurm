#!/bin/bash
#SBATCH --job-name=e0_strict
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --account=a100acct
#SBATCH --array=0-15
#SBATCH --output=/export/fs05/snapaug1/naturalness_experiments/naturalness_runs/TRACE_V2_e0_strict/slurm_%A_%a.out

# E0 completion: 4 baselines x 4 feature variants = 16 jobs on the strict_v2
# split, one per array task. Mirrors sweep_ablation.sh's exact hyperparameters
# (the ones that produced the legacy_v1 16-cell reference table).
set -euo pipefail

PY=/home/snapaug1/spear/spear_venv/bin/python3
SCRIPT=/home/snapaug1/spear/TRACE-V2/train.py
SPLIT_DIR=/home/snapaug1/spear/golden_datasets/cumulative_dataset_v6/strict_v2
EMBED_ROOT=/export/fs05/snapaug1/naturalness_experiments/training_meta/slide3s/embeds_vec
OUT_BASE=/export/fs05/snapaug1/naturalness_experiments/naturalness_runs/TRACE_V2_e0_strict
SHARED_INDEX=/export/fs05/snapaug1/naturalness_experiments/naturalness_runs/SLT_naturalness_sweep/embed_index.pkl

echo "[$(date)] host=$(hostname) task=${SLURM_ARRAY_TASK_ID}"
$PY -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

# Flatten the 4x4 job matrix by SLURM_ARRAY_TASK_ID.
VARIANTS=(speech_only speech_context speech_rel speech_context_rel)
declare -A VARIANT_FLAGS
VARIANT_FLAGS[speech_only]="--speech-only"
VARIANT_FLAGS[speech_context]="--no-rel-text --no-cats"
VARIANT_FLAGS[speech_rel]="--no-context"
VARIANT_FLAGS[speech_context_rel]=""

LABELS=(mlp dyadformer trace_latefusion trace_crossencoder)
MODEL_TYPES=(mlp base_transformer llama crossattn)
EXTRA_ARGS=(
  "--batch-size 1024 --num-epochs 25"
  "--batch-size 512  --num-epochs 15"
  "--batch-size 512  --num-epochs 15"
  "--batch-size 256  --num-epochs 15 --crossattn-layers 2"
)

variant_idx=$(( SLURM_ARRAY_TASK_ID / 4 ))
model_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))
variant="${VARIANTS[$variant_idx]}"
label="${LABELS[$model_idx]}"
model_type="${MODEL_TYPES[$model_idx]}"
extra="${EXTRA_ARGS[$model_idx]}"
vflag="${VARIANT_FLAGS[$variant]}"

out_dir="${OUT_BASE}/${variant}/${label}"
mkdir -p "$out_dir"

if [ -f "${out_dir}/test_metrics.json" ]; then
  echo "[SKIP] ${variant}/${label} -- test_metrics.json already exists."
  exit 0
fi

echo "[START] ${variant}/${label} (--model-type ${model_type})"
set -x
torchrun --nproc_per_node=1 "$SCRIPT" \
  --train-csv   "${SPLIT_DIR}/train.csv" \
  --val-csv     "${SPLIT_DIR}/val.csv" \
  --test-csv    "${SPLIT_DIR}/test.csv" \
  --embed-root  "$EMBED_ROOT" \
  --index-cache "$SHARED_INDEX" \
  --cache-mode  speaker --cache-dtype float16 --num-workers 0 --cache-items 2048 \
  --cache-root  "/tmp/snapaug1/e0_strict_cache_${SLURM_ARRAY_TASK_ID}" \
  --max-seq-len 512 \
  --text-model  sentence-transformers/all-MiniLM-L6-v2 --prebuild-text \
  --text-batch-size 64 --text-max-len 256 --text-fp16 \
  --rel-text-template "{rel_detail}" \
  --embed-noise-std 0.3 --token-mask-prob 0.3 \
  --model-type "$model_type" \
  --out-dir    "$out_dir" \
  $extra $vflag
set +x
echo "[DONE] ${variant}/${label}"
