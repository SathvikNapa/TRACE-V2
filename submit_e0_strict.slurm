#!/bin/bash
#SBATCH --job-name=e0_strict
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=a100acct
#SBATCH --array=0-15%4
#SBATCH --output=/export/fs05/snapaug1/naturalness_experiments/naturalness_runs/TRACE_V2_e0_strict/slurm_%A_%a.out

# E0 completion: 4 baselines x 4 feature variants = 16 jobs on the strict_v2
# split, one per array task.
#
# IMPORTANT hyperparameter correction (after a second, wrong attempt at this
# array): batch_size/num_epochs below were originally copied from
# sweep_ablation.sh's CURRENT contents (1024/25, 512/15, 512/15, 256/15), but
# that does NOT match what actually produced the legacy_v1 reference numbers
# on disk -- read directly from each variant's hyperparams.json, the real
# values are mlp=512/50, dyadformer=256/30, trace_latefusion=256/30,
# trace_crossencoder=128/30 (sweep_ablation.sh must have been edited after
# that run). Running with the wrong (smaller-batch... no, LARGER batch, FEWER
# epochs) config gave ~4x fewer gradient steps than the working recipe, which
# collapsed trace_latefusion/trace_crossencoder to degenerate majority-class
# prediction on every feature variant while the simpler dyadformer baseline
# still found a real solution -- NOT evidence of speaker-identity leakage,
# just severe undertraining of the harder-to-optimize models. Fixed below to
# match the actual hyperparams.json values.
#
# Lessons from the first (cancelled) attempt at this array:
#  - num-workers 0 (copied from sweep_ablation.sh) meant the 16 requested
#    CPUs sat idle while one thread did all data loading synchronously --
#    batches took 40min-20+hrs. Now --num-workers 8, matching cpus-per-task.
#  - Each task previously got its OWN cache dir and all 16 ran concurrently
#    across different nodes, so all 16 redundantly rebuilt the full
#    per-speaker embedding cache from scratch at once, ~16x the NFS load of
#    the legacy sweep (which ran sequentially in one job, one shared cache).
#    Now all tasks share ONE cache dir on NFS (survives across nodes, unlike
#    node-local /tmp) so later tasks reuse what earlier ones already built.
#    train.py's _atomic_npy_save + _acquire_lock already handle concurrent
#    writers to the same cache safely (PID-suffixed tmp file + rename, with
#    a lock around the write) -- this is a supported pattern, not a hack.
#  - %4 throttles the array to 4 concurrent tasks instead of all 16 at once,
#    bounding how hard this hits NFS while still getting real parallelism.
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
  "--batch-size 512  --num-epochs 50"
  "--batch-size 256  --num-epochs 30"
  "--batch-size 256  --num-epochs 30"
  "--batch-size 128  --num-epochs 30 --crossattn-layers 2"
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
  --cache-mode  speaker --cache-dtype float16 --num-workers 8 --cache-items 2048 \
  --cache-root  "${OUT_BASE}/_shared_cache" \
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
