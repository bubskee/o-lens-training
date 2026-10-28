#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LAB=${LAB:-/workspace/o-lens-qwen35-4b}
PY=${PY:-.venv/bin/python}
VPY=${VPY:-/root/olens-vllm-venv/bin/python}
TEACHER="$LAB/oracle/l30-ao-sft-500-v1/checkpoints/step_000500"
ITEMS="$LAB/oracle/l30-distill-items-v1"
AR="$LAB/ar/l30_200k_v3/checkpoints/step_001412"
MERGED=/root/olens-merged/l30-step500
# Modest first bank; do not silently lower counts or change an existing bank.
NTRAIN=${NTRAIN:-128}
NVAL=${NVAL:-32}
stage=${1:?Use capacity, prepare, merge, sample-smoke, sample, select, assemble, student-dry, student-smoke}
split=${2:-train}
case "$split" in train) count=$NTRAIN;; val) count=$NVAL;; *) echo "Invalid split" >&2; exit 2;; esac
SAMPLES="$LAB/oracle/l30-teacher-$split-v1"
SELECT="$LAB/oracle/l30-nnomp-$split-v1"
TARGETS="$LAB/oracle/l30-bullets-$split-v1"
case "$stage" in
  capacity|prepare)
    extra=()
    if [[ "$stage" == capacity ]]; then extra+=(--dry-run); fi
    "$PY" -u oracle/prepare_distill_items.py --layer 30 \
      --data "$LAB/data/olens_200k_l22_l26_l30" --source-config "$AR/run_config.json" \
      --teacher "$TEACHER" --output "$ITEMS" --n-train "$NTRAIN" --n-val "$NVAL" "${extra[@]}"
    ;;
  merge)
    "$PY" -u oracle/merge_teacher_vllm.py --teacher "$TEACHER" --output "$MERGED"
    ;;
  sample|sample-smoke)
    if [[ "$stage" == sample-smoke ]]; then
      count=2
      SAMPLES="$LAB/oracle/l30-teacher-$split-smoke-v1"
    fi
    "$VPY" -u oracle/sample_distill_teacher_vllm.py --teacher "$TEACHER" \
      --items "$ITEMS" --merged "$MERGED" --output "$SAMPLES" --split "$split" \
      --count "$count" --n-samples 64 --max-new-tokens 100 --chunk-items 2 \
      --max-num-seqs 128 --gpu-memory-utilization .85
    ;;
  select)
    "$PY" -u oracle/select_distill_nnomp.py --samples "$SAMPLES" --items "$ITEMS" \
      --ar-checkpoint "$AR" --output "$SELECT" --batch-size 16
    ;;
  assemble)
    "$PY" -u oracle/assemble_distill_bullets.py --selection "$SELECT" --items "$ITEMS" \
      --output "$TARGETS" --min-picks 4 --filter-policy no_unigram
    ;;
  student-dry|student-smoke)
    extra=()
    if [[ "$stage" == student-dry ]]; then extra+=(--dry-run); fi
    "$PY" -u oracle/train_distill_student.py \
      --train "$LAB/oracle/l30-bullets-train-v1" --val "$LAB/oracle/l30-bullets-val-v1" \
      --init-from "$TEACHER" --output "$LAB/oracle/l30-student-smoke-v1" \
      --max-steps 5 --save-every 2 --batch-size 1 --grad-accum 4 "${extra[@]}"
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2;;
esac
