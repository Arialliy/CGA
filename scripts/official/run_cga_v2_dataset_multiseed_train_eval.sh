#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
: "${DATASET_DIR:=datasets}"
: "${DATASET_NAME:=NUDT-SIRST}"
: "${EPOCHS:=400}"
: "${EPOCH:=${EPOCHS}}"
: "${OUTPUT_DIR:=results/official}"

SEED42_SUMMARY="docs/internal/cga_v2/gate_p2_seed42_${DATASET_NAME}/summary.json"
if [[ ! -f "${SEED42_SUMMARY}" ]]; then
  echo "Missing ${SEED42_SUMMARY}; run P2 seed42 first." >&2
  exit 1
fi
"${PYTHON}" - "$SEED42_SUMMARY" <<'PY'
import json
import sys

p2 = json.load(open(sys.argv[1], encoding="utf-8"))
if p2.get("gate_pass") is not True:
    raise SystemExit("Latest P2 seed42 gate did not pass; stop before P3.")
PY

for seed in 43 44; do
  P2_OUTPUT="docs/internal/cga_v2/gate_p2_seed${seed}_${DATASET_NAME}/summary.json" \
    DATASET_DIR="${DATASET_DIR}" DATASET_NAME="${DATASET_NAME}" SEED="${seed}" \
    EPOCHS="${EPOCHS}" EPOCH="${EPOCH}" OUTPUT_DIR="${OUTPUT_DIR}" \
    bash scripts/official/run_cga_v2_dataset_one_seed_paired_train_eval.sh || true
done

SEED_SUMMARIES=(
  "docs/internal/cga_v2/gate_p2_seed42_${DATASET_NAME}/summary.json"
  "docs/internal/cga_v2/gate_p2_seed43_${DATASET_NAME}/summary.json"
  "docs/internal/cga_v2/gate_p2_seed44_${DATASET_NAME}/summary.json"
)
for path in "${SEED_SUMMARIES[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${path}; cannot aggregate P3." >&2
    exit 1
  fi
done

"${PYTHON}" -m tools.official.summarize_cga_v2_multiseed \
  --dataset_name "${DATASET_NAME}" \
  --epoch "${EPOCH}" \
  --threshold 0.5 \
  --baseline MSHNetOHEM \
  --candidate MSHNetCGA \
  --seed_summaries "${SEED_SUMMARIES[@]}" \
  --output "docs/internal/cga_v2/gate_p3_multiseed_${DATASET_NAME}/summary.json"
