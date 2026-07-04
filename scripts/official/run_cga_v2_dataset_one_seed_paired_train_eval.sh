#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
: "${DATASET_DIR:=datasets}"
: "${DATASET_NAME:=NUDT-SIRST}"
: "${SEED:=42}"
: "${EPOCHS:=400}"
: "${EPOCH:=${EPOCHS}}"
: "${OUTPUT_DIR:=results/official}"
: "${PREFLIGHT_SUMMARY:=docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/summary.json}"
: "${P2_OUTPUT:=docs/internal/cga_v2/gate_p2_seed${SEED}_${DATASET_NAME}/summary.json}"
: "${P2B_SUMMARY:=docs/internal/cga_v2/gate_p2b_protocol_diff_audit_${DATASET_NAME}/summary.json}"
: "${P2B_RERUN_MARKER:=docs/internal/cga_v2/gate_p2b_protocol_diff_audit_${DATASET_NAME}/rerun_consumed.json}"

P2B_RERUN_MODE=0
if [[ "${SEED}" == "42" && -f "${P2_OUTPUT}" ]]; then
  guard_status="$("${PYTHON}" - "$P2_OUTPUT" "$P2B_SUMMARY" "$P2B_RERUN_MARKER" <<'PY'
import json
import sys
from pathlib import Path

p2 = json.load(open(sys.argv[1], encoding="utf-8"))
p2b_path = Path(sys.argv[2])
marker = Path(sys.argv[3])
if p2.get("gate_pass") is True:
    print("P2_ALREADY_PASS")
elif marker.exists():
    print("P2B_RERUN_ALREADY_CONSUMED")
elif not p2b_path.exists():
    print("P2B_REQUIRED")
else:
    p2b = json.load(open(p2b_path, encoding="utf-8"))
    if p2b.get("decision") == "P2B_IMPL_MISMATCH_PATCHED_RERUN_P1_P2_ONCE" and p2b.get("rerun_p1_p2_once_allowed") is True:
        print("ALLOW_P2B_RERUN")
    else:
        print("P2B_NOT_ALLOWING_RERUN")
PY
)"
  case "${guard_status}" in
    ALLOW_P2B_RERUN)
      P2B_RERUN_MODE=1
      FORCE_TRAIN=1
      ;;
    P2_ALREADY_PASS)
      echo "P2 seed42 already passed; refusing to rerun." >&2
      exit 0
      ;;
    *)
      echo "${guard_status}; stop before P2 seed42 rerun." >&2
      exit 1
      ;;
  esac
fi

DATASET_DIR="${DATASET_DIR}" DATASET_NAME="${DATASET_NAME}" OUTPUT="${PREFLIGHT_SUMMARY}" \
  bash scripts/official/run_cga_v2_dataset_preflight.sh

for manifest_model in MSHNetOHEM MSHNetCGA; do
  "${PYTHON}" -m tools.official.export_cga_v2_protocol_manifest \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --seed "${SEED}" \
    --model "${manifest_model}" \
    --epoch "${EPOCH}" \
    --threshold 0.5 \
    --output_dir "${OUTPUT_DIR}" \
    --preflight_summary "${PREFLIGHT_SUMMARY}" >/dev/null
done

split_exists() {
  local split_name="$1"
  "${PYTHON}" - "$DATASET_DIR" "$DATASET_NAME" "$split_name" <<'PY'
import sys
from dataset import split_exists
raise SystemExit(0 if split_exists(sys.argv[1], sys.argv[2], sys.argv[3]) else 1)
PY
}

run_train_eval() {
  local model_name="$1"
  local ckpt="${OUTPUT_DIR}/${model_name}/seed${SEED}/${DATASET_NAME}/${model_name}_${EPOCH}.pth.tar"
  if [[ ! -f "${ckpt}" || "${FORCE_TRAIN:-0}" == "1" ]]; then
    MODEL_NAME="${model_name}" DATASET_DIR="${DATASET_DIR}" DATASET_NAME="${DATASET_NAME}" \
      SEED="${SEED}" EPOCHS="${EPOCHS}" \
      bash scripts/official/run_cga_v2_train_seed.sh --output_dir "${OUTPUT_DIR}"
  fi
  MODEL_NAME="${model_name}" DATASET_DIR="${DATASET_DIR}" DATASET_NAME="${DATASET_NAME}" \
    SEED="${SEED}" EPOCH="${EPOCH}" CHECKPOINT="${ckpt}" SPLIT="test" \
    bash scripts/official/run_cga_v2_test_seed.sh --output_dir "${OUTPUT_DIR}"
  if split_exists hcval; then
    MODEL_NAME="${model_name}" DATASET_DIR="${DATASET_DIR}" DATASET_NAME="${DATASET_NAME}" \
      SEED="${SEED}" EPOCH="${EPOCH}" CHECKPOINT="${ckpt}" SPLIT="hcval" \
      bash scripts/official/run_cga_v2_test_seed.sh --output_dir "${OUTPUT_DIR}"
  fi
}

run_train_eval MSHNetOHEM
run_train_eval MSHNetCGA

BASE_FULL="${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json"
CAND_FULL="${OUTPUT_DIR}/MSHNetCGA/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json"
BASE_HCVAL="${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json"
CAND_HCVAL="${OUTPUT_DIR}/MSHNetCGA/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json"

HCVAL_ARGS=()
if [[ -f "${BASE_HCVAL}" && -f "${CAND_HCVAL}" ]]; then
  HCVAL_ARGS=(--baseline_hcval "${BASE_HCVAL}" --candidate_hcval "${CAND_HCVAL}")
fi

set +e
"${PYTHON}" -m tools.official.summarize_cga_v2_one_seed \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --epoch "${EPOCH}" \
  --threshold 0.5 \
  --baseline MSHNetOHEM \
  --candidate MSHNetCGA \
  --preflight_summary "${PREFLIGHT_SUMMARY}" \
  --baseline_full "${BASE_FULL}" \
  --candidate_full "${CAND_FULL}" \
  "${HCVAL_ARGS[@]}" \
  --output "${P2_OUTPUT}"
summary_status=$?
set -e

if [[ "${P2B_RERUN_MODE}" == "1" ]]; then
  "${PYTHON}" - "$P2B_RERUN_MARKER" "$P2_OUTPUT" "$summary_status" <<'PY'
import json
import sys
from pathlib import Path

marker = Path(sys.argv[1])
marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text(json.dumps({
    "rerun_consumed": True,
    "p2_output": sys.argv[2],
    "summary_exit_code": int(sys.argv[3]),
}, indent=2, sort_keys=True), encoding="utf-8")
PY
fi
exit "$summary_status"
