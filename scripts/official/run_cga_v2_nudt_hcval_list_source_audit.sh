#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"

PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

: "${DATASET_DIR:?DATASET_DIR is required}"
: "${DATASET_NAME:=NUDT-SIRST}"
: "${HCVAL_LIST:?HCVAL_LIST is required}"
: "${SOURCE_NOTE:?SOURCE_NOTE is required}"
: "${DATASET_REGISTRY:=configs/datasets.yaml}"

"${PYTHON}" -m tools.official.check_cga_v2_nudt_hcval_list_source \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --candidate_hcval_list "${HCVAL_LIST}" \
  --source_note "${SOURCE_NOTE}" \
  --registry "${DATASET_REGISTRY}" \
  --output "docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/hcval_source_summary.json"
