#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "${ROOT}"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

DATASET_NAME=${DATASET_NAME:-NUDT-SIRST}
SEED=${SEED:-42}

"${PYTHON}" -m tools.official.check_cga_v2_p2b_protocol_diff_audit \
  --p2_summary "docs/internal/cga_v2/gate_p2_seed${SEED}_${DATASET_NAME}/summary.json" \
  --p2a_summary "docs/internal/cga_v2/gate_p2_impl_audit_${DATASET_NAME}/summary.json" \
  --preflight_summary "docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/summary.json" \
  --output "docs/internal/cga_v2/gate_p2b_protocol_diff_audit_${DATASET_NAME}/summary.json" \
  --dataset "${DATASET_NAME}" \
  --seed "${SEED}"
