# CGA-v2 New-Repo P2 Seed42 Failure: Implementation Audit Plan

## 0. Current decision

Current gate:

```text
Gate-CGA-v2-P2-seed42-reproduction
gate_pass = false
decision = P2_FAIL_IMPL_AUDIT_ALLOWED
```

This is **not yet** a final scientific stop. It means the new repository has failed the first seed42 reproduction gate and must enter a bounded implementation/protocol audit before any further experiment.

Forbidden before audit closes:

```text
seed43/44
NUAA-SIRST
IRSTD-1K
HC-Test / blind / external
threshold search
checkpoint search
seed search
architecture tuning
loss tuning
ablation promotion
```

Allowed:

```text
read-only implementation audit
identity/schema fixes
path/list/hash fixes
metric identity fixes
one bounded rerun of P1/P2 only if the audit finds a concrete implementation/protocol mismatch
```

---

## 1. Why this failure is suspicious

The seed42 result is not a small miss. Both Full and HC-Val moved against CGA:

```text
Full:
  OHEM mIoU = 0.9239998629
  CGA  mIoU = 0.9174775386
  delta = -0.00652

  OHEM Pd = 0.980952381
  CGA  Pd = 0.977777778
  delta = -0.00317

  OHEM FA ppm = 6.8711
  CGA  FA ppm = 9.5827
  delta = +2.71165

HC-Val:
  OHEM mIoU = 0.8556149733
  CGA  mIoU = 0.7582938389
  delta = -0.09732

  OHEM FA ppm = 22.8882
  CGA  FA ppm = 122.0703
  delta = +99.1821
```

The most important diagnostic clue is that **the new OHEM anchor looks very different from the old frozen OHEM anchor**. In the old main-repo seed42 Full summary, OHEM mIoU was around 0.83435, not 0.92400. In the old seed42 HC-Val summary, OHEM mIoU was around 0.60479, not 0.85561.

That means the first suspect is **protocol mismatch**, not CGA theory failure.

Likely mismatch classes:

```text
1. Different train/test/HC-Val split.
2. Different dataset root or list files.
3. Different evaluation split semantics.
4. Different preprocessing or resizing.
5. Different metric implementation.
6. Different baseline training configuration.
7. Different OHEM loss configuration.
8. Different model registry mapping.
9. Different checkpoint epoch or summary identity.
10. CGA loss/aux target bug.
```

---

## 2. Do not continue to seed43/44

P3 multiseed is only meaningful if seed42 P2 is a valid reproduction seed. Since seed42 failed and the OHEM anchor itself looks non-comparable, continuing to seed43/44 would only multiply a possibly wrong protocol.

Decision remains:

```text
NO_P3_BEFORE_P2_IMPL_AUDIT_CLOSES
```

---

## 3. New gate: P2A implementation audit

Add a new gate:

```text
Gate-CGA-v2-P2A-seed42-implementation-audit
```

Output:

```text
docs/internal/cga_v2/gate_p2_impl_audit_NUDT-SIRST/summary.json
```

Allowed decisions:

```text
P2A_IMPL_MISMATCH_FOUND_RERUN_P1_P2_ALLOWED
P2A_NO_IMPL_MISMATCH_STOP_NEW_REPO_CGA_V2_AT_NUDT_SEED42
P2A_AUDIT_INCOMPLETE_STOP
```

---

## 4. Audit checklist

### 4.1 Dataset and split identity

Check and record:

```text
DATASET_NAME
DATASET_DIR
dataset_spec_sha256
train_list_path
test_list_path
hcval_list_path
train_list_sha256
test_list_sha256
hcval_list_sha256
train_count
test_count
hcval_count
train_test_overlap_count
duplicate_item_count
```

Hard check:

```text
The P2 train/test/HC-Val list hashes must match the P1 preflight summary.
```

If historical frozen artifacts are being used only as sanity reference, compare but do not require exact equality. If the new repo intends to reproduce historical evidence, then list/hash mismatch is a blocking mismatch.

### 4.2 Evaluation split identity

Check and record:

```text
Full evaluation list path/hash
HC-Val evaluation list path/hash
number of evaluated images
metric threshold = 0.5
mask policy
image resize policy
prediction resize policy
```

Fail if:

```text
Full or HC-Val summary does not declare dataset=NUDT-SIRST
summary does not declare seed=42
summary does not declare threshold=0.5
summary does not declare checkpoint_epoch=400
summary does not declare model identity
```

### 4.3 OHEM anchor identity

Check OHEM:

```text
model_name = MSHNetOHEM
loss_name = OHEM-compatible loss
auxiliary heads disabled
CGA target generation disabled
checkpoint path under results/official/MSHNetOHEM/seed42/NUDT-SIRST/
checkpoint epoch = 400
```

If OHEM is much stronger than historical OHEM, that is not automatically bad, but it must be explained by split/protocol differences. If no explanation exists, mark protocol mismatch.

### 4.4 CGA identity

Check CGA:

```text
model_name = MSHNetCGA
base segmentation path active
CGA auxiliary heads active only in training
eval output uses final_logit only
sigmoid + threshold 0.5
no auxiliary output in test/evaluate
```

### 4.5 CGA loss and target audit

Check:

```text
lambda_center
lambda_boundary
lambda_scale
lambda_peak
loss terms present and finite
target maps non-empty
target maps same spatial size as output or correctly resized
scale target normalized
boundary target not covering full mask
peak target sparse
center target finite
```

### 4.6 Training determinism and checkpoint identity

Check:

```text
seed = 42
torch/cuda/cudnn seeds set
deterministic flags recorded
epoch = 400
checkpoint loaded for eval is epoch400
checkpoint hash recorded
training config hash recorded
```

### 4.7 Metrics identity

Check:

```text
mIoU
nIoU
Precision
Recall / Pd
FA_ppm
F1
component FP
```

Confirm all are computed with the same foreground threshold and same binary mask policy.

---

## 5. Code modifications

### 5.1 Add audit plan

```text
docs/internal/cga_v2/gate_p2_impl_audit_plan.json
```

Required fields:

```json
{
  "gate": "Gate-CGA-v2-P2A-seed42-implementation-audit",
  "dataset": "NUDT-SIRST",
  "seed": 42,
  "threshold": 0.5,
  "checkpoint_epoch": 400,
  "candidate": "MSHNetCGA",
  "baseline": "MSHNetOHEM",
  "audit_scope": [
    "dataset_split_identity",
    "eval_split_identity",
    "ohem_anchor_identity",
    "cga_identity",
    "cga_loss_target_identity",
    "checkpoint_identity",
    "metric_identity"
  ]
}
```

### 5.2 Add audit tool

```text
tools/official/check_cga_v2_p2_impl_audit.py
```

Responsibilities:

```text
load P1 preflight summary
load P2 seed42 summary
load OHEM Full / HC-Val summaries
load CGA Full / HC-Val summaries
validate dataset/seed/threshold/epoch/model identity
validate list hashes and summary hashes
compare OHEM anchor against historical sanity range if configured
emit machine-readable audit summary
```

### 5.3 Add audit script

```text
scripts/official/run_cga_v2_p2_impl_audit.sh
```

Script rules:

```bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

"${PYTHON}" -m tools.official.check_cga_v2_p2_impl_audit \
  --dataset_name "${DATASET_NAME:-NUDT-SIRST}" \
  --seed "${SEED:-42}" \
  --p1_summary "docs/internal/cga_v2/dataset_preflight/NUDT-SIRST/summary.json" \
  --p2_summary "docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/summary.json" \
  --output "docs/internal/cga_v2/gate_p2_impl_audit_NUDT-SIRST/summary.json"
```

### 5.4 Add tests

```text
tests/test_cga_v2_p2_impl_audit.py
```

Test cases:

```text
passes on well-formed summaries
fails if dataset mismatch
fails if seed mismatch
fails if threshold mismatch
fails if checkpoint epoch mismatch
fails if model identity mismatch
fails if P2 list hashes differ from P1
flags OHEM anchor anomaly
blocks P3 if P2A is not closed
```

### 5.5 Strengthen P3 guard

Modify:

```text
scripts/official/run_cga_v2_dataset_multiseed_train_eval.sh
```

Before running seed43/44, require:

```text
P2 gate_pass == true
OR
P2A decision == P2A_IMPL_MISMATCH_FOUND_RERUN_P1_P2_ALLOWED followed by a new P2 gate_pass == true
```

Otherwise stop.

---

## 6. Interpretation rules after P2A

### Case A: implementation/protocol mismatch found

Examples:

```text
wrong split
wrong HC-Val list
wrong dataset root
wrong model registry
wrong OHEM loss
wrong eval threshold
wrong checkpoint
eval using aux output
metric mismatch
```

Decision:

```text
P2A_IMPL_MISMATCH_FOUND_RERUN_P1_P2_ALLOWED
```

Allowed:

```text
fix only the identified mismatch
rerun P1
rerun P2 seed42 once
```

Forbidden:

```text
change model structure
change loss weights
change threshold
change checkpoint
search seed
run seed43/44 before new P2 passes
```

### Case B: no mismatch found

Decision:

```text
P2A_NO_IMPL_MISMATCH_STOP_NEW_REPO_CGA_V2_AT_NUDT_SEED42
```

Meaning:

```text
The new repo implementation does not reproduce the CGA-v2 positive signal.
CGA-v2 cannot be used as AAAI main-method evidence from this new repo.
```

Allowed:

```text
keep repo as implementation / negative-analysis package
write failure analysis
```

Forbidden:

```text
seed43/44
ablation
NUAA/IRSTD rescue
paper main claim
architecture/loss/threshold/checkpoint tuning
```

### Case C: audit incomplete

Decision:

```text
P2A_AUDIT_INCOMPLETE_STOP
```

Do not run more experiments.

---

## 7. Minimal commands

```bash
cd /home/ly/AAAI/cga_v2_repo_grade_code_overlay

PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON}" -m py_compile \
  tools/official/check_cga_v2_p2_impl_audit.py

"${PYTHON}" -m pytest \
  tests/test_cga_v2_p2_impl_audit.py -q

bash -n scripts/official/run_cga_v2_p2_impl_audit.sh

git diff --check

DATASET_NAME=NUDT-SIRST \
SEED=42 \
bash scripts/official/run_cga_v2_p2_impl_audit.sh
```

---

## 8. Key conclusion

Do not treat this as immediate model failure yet. The OHEM anchor itself differs strongly from the old frozen protocol, so the next correct step is:

```text
P2A implementation/protocol audit
```

Only if P2A finds no mismatch should you stop CGA-v2 as an AAAI main-method route in the new repo.
