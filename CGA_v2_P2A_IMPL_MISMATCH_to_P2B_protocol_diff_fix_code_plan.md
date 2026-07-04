# CGA-v2 P2A Implementation-Mismatch After Seed42 Failure: P2B Protocol-Diff Audit and One-Time Fix Plan

## 0. Current status

Current repository:

```text
/home/ly/AAAI/cga_v2_repo_grade_code_overlay
https://github.com/Arialliy/CGA
```

Latest machine decision:

```text
Gate-CGA-v2-P2-seed42-reproduction
  gate_pass = false
  decision = P2_FAIL_IMPL_AUDIT_ALLOWED

Gate-CGA-v2-P2A-seed42-implementation-audit
  audit_complete = true
  decision = P2A_IMPL_MISMATCH_FOUND_RERUN_P1_P2_ALLOWED
  mismatches:
    - ohem_full_miou_outside_historical_sanity
    - ohem_hcval_miou_outside_historical_sanity

P3 guard:
  allowed = false
  reason = P2A_MISMATCH_FOUND_BUT_RERUN_P2_NOT_PASS
```

Current seed42 new-repo numbers:

```text
Full:
  OHEM mIoU = 0.9239998629
  CGA  mIoU = 0.9174775386
  delta mIoU = -0.00652
  delta Pd   = -0.00317
  delta FA   = +2.71165 ppm
  delta F1   = -0.00354

HC-Val:
  OHEM mIoU = 0.8556149733
  CGA  mIoU = 0.7582938389
  OHEM FA ppm = 22.8882
  CGA  FA ppm = 122.0703
```

Historical frozen sanity reference from the old main repo:

```text
Historical seed42 Full:
  OHEM mIoU ≈ 0.83435
  CGA  mIoU ≈ 0.88850

Historical seed42 HC-Val:
  OHEM mIoU ≈ 0.60479
  CGA  mIoU ≈ 0.62651
```

The abnormal part is not only that CGA is worse. The stronger signal is that the new-repo **OHEM anchor** is far outside historical sanity. A Full OHEM mIoU of 0.924 and HC-Val OHEM mIoU of 0.856 indicate that the new repo is probably not evaluating the same protocol as the historical frozen NUDT protocol.

Therefore, the next step is **not** model tuning. The next step is a protocol-diff audit and one-time protocol fix.

---

## 1. Formal decision

```text
Decision:
  PROCEED_TO_P2B_PROTOCOL_DIFF_AUDIT_ONLY

Allowed:
  Inspect and fix implementation/protocol identity mismatches found by P2A.
  Rerun P1/P2 seed42 at most once after a documented fix.

Forbidden:
  seed43/44
  NUAA-SIRST
  IRSTD-1K
  architecture tuning
  loss tuning beyond restoring intended frozen config
  threshold search
  checkpoint search
  seed search
  ablation winner promotion
  changing the P2 decision thresholds after seeing results
```

If P2B identifies a concrete mismatch and fixes it:

```text
P2B_IMPL_MISMATCH_PATCHED_RERUN_P1_P2_ONCE
```

If P2B does not find a concrete mismatch:

```text
P2B_NO_ACTIONABLE_IMPL_MISMATCH_STOP_NEW_REPO_CGA_V2_AT_NUDT_SEED42
```

If the one-time rerun still fails:

```text
STOP_NEW_REPO_CGA_V2_AT_NUDT_SEED42_REPRODUCTION
```

---

## 2. Why this is not yet a scientific failure of CGA-v2

The new repo is supposed to regenerate its own evidence, because historical frozen results cannot be used as current-repo results. That rule is correct. However, the new repo first has to prove it is running the **same intended benchmark protocol**.

The historical paper route was explicitly narrow-claim and evidence-first, not a new-SOTA packaging route. It required the new overlay repo to generate its own NUDT artifacts, but it also assumed the dataset protocol, evaluation identity, checkpoint identity, and metric definitions were consistent with the intended experiment.

Here the OHEM anchor itself is too strong compared with the historical anchor, especially on HC-Val. That strongly suggests one or more of the following:

```text
1. split mismatch
2. HC-Val list mismatch
3. dataset root mismatch
4. item-id parsing mismatch
5. preprocessing / resize / normalization mismatch
6. eval threshold mismatch
7. metric implementation mismatch
8. OHEM loss config mismatch
9. model registry mismatch
10. checkpoint epoch / summary identity mismatch
```

Until these are ruled out, treating the result as a model failure is premature.

---

## 3. Most likely mismatch sources

### 3.1 Split identity mismatch

This is the first suspect.

The new P1 summary reportedly has:

```text
train = 663
test  = 664
```

Earlier historical development often referenced a different train count, and the frozen historical artifacts must be treated as the identity source for the old protocol. The new repo must record and compare:

```text
train_list_sha256
test_list_sha256
hcval_list_sha256
item counts
train/test overlap
hcval relation to train/test
```

If the split hash differs from the intended frozen split, do not train. Fix the split source first.

### 3.2 HC-Val identity mismatch

The HC-Val split was missing earlier and then had to be restored. That makes HC-Val identity the second highest-risk issue.

P2B must verify:

```text
hcval_list_path
hcval_list_sha256
hcval item count
hcval train overlap
hcval policy: independent split or test subset
source note
created before seed42 training
```

If HC-Val is not the same frozen split, the new HC-Val result cannot be compared to the historical sanity anchor.

### 3.3 Metric implementation mismatch

The new repo must prove that these fields mean the same thing as before:

```text
mIoU
nIoU
Precision
Pd
FA_ppm
F1
FP components
```

Potential drift points:

```text
- per-image IoU average vs global IoU
- object-level Pd definition
- component matching threshold
- FA ppm denominator
- whether ignored empty images exist
- thresholding rule: >0.5 vs >=0.5
```

### 3.4 OHEM loss/config mismatch

If the baseline is not the intended `MSHNetOHEM`, the paired comparison is invalid.

P2B must record:

```text
model_name
loss_name
OHEM enabled flag
OHEM ratio/top-k/hard mining config
SLS/IoU/BCE components
optimizer
lr schedule
batch size
epoch count
seed
augmentation
```

### 3.5 Inference identity mismatch

The CGA claim depends on:

```text
training-time auxiliary heads only
inference uses final_logit -> sigmoid -> threshold 0.5
```

P2B must verify:

```text
train mode returns final_logit + aux_outputs
eval mode uses final_logit only
aux outputs do not enter test.py / evaluate.py
threshold = 0.5
checkpoint_epoch = 400
```

---

## 4. New Gate: P2B Protocol-Diff Audit

### 4.1 Gate name

```text
Gate-CGA-v2-P2B-protocol-diff-audit
```

### 4.2 Inputs

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/summary.json

docs/internal/cga_v2/gate_p2_impl_audit_NUDT-SIRST/summary.json

docs/internal/cga_v2/dataset_preflight/NUDT-SIRST/summary.json

configs/datasets.yaml
configs/closest_baselines.yaml

results/official/MSHNetOHEM/seed42/NUDT-SIRST/*
results/official/MSHNetCGA/seed42/NUDT-SIRST/*
```

Optional historical reference inputs:

```text
/home/ly/AAAI/OHCM-MSHNet-main/docs/internal/cga_v2/gate_d_seed43_44_NUDT-SIRST/summary.json
/home/ly/AAAI/OHCM-MSHNet-main/docs/internal/cga_v2/gate_c_seed42_train_NUDT-SIRST/eval_full_ohem/summary_metrics.json
/home/ly/AAAI/OHCM-MSHNet-main/docs/internal/cga_v2/gate_c_seed42_train_NUDT-SIRST/eval_hcval_ohem/summary_metrics.json
```

These historical files are sanity references only. They do not become new-repo paper evidence.

### 4.3 Output

```text
docs/internal/cga_v2/gate_p2b_protocol_diff_audit_NUDT-SIRST/summary.json
```

Required fields:

```json
{
  "gate": "Gate-CGA-v2-P2B-protocol-diff-audit",
  "dataset": "NUDT-SIRST",
  "seed": 42,
  "audit_complete": true,
  "p2_decision": "P2_FAIL_IMPL_AUDIT_ALLOWED",
  "p2a_decision": "P2A_IMPL_MISMATCH_FOUND_RERUN_P1_P2_ALLOWED",
  "mismatch_categories": [],
  "actionable_mismatch_found": false,
  "allowed_fix_scope": [],
  "rerun_p1_p2_once_allowed": false,
  "decision": "P2B_NO_ACTIONABLE_IMPL_MISMATCH_STOP_NEW_REPO_CGA_V2_AT_NUDT_SEED42"
}
```

If a concrete mismatch is found:

```json
{
  "actionable_mismatch_found": true,
  "mismatch_categories": ["split_identity_mismatch"],
  "allowed_fix_scope": ["restore_frozen_split_lists_only"],
  "rerun_p1_p2_once_allowed": true,
  "decision": "P2B_IMPL_MISMATCH_PATCHED_RERUN_P1_P2_ONCE"
}
```

---

## 5. Code modifications

### 5.1 Add plan JSON

File:

```text
docs/internal/cga_v2/gate_p2b_protocol_diff_audit_plan.json
```

Suggested content:

```json
{
  "gate": "Gate-CGA-v2-P2B-protocol-diff-audit",
  "purpose": "Identify protocol or implementation identity mismatches after P2 seed42 failure and P2A OHEM-anchor sanity mismatch.",
  "allowed_actions": [
    "read summaries",
    "compare dataset split identity",
    "compare metric identity",
    "compare model/loss/eval identity",
    "produce protocol diff report",
    "allow one P1/P2 rerun only if actionable mismatch is found"
  ],
  "forbidden_actions": [
    "seed43/44",
    "NUAA-SIRST",
    "IRSTD-1K",
    "architecture tuning",
    "loss tuning except restoring intended frozen config",
    "threshold search",
    "checkpoint search",
    "seed search"
  ]
}
```

### 5.2 Add protocol manifest exporter

File:

```text
tools/official/export_cga_v2_protocol_manifest.py
```

Responsibilities:

```text
- read configs/datasets.yaml
- read dataset preflight summary
- read train/test/hcval list files
- compute list sha256
- record model/loss/eval config identity
- record command-line arguments
- record git commit if available
- write protocol_manifest.json into each run directory
```

Required manifest fields:

```json
{
  "dataset": "NUDT-SIRST",
  "seed": 42,
  "model": "MSHNetOHEM or MSHNetCGA",
  "epoch": 400,
  "threshold": 0.5,
  "dataset_dir": "...",
  "dataset_spec_sha256": "...",
  "train_list_sha256": "...",
  "test_list_sha256": "...",
  "hcval_list_sha256": "...",
  "loss_name": "...",
  "model_registry_entry": "...",
  "eval_script": "...",
  "metric_version": "..."
}
```

### 5.3 Add protocol diff checker

File:

```text
tools/official/check_cga_v2_p2b_protocol_diff_audit.py
```

Required checks:

```text
1. P2 summary identity
   - gate == Gate-CGA-v2-P2-seed42-reproduction
   - dataset == NUDT-SIRST
   - seed == 42
   - threshold == 0.5
   - epoch == 400

2. P2A summary identity
   - gate == Gate-CGA-v2-P2A-seed42-implementation-audit
   - decision == P2A_IMPL_MISMATCH_FOUND_RERUN_P1_P2_ALLOWED

3. Dataset split identity
   - train/test/hcval list paths exist
   - sha256 present
   - counts present
   - no overlap unless explicitly permitted

4. Summary identity
   - baseline summary says model == MSHNetOHEM
   - candidate summary says model == MSHNetCGA
   - dataset == NUDT-SIRST
   - seed == 42
   - threshold == 0.5
   - epoch == 400

5. Historical sanity comparison
   - report historical OHEM deltas as sanity only
   - do not use historical result as new-repo evidence

6. Decision
   - if actionable mismatch: allow one rerun
   - else stop new-repo CGA-v2 as AAAI main method
```

### 5.4 Add run script

File:

```text
scripts/official/run_cga_v2_p2b_protocol_diff_audit.sh
```

Requirements:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/cga_v2_repo_grade_code_overlay}
cd "${ROOT}"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

"${PYTHON}" -m tools.official.check_cga_v2_p2b_protocol_diff_audit \
  --p2_summary docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/summary.json \
  --p2a_summary docs/internal/cga_v2/gate_p2_impl_audit_NUDT-SIRST/summary.json \
  --preflight_summary docs/internal/cga_v2/dataset_preflight/NUDT-SIRST/summary.json \
  --output docs/internal/cga_v2/gate_p2b_protocol_diff_audit_NUDT-SIRST/summary.json
```

### 5.5 Add tests

File:

```text
tests/test_cga_v2_p2b_protocol_diff_audit.py
```

Test cases:

```text
- rejects missing P2 summary
- rejects wrong P2 gate name
- rejects wrong dataset
- rejects wrong seed
- rejects missing preflight list hashes
- detects OHEM anchor sanity mismatch
- allows rerun only when actionable mismatch exists
- stops when no actionable mismatch exists
```

### 5.6 Strengthen P2/P3 guards

Modify:

```text
scripts/official/run_cga_v2_dataset_one_seed_paired_train_eval.sh
scripts/official/run_cga_v2_dataset_multiseed_train_eval.sh
```

Rules:

```text
P2 rerun allowed only if:
  P2B decision == P2B_IMPL_MISMATCH_PATCHED_RERUN_P1_P2_ONCE
  rerun_p1_p2_once_allowed == true
  rerun marker does not already exist

P3 allowed only if:
  latest P2 summary gate_pass == true
```

Add marker after rerun:

```text
docs/internal/cga_v2/gate_p2b_protocol_diff_audit_NUDT-SIRST/rerun_consumed.json
```

This prevents repeated P2 reruns until a good seed42 is found.

---

## 6. What fixes are allowed if P2B finds a mismatch?

Allowed fixes are limited to restoring intended protocol identity.

Allowed:

```text
- restore correct frozen train/test/hcval list
- correct dataset registry path or item_format
- correct model registry name mapping
- correct OHEM/CGA loss config to intended frozen values
- correct eval threshold to 0.5
- correct checkpoint epoch to 400
- correct train/eval mode bug
- correct summary identity metadata
- correct metric formula if it demonstrably differs from intended definition
```

Forbidden:

```text
- changing architecture
- changing CGA lambdas to improve metrics
- changing threshold
- selecting a different checkpoint
- changing seed
- editing train/test/hcval after seeing results
- removing hard samples
- using NUAA/IRSTD to rescue NUDT failure
```

---

## 7. Minimum execution sequence

```bash
cd /home/ly/AAAI/cga_v2_repo_grade_code_overlay
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON}" -m py_compile \
  tools/official/export_cga_v2_protocol_manifest.py \
  tools/official/check_cga_v2_p2b_protocol_diff_audit.py

"${PYTHON}" -m pytest \
  tests/test_cga_v2_p2b_protocol_diff_audit.py -q

bash -n scripts/official/run_cga_v2_p2b_protocol_diff_audit.sh
bash -n scripts/official/run_cga_v2_dataset_one_seed_paired_train_eval.sh
bash -n scripts/official/run_cga_v2_dataset_multiseed_train_eval.sh

git diff --check

DATASET_NAME=NUDT-SIRST \
SEED=42 \
bash scripts/official/run_cga_v2_p2b_protocol_diff_audit.sh
```

Only if P2B outputs:

```text
P2B_IMPL_MISMATCH_PATCHED_RERUN_P1_P2_ONCE
```

then rerun P1/P2 once:

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_dataset_preflight.sh

DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
bash scripts/official/run_cga_v2_dataset_one_seed_paired_train_eval.sh
```

---

## 8. Interpretation after P2B

### Case A: P2B finds split/protocol mismatch

Fix exactly that mismatch. Rerun P1/P2 once. If P2 passes, continue to P3.

### Case B: P2B finds metric implementation mismatch

Fix metric identity only if the current metric is demonstrably inconsistent with the intended definition. Rerun P1/P2 once.

### Case C: P2B finds no actionable mismatch

Stop.

```text
STOP_NEW_REPO_CGA_V2_AT_NUDT_SEED42_REPRODUCTION
```

No seed43/44. No NUAA/IRSTD. No model tuning.

### Case D: P2B finds mismatch, fix applied, rerun still fails

Stop.

```text
STOP_NEW_REPO_CGA_V2_AT_NUDT_SEED42_AFTER_ONE_PROTOCOL_FIX
```

---

## 9. Why this is the correct next step

The narrow AAAI route is evidence-first. The new repository cannot borrow old results; it has to regenerate its own NUDT artifacts. But before treating the new P2 failure as scientific, the OHEM anchor must be sane.

Since the new OHEM anchor is far outside the historical sanity envelope, the only responsible next step is protocol-diff audit.

The final decision after this step is binary:

```text
If actionable implementation/protocol mismatch exists:
  fix once and rerun P1/P2 once.

If not:
  stop CGA-v2 new-repo AAAI main-method route.
```
