# CGA-v2 New Repo: NUDT P1 Preflight Failure and HC-Val List Rescue Plan

## 0. Verdict

Current P1 stop is correct.

```text
Gate: P1 NUDT-SIRST dataset preflight
Decision: STOP_NEW_REPO_CGA_V2_AT_DATASET_PREFLIGHT
Reason: missing_hcval_list
Missing file:
  /home/ly/AAAI/OHCM-MSHNet-main/datasets/NUDT-SIRST/img_idx/hcval_NUDT-SIRST.txt
```

Do **not** start seed42 training yet.

The uploaded `NUDT-SIRST.zip` does **not** solve the P1 failure, because it does not contain `hcval_NUDT-SIRST.txt`. It also should not be used as a replacement dataset.

---

## 1. Uploaded zip inspection

I inspected `/mnt/data/NUDT-SIRST.zip`.

### 1.1 `img_idx` contents

```text
NUDT-SIRST/img_idx/train_NUDT-SIRST.txt       663 items
NUDT-SIRST/img_idx/test_NUDT-SIRST.txt        664 items
NUDT-SIRST/img_idx/test_NUDT-SIRST copy.txt   664 items
```

There is **no**:

```text
NUDT-SIRST/img_idx/hcval_NUDT-SIRST.txt
```

### 1.2 Image / mask file integrity

The zip is not a clean train/eval dataset package:

```text
NUDT-SIRST/images/: 1328 files
  zero-byte files:    1063
  nonzero files:       265

NUDT-SIRST/masks/: 1327 files
  zero-byte files:    1327
  nonzero files:        0
```

Therefore:

```text
Do not unzip this package over the current dataset.
Do not use it as DATASET_DIR for training/evaluation.
Do not treat it as official NUDT-SIRST data.
```

It can only be used as a weak reference for train/test list names if needed. Your current train/test preflight already passed, so the zip is not needed for train/test.

---

## 2. What this means

The current issue is not a model issue and not a loader issue. It is a split-definition issue:

```text
The new repo requires NUDT-SIRST Full + HC-Val evidence.
The dataset root has train/test lists but no frozen HC-Val list.
```

Since the paper claim uses Full and HC-Val, the HC-Val list is not optional for paper evidence.

The current paper positioning is narrow and evidence-first: CGA-v2 should be treated as target-preserving component-geometry regularization, not as a new SOTA or full hard-clutter closure method. Historical artifacts cannot be reused as new-repo evidence; the new repo must regenerate its own NUDT artifacts before paper evidence can be claimed.

---

## 3. Correct next gate

Add a new small gate before seed42:

```text
Gate-CGA-v2-P1A-NUDT-HCVal-List-Source-Audit
```

Purpose:

```text
Recover or register a frozen hcval_NUDT-SIRST.txt source.
Verify it is not derived post hoc from results.
Verify its items are a subset of known NUDT ids.
Verify all listed image/mask files exist and pass strict preflight.
Write split hash and source metadata.
```

Allowed outcomes:

```text
PASS:
  hcval_NUDT-SIRST.txt is restored from a frozen source.
  Proceed to rerun P1 dataset preflight.

FAIL:
  STOP_NEW_REPO_CGA_V2_AT_NUDT_HCVAL_SPLIT_MISSING
  Do not start seed42 paper-evidence training.
```

---

## 4. Acceptable sources for `hcval_NUDT-SIRST.txt`

Allowed:

```text
1. A previously frozen internal split file from OHCM-MSHNet-main or historical experiment artifacts.
2. A committed split file with hash, creation date, and source note.
3. A pre-existing official/internal HC-Val list that was defined before CGA-v2 new-repo results.
```

Not allowed:

```text
1. Creating HC-Val from current model errors.
2. Selecting hard samples after seeing CGA/OHEM outputs in the new repo.
3. Using the entire test list and calling it HC-Val.
4. Copying `test_NUDT-SIRST.txt` to `hcval_NUDT-SIRST.txt` without a documented pre-existing policy.
5. Using the uploaded zip to replace raw images/masks.
```

If no frozen HC-Val list can be found, then the new repo can still run a Full-only smoke/reproduction experiment, but it cannot claim the Full + HC-Val NUDT paper evidence.

---

## 5. Code changes

### 5.1 Update `configs/datasets.yaml`

Keep BasicIRSTD-style `dataset_name` selection. Add explicit split policy for NUDT:

```yaml
datasets:
  NUDT-SIRST:
    root_name: NUDT-SIRST
    image_dir: images
    mask_dir: masks
    list_dir: img_idx
    splits:
      train:
        file: train_NUDT-SIRST.txt
        required: true
      test:
        file: test_NUDT-SIRST.txt
        required: true
      hcval:
        file: hcval_NUDT-SIRST.txt
        required: true
        source_required: true
        claim_role: hard_clutter_validation
    item_format:
      list_has_extension: false
      image_suffix: .png
      mask_suffix: .png
      id_prefix: ""
      id_suffix: ""
      strip_extension_from_list_item: true
    mask_policy:
      type: binary
      allowed_values: [0, 255]
    expected_counts:
      train: null
      test: null
      hcval: null
```

Do not hard-code train/test/hcval counts until preflight writes a frozen summary.

---

### 5.2 Add HC-Val split source audit tool

New file:

```text
tools/official/check_cga_v2_nudt_hcval_list_source.py
```

Required behavior:

```text
Inputs:
  --dataset_dir
  --dataset_name NUDT-SIRST
  --candidate_hcval_list
  --source_note
  --output

Checks:
  candidate_hcval_list exists
  non-empty
  no duplicate ids
  no path traversal
  all items exist in image/mask dirs
  image/mask readable
  image/mask size match
  mask values valid under registry mask_policy
  hcval items do not overlap train unless explicitly allowed by registry policy
  split hash is written
  source_note is non-empty
```

Output example:

```json
{
  "gate": "Gate-CGA-v2-P1A-NUDT-HCVal-List-Source-Audit",
  "gate_pass": true,
  "decision": "NUDT_HCVAL_LIST_SOURCE_ACCEPTED",
  "dataset": "NUDT-SIRST",
  "hcval_list_path": "/.../hcval_NUDT-SIRST.txt",
  "hcval_list_sha256": "...",
  "hcval_count": 0,
  "source_note": "...",
  "source_accepted_before_new_repo_seed42": true,
  "next_allowed_gate": "Gate-CGA-v2-P1-dataset-preflight"
}
```

Failure decision:

```text
STOP_NEW_REPO_CGA_V2_AT_NUDT_HCVAL_SPLIT_MISSING
```

---

### 5.3 Add script

New file:

```text
scripts/official/run_cga_v2_nudt_hcval_list_source_audit.sh
```

Use fail-closed shell style:

```bash
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

"${PYTHON}" -m tools.official.check_cga_v2_nudt_hcval_list_source \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --candidate_hcval_list "${HCVAL_LIST}" \
  --source_note "${SOURCE_NOTE}" \
  --output "docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/hcval_source_summary.json"
```

---

### 5.4 Update dataset preflight

Existing P1 preflight should keep failing if required HC-Val is missing.

Add explicit fields to P1 summary:

```json
{
  "missing_required_splits": ["hcval"],
  "missing_hcval_list": true,
  "train_list_sha256": "...",
  "test_list_sha256": "...",
  "hcval_list_sha256": null,
  "next_allowed_gate": "Gate-CGA-v2-P1A-NUDT-HCVal-List-Source-Audit"
}
```

This makes the next step machine-readable.

---

### 5.5 Add zero-byte guard

Because the uploaded zip contains many zero-byte files, preflight must explicitly reject zero-byte image/mask files.

Add fields:

```json
{
  "zero_byte_images": 0,
  "zero_byte_masks": 0,
  "zero_byte_examples": []
}
```

Any zero-byte item in required split should fail:

```text
STOP_DATASET_DUE_TO_ZERO_BYTE_FILES
```

This prevents accidentally using the uploaded zip as `DATASET_DIR`.

---

### 5.6 Add tests

New tests:

```text
tests/test_cga_v2_nudt_hcval_list_source.py
```

Test cases:

```text
1. missing hcval list -> fail
2. empty hcval list -> fail
3. duplicate ids -> fail
4. path traversal -> fail
5. valid hcval list -> pass and writes sha256
6. zero-byte image/mask in split -> fail
7. copying test list as hcval without source_note -> fail
```

---

## 6. Execution order

### Step 1: Do not use uploaded zip as dataset root

No command should use:

```text
DATASET_DIR=/path/to/unzipped/NUDT-SIRST.zip/NUDT-SIRST
```

until it passes zero-byte preflight. Based on inspection, it will fail.

---

### Step 2: Find or provide frozen HC-Val file

Search likely locations:

```bash
find /home/ly/AAAI -name 'hcval_NUDT-SIRST.txt' -o -name '*hcval*NUDT*.txt'
```

If found, audit it:

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
HCVAL_LIST=/path/to/frozen/hcval_NUDT-SIRST.txt \
SOURCE_NOTE='Recovered from pre-existing frozen HC-Val split before new-repo seed42 training.' \
bash scripts/official/run_cga_v2_nudt_hcval_list_source_audit.sh
```

If audit passes, copy it into the dataset root with manifest:

```bash
cp /path/to/frozen/hcval_NUDT-SIRST.txt \
  /home/ly/AAAI/OHCM-MSHNet-main/datasets/NUDT-SIRST/img_idx/hcval_NUDT-SIRST.txt
```

Then rerun P1:

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_dataset_preflight.sh
```

---

### Step 3: Only after P1 passes, run seed42

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
bash scripts/official/run_cga_v2_dataset_one_seed_paired_train_eval.sh
```

---

## 7. If HC-Val cannot be recovered

If no frozen HC-Val list exists:

```text
Decision:
  STOP_NEW_REPO_CGA_V2_AT_NUDT_HCVAL_SPLIT_MISSING
```

Allowed:

```text
Full-only implementation smoke training
repo release contract tests
negative/limitation note
```

Not allowed:

```text
NUDT Full + HC-Val paper claim
HC-Val result table
AAAI main evidence claim based on HC-Val
creating a new HC-Val split after seeing model outputs
```

---

## 8. Paper claim implication

The paper route remains narrow:

```text
CGA-v2: target-preserving component-geometry regularization for MSHNet-style IRSTD.
```

But the new repo must generate its own evidence. Historical results are sanity references only.

If NUDT HC-Val cannot be restored, then the current new repo cannot support the planned Full + HC-Val AAAI main table. It should not proceed to seed42 paper-evidence training.

---

## 9. Final recommendation

```text
Do not train yet.
Do not use the uploaded zip as dataset root.
Do not create HC-Val from current results.

Next:
  recover/audit frozen hcval_NUDT-SIRST.txt,
  rerun P1,
  then start seed42 only after P1 passes.
```
