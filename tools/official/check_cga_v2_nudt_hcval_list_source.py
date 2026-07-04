"""Audit a frozen NUDT-SIRST HC-Val list before paper-evidence training."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from dataset import (
    dataset_root,
    get_dataset_entry,
    normalize_item_id,
    read_split_items,
    resolve_item_paths,
    sha256_file,
    split_spec,
)


GATE = "Gate-CGA-v2-P1A-NUDT-HCVal-List-Source-Audit"
PASS_DECISION = "NUDT_HCVAL_LIST_SOURCE_ACCEPTED"
FAIL_DECISION = "STOP_NEW_REPO_CGA_V2_AT_NUDT_HCVAL_SPLIT_MISSING"


def _has_path_traversal(raw: str) -> bool:
    value = raw.strip().replace("\\", "/")
    if not value:
        return False
    path = PurePosixPath(value)
    return path.is_absolute() or ".." in path.parts


def _read_candidate(path: Path, entry: dict[str, Any]) -> tuple[list[str], list[str]]:
    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    traversal = [line for line in raw_lines if _has_path_traversal(line)]
    items = [normalize_item_id(line, entry) for line in raw_lines if not _has_path_traversal(line)]
    items = [item for item in items if item]
    return items, traversal


def _open_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def _inspect_item(
    dataset_dir: Path,
    dataset_name: str,
    registry_path: Path,
    item_id: str,
    allowed_values: set[int],
) -> dict[str, int]:
    image_path, mask_path = resolve_item_paths(dataset_dir, dataset_name, item_id, registry_path)
    row = {
        "missing_images": 0,
        "missing_masks": 0,
        "zero_byte_images": 0,
        "zero_byte_masks": 0,
        "unreadable_images": 0,
        "unreadable_masks": 0,
        "size_mismatch": 0,
        "illegal_mask_value_files": 0,
    }
    if not image_path.exists():
        row["missing_images"] = 1
        return row
    if not mask_path.exists():
        row["missing_masks"] = 1
        return row
    if image_path.stat().st_size == 0:
        row["zero_byte_images"] = 1
        return row
    if mask_path.stat().st_size == 0:
        row["zero_byte_masks"] = 1
        return row
    try:
        image = _open_gray(image_path)
    except Exception:
        row["unreadable_images"] = 1
        return row
    try:
        mask = _open_gray(mask_path)
    except Exception:
        row["unreadable_masks"] = 1
        return row
    if image.shape != mask.shape:
        row["size_mismatch"] = 1
    if not set(int(v) for v in np.unique(mask)).issubset(allowed_values):
        row["illegal_mask_value_files"] = 1
    return row


def run_audit(
    *,
    dataset_dir: Path,
    dataset_name: str,
    candidate_hcval_list: Path,
    source_note: str,
    registry_path: Path,
) -> dict[str, Any]:
    entry = get_dataset_entry(dataset_name, registry_path)
    root = dataset_root(dataset_dir, dataset_name, registry_path).resolve()
    mask_policy = entry.get("mask_policy", {})
    allowed_values = set(int(v) for v in mask_policy.get("allowed_values", [0, 255]))
    failures: list[str] = []
    examples: dict[str, list[str]] = {}

    if dataset_name != "NUDT-SIRST":
        failures.append("dataset_name_must_be_NUDT-SIRST")
    if not str(source_note).strip():
        failures.append("source_note_required")
    if not candidate_hcval_list.exists():
        failures.append("missing_hcval_list")
        items: list[str] = []
        traversal: list[str] = []
    else:
        items, traversal = _read_candidate(candidate_hcval_list, entry)

    if candidate_hcval_list.exists() and not items:
        failures.append("empty_hcval_list")
    if traversal:
        failures.append("path_traversal_items")
        examples["path_traversal_items"] = traversal[:20]

    counts = Counter(items)
    duplicates = sorted(item for item, count in counts.items() if count > 1)
    if duplicates:
        failures.append("duplicate_hcval_items")
        examples["duplicate_hcval_items"] = duplicates[:20]

    try:
        train_items = set(read_split_items(dataset_dir, dataset_name, "train", registry_path, required=True))
    except Exception:
        train_items = set()
        failures.append("cannot_read_train_split")
    hcval_train_overlap = sorted(set(items) & train_items)
    allow_train_overlap = bool(split_spec(entry, "hcval").get("allow_train_overlap", False))
    if hcval_train_overlap and not allow_train_overlap:
        failures.append("hcval_train_overlap")
        examples["hcval_train_overlap"] = hcval_train_overlap[:20]

    aggregate = {
        "missing_images": 0,
        "missing_masks": 0,
        "zero_byte_images": 0,
        "zero_byte_masks": 0,
        "unreadable_images": 0,
        "unreadable_masks": 0,
        "size_mismatch": 0,
        "illegal_mask_value_files": 0,
    }
    for item_id in sorted(counts):
        row = _inspect_item(dataset_dir, dataset_name, registry_path, item_id, allowed_values)
        for key, value in row.items():
            aggregate[key] += value
            if value:
                examples.setdefault(key, [])
                if len(examples[key]) < 20:
                    examples[key].append(item_id)
    for key, value in aggregate.items():
        if value:
            failures.append(key)

    gate_pass = not failures
    return {
        "gate": GATE,
        "gate_pass": gate_pass,
        "decision": PASS_DECISION if gate_pass else FAIL_DECISION,
        "dataset": dataset_name,
        "dataset_root": str(root),
        "hcval_list_path": str(candidate_hcval_list.resolve()) if candidate_hcval_list.exists() else str(candidate_hcval_list),
        "hcval_list_sha256": sha256_file(candidate_hcval_list) if candidate_hcval_list.exists() else None,
        "hcval_count": len(items),
        "duplicate_count": len(duplicates),
        "path_traversal_count": len(traversal),
        "hcval_train_overlap_count": len(hcval_train_overlap),
        "source_note": source_note,
        "source_accepted_before_new_repo_seed42": gate_pass,
        "next_allowed_gate": "Gate-CGA-v2-P1-dataset-preflight" if gate_pass else None,
        "failures": failures,
        "examples": examples,
        **aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit NUDT-SIRST hcval_NUDT-SIRST.txt source.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--candidate_hcval_list", required=True)
    parser.add_argument("--source_note", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry", default="configs/datasets.yaml")
    args = parser.parse_args()

    summary = run_audit(
        dataset_dir=Path(args.dataset_dir),
        dataset_name=args.dataset_name,
        candidate_hcval_list=Path(args.candidate_hcval_list),
        source_note=args.source_note,
        registry_path=Path(args.registry),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
