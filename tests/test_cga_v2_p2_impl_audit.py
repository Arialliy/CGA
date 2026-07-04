from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from tools.official.check_cga_v2_p2_impl_audit import (
    DECISION_MISMATCH,
    DECISION_NO_MISMATCH,
    p3_guard_status,
    run_audit,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_png(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=value).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (16, 16), color=0)
    for y in range(6, 10):
        for x in range(6, 10):
            image.putpixel((x, y), 255)
    image.save(path)


def _metric(
    *,
    model: str,
    split: str,
    dataset: str = "NUDT-SIRST",
    seed: int = 42,
    threshold: float = 0.5,
    epoch: int = 400,
    miou: float = 0.8,
    pred_dir: Path,
    ckpt: Path,
) -> dict:
    pred_dir.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"checkpoint")
    return {
        "F1": 0.8,
        "FA": 0.0,
        "FA_ppm": 0.0,
        "FP_components": 0.0,
        "Pd": 1.0,
        "Precision": 0.8,
        "Recall": 0.8,
        "checkpoint": str(ckpt),
        "checkpoint_epoch": epoch,
        "dataset": dataset,
        "epoch": epoch,
        "mIoU": miou,
        "model": model,
        "nIoU": miou,
        "prediction_dir": str(pred_dir),
        "seed": seed,
        "split": split,
        "threshold": threshold,
        "train_dataset": "NUDT-SIRST",
    }


def _write_predictions(path: Path, count: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (path / f"{i}.png").write_bytes(b"pred")


def _make_case(
    tmp_path: Path,
    *,
    dataset: str = "NUDT-SIRST",
    summary_dataset: str = "NUDT-SIRST",
    seed: int = 42,
    threshold: float = 0.5,
    epoch: int = 400,
    candidate_model: str = "MSHNetCGA",
    list_hash_mismatch: bool = False,
    ohem_full_miou: float = 0.83435,
    ohem_hcval_miou: float = 0.60479,
) -> tuple[Path, Path, Path]:
    root = tmp_path
    dataset_root = root / "datasets" / dataset
    list_dir = dataset_root / "img_idx"
    list_dir.mkdir(parents=True)
    (list_dir / f"train_{dataset}.txt").write_text("a\n", encoding="utf-8")
    (list_dir / f"test_{dataset}.txt").write_text("b\nc\n", encoding="utf-8")
    (list_dir / f"hcval_{dataset}.txt").write_text("c\n", encoding="utf-8")
    for item in ("a", "b", "c"):
        _write_png(dataset_root / "images" / f"{item}.png", value=10)
        _write_mask(dataset_root / "masks" / f"{item}.png")

    p1 = {
        "dataset_name": dataset,
        "dataset_registry_sha256": "registry-sha",
        "dataset_root": str(dataset_root),
        "decision": "DATASET_PREFLIGHT_PASS",
        "duplicate_item_count": 0,
        "gate_pass": True,
        "hcval_count": 1,
        "hcval_list_sha256": _sha(list_dir / f"hcval_{dataset}.txt"),
        "test_count": 2,
        "test_list_sha256": _sha(list_dir / f"test_{dataset}.txt"),
        "train_count": 1,
        "train_list_sha256": _sha(list_dir / f"train_{dataset}.txt"),
        "train_test_overlap_count": 0,
    }
    p2_hash = "wrong" if list_hash_mismatch else p1["train_list_sha256"]
    _write_predictions(root / "results/official/MSHNetOHEM/seed42/NUDT-SIRST/test/predictions", 2)
    _write_predictions(root / "results/official/MSHNetCGA/seed42/NUDT-SIRST/test/predictions", 2)
    _write_predictions(root / "results/official/MSHNetOHEM/seed42/NUDT-SIRST/hcval/predictions", 1)
    _write_predictions(root / "results/official/MSHNetCGA/seed42/NUDT-SIRST/hcval/predictions", 1)
    baseline_full = _metric(
        model="MSHNetOHEM",
        split="test",
        dataset=summary_dataset,
        seed=seed,
        threshold=threshold,
        epoch=epoch,
        miou=ohem_full_miou,
        pred_dir=root / "results/official/MSHNetOHEM/seed42/NUDT-SIRST/test/predictions",
        ckpt=root / "results/official/MSHNetOHEM/seed42/NUDT-SIRST/MSHNetOHEM_400.pth.tar",
    )
    candidate_full = _metric(
        model=candidate_model,
        split="test",
        dataset=summary_dataset,
        seed=seed,
        threshold=threshold,
        epoch=epoch,
        miou=0.7,
        pred_dir=root / "results/official/MSHNetCGA/seed42/NUDT-SIRST/test/predictions",
        ckpt=root / "results/official/MSHNetCGA/seed42/NUDT-SIRST/MSHNetCGA_400.pth.tar",
    )
    baseline_hcval = _metric(
        model="MSHNetOHEM",
        split="hcval",
        dataset=summary_dataset,
        seed=seed,
        threshold=threshold,
        epoch=epoch,
        miou=ohem_hcval_miou,
        pred_dir=root / "results/official/MSHNetOHEM/seed42/NUDT-SIRST/hcval/predictions",
        ckpt=root / "results/official/MSHNetOHEM/seed42/NUDT-SIRST/MSHNetOHEM_400.pth.tar",
    )
    candidate_hcval = _metric(
        model=candidate_model,
        split="hcval",
        dataset=summary_dataset,
        seed=seed,
        threshold=threshold,
        epoch=epoch,
        miou=0.5,
        pred_dir=root / "results/official/MSHNetCGA/seed42/NUDT-SIRST/hcval/predictions",
        ckpt=root / "results/official/MSHNetCGA/seed42/NUDT-SIRST/MSHNetCGA_400.pth.tar",
    )
    p2 = {
        "baseline": "MSHNetOHEM",
        "candidate": "MSHNetCGA",
        "dataset_name": dataset,
        "dataset_registry_sha256": p1["dataset_registry_sha256"],
        "decision": "P2_FAIL_IMPL_AUDIT_ALLOWED",
        "epoch": epoch,
        "full": {"baseline": baseline_full, "candidate": candidate_full, "delta": {}},
        "gate": "Gate-CGA-v2-P2-seed42-reproduction",
        "gate_pass": False,
        "hcval": {"available": True, "baseline": baseline_hcval, "candidate": candidate_hcval, "delta": {}},
        "hcval_list_sha256": p1["hcval_list_sha256"],
        "seed": seed,
        "test_list_sha256": p1["test_list_sha256"],
        "threshold": threshold,
        "train_list_sha256": p2_hash,
    }
    (root / "results/official/MSHNetOHEM/seed42/NUDT-SIRST/train_log.jsonl").write_text(
        json.dumps({"dataset": dataset, "epoch": epoch, "model": "MSHNetOHEM", "seed": 42}) + "\n",
        encoding="utf-8",
    )
    (root / "results/official/MSHNetCGA/seed42/NUDT-SIRST/train_log.jsonl").write_text(
        json.dumps({
            "cga_boundary": 0.1,
            "cga_center": 0.1,
            "cga_peak": 0.1,
            "cga_scale": 0.1,
            "cga_w": 1.0,
            "dataset": dataset,
            "epoch": epoch,
            "model": "MSHNetCGA",
            "seed": 42,
        }) + "\n",
        encoding="utf-8",
    )
    p1_path = root / "p1.json"
    p2_path = root / "p2.json"
    p1_path.write_text(json.dumps(p1), encoding="utf-8")
    p2_path.write_text(json.dumps(p2), encoding="utf-8")
    return root, p1_path, p2_path


def _audit(root: Path, p1: Path, p2: Path) -> dict:
    return run_audit(dataset_name="NUDT-SIRST", seed=42, p1_summary=p1, p2_summary=p2, root=root)


def test_passes_on_well_formed_summaries(tmp_path: Path) -> None:
    root, p1, p2 = _make_case(tmp_path)
    summary = _audit(root, p1, p2)
    assert summary["gate_pass"] is True
    assert summary["decision"] == DECISION_NO_MISMATCH
    assert summary["mismatches"] == []


def test_fails_if_dataset_mismatch(tmp_path: Path) -> None:
    root, p1, p2 = _make_case(tmp_path, summary_dataset="OTHER")
    summary = _audit(root, p1, p2)
    assert summary["decision"] == DECISION_MISMATCH
    assert "full_baseline_dataset_mismatch" in summary["mismatches"]


def test_fails_if_seed_mismatch(tmp_path: Path) -> None:
    root, p1, p2 = _make_case(tmp_path, seed=43)
    summary = _audit(root, p1, p2)
    assert summary["decision"] == DECISION_MISMATCH
    assert "p2_seed_mismatch" in summary["mismatches"]
    assert "full_baseline_seed_mismatch" in summary["mismatches"]


def test_fails_if_threshold_mismatch(tmp_path: Path) -> None:
    root, p1, p2 = _make_case(tmp_path, threshold=0.6)
    summary = _audit(root, p1, p2)
    assert summary["decision"] == DECISION_MISMATCH
    assert "p2_threshold_mismatch" in summary["mismatches"]
    assert "full_candidate_threshold_mismatch" in summary["mismatches"]


def test_fails_if_checkpoint_epoch_mismatch(tmp_path: Path) -> None:
    root, p1, p2 = _make_case(tmp_path, epoch=399)
    summary = _audit(root, p1, p2)
    assert summary["decision"] == DECISION_MISMATCH
    assert "p2_epoch_mismatch" in summary["mismatches"]
    assert "hcval_baseline_checkpoint_epoch_mismatch" in summary["mismatches"]


def test_fails_if_model_identity_mismatch(tmp_path: Path) -> None:
    root, p1, p2 = _make_case(tmp_path, candidate_model="MSHNetOHEM")
    summary = _audit(root, p1, p2)
    assert summary["decision"] == DECISION_MISMATCH
    assert "full_candidate_model_mismatch" in summary["mismatches"]


def test_fails_if_p2_list_hashes_differ_from_p1(tmp_path: Path) -> None:
    root, p1, p2 = _make_case(tmp_path, list_hash_mismatch=True)
    summary = _audit(root, p1, p2)
    assert summary["decision"] == DECISION_MISMATCH
    assert "p2_train_list_sha256_differs_from_p1" in summary["mismatches"]


def test_flags_ohem_anchor_anomaly(tmp_path: Path) -> None:
    root, p1, p2 = _make_case(tmp_path, ohem_full_miou=0.924, ohem_hcval_miou=0.856)
    summary = _audit(root, p1, p2)
    assert summary["decision"] == DECISION_MISMATCH
    assert "ohem_full_miou_outside_historical_sanity" in summary["mismatches"]
    assert "ohem_hcval_miou_outside_historical_sanity" in summary["mismatches"]


def test_blocks_p3_if_p2a_is_not_closed() -> None:
    assert p3_guard_status({"gate_pass": True})["allowed"] is True
    assert p3_guard_status({"gate_pass": False})["allowed"] is False
    assert p3_guard_status({"gate_pass": False}, {"decision": DECISION_NO_MISMATCH})["allowed"] is False
    assert p3_guard_status({"gate_pass": False}, {"decision": DECISION_MISMATCH})["allowed"] is False
