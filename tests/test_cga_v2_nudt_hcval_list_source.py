from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from tools.official.check_cga_v2_nudt_hcval_list_source import run_audit
from tools.official.check_cga_v2_dataset_preflight import run_preflight


def _write_png(path: Path, value: int = 0, size: tuple[int, int] = (16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=value).save(path)


def _write_zero(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _make_registry(path: Path) -> None:
    data = {
        "datasets": {
            "NUDT-SIRST": {
                "root_name": "NUDT-SIRST",
                "image_dir": "images",
                "mask_dir": "masks",
                "list_dir": "img_idx",
                "splits": {
                    "train": {"file": "train_NUDT-SIRST.txt", "required": True},
                    "test": {"file": "test_NUDT-SIRST.txt", "required": True},
                    "hcval": {
                        "file": "hcval_NUDT-SIRST.txt",
                        "required": True,
                        "source_required": True,
                        "claim_role": "hard_clutter_validation",
                    },
                },
                "expected_counts": {"train": None, "test": None, "hcval": None},
                "item_format": {
                    "list_has_extension": False,
                    "image_suffix": ".png",
                    "mask_suffix": ".png",
                    "id_prefix": "",
                    "id_suffix": "",
                    "strip_extension_from_list_item": True,
                },
                "mask_policy": {
                    "type": "binary",
                    "allowed_values": [0, 255],
                    "canonical_required": False,
                },
                "claim_policy": {
                    "role": "main_evidence_after_three_seed_pass",
                    "external_validation_claim_allowed": False,
                    "benchmark_validation_claim_allowed": False,
                },
            }
        }
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _make_dataset(tmp_path: Path, *, hcval_text: str | None = "c\n") -> tuple[Path, Path, Path]:
    dataset_dir = tmp_path / "datasets"
    root = dataset_dir / "NUDT-SIRST"
    registry = tmp_path / "datasets.yaml"
    _make_registry(registry)
    (root / "img_idx").mkdir(parents=True)
    (root / "img_idx" / "train_NUDT-SIRST.txt").write_text("a\n", encoding="utf-8")
    (root / "img_idx" / "test_NUDT-SIRST.txt").write_text("b\n", encoding="utf-8")
    if hcval_text is not None:
        (root / "img_idx" / "hcval_NUDT-SIRST.txt").write_text(hcval_text, encoding="utf-8")
    for item in ("a", "b", "c", "d"):
        _write_png(root / "images" / f"{item}.png", value=10)
        _write_png(root / "masks" / f"{item}.png", value=255)
    return dataset_dir, root, registry


def _audit(dataset_dir: Path, registry: Path, candidate: Path, source_note: str = "Frozen pre-existing HC-Val split.") -> dict:
    return run_audit(
        dataset_dir=dataset_dir,
        dataset_name="NUDT-SIRST",
        candidate_hcval_list=candidate,
        source_note=source_note,
        registry_path=registry,
    )


def test_missing_hcval_list_fails(tmp_path: Path) -> None:
    dataset_dir, root, registry = _make_dataset(tmp_path, hcval_text=None)

    summary = _audit(dataset_dir, registry, root / "img_idx" / "hcval_NUDT-SIRST.txt")

    assert summary["gate_pass"] is False
    assert summary["decision"] == "STOP_NEW_REPO_CGA_V2_AT_NUDT_HCVAL_SPLIT_MISSING"
    assert "missing_hcval_list" in summary["failures"]


def test_empty_hcval_list_fails(tmp_path: Path) -> None:
    dataset_dir, root, registry = _make_dataset(tmp_path, hcval_text="")

    summary = _audit(dataset_dir, registry, root / "img_idx" / "hcval_NUDT-SIRST.txt")

    assert summary["gate_pass"] is False
    assert "empty_hcval_list" in summary["failures"]


def test_duplicate_ids_fail(tmp_path: Path) -> None:
    dataset_dir, root, registry = _make_dataset(tmp_path, hcval_text="c\nc\n")

    summary = _audit(dataset_dir, registry, root / "img_idx" / "hcval_NUDT-SIRST.txt")

    assert summary["gate_pass"] is False
    assert "duplicate_hcval_items" in summary["failures"]
    assert summary["duplicate_count"] == 1


def test_path_traversal_fails(tmp_path: Path) -> None:
    dataset_dir, root, registry = _make_dataset(tmp_path, hcval_text="../secret\n")

    summary = _audit(dataset_dir, registry, root / "img_idx" / "hcval_NUDT-SIRST.txt")

    assert summary["gate_pass"] is False
    assert "path_traversal_items" in summary["failures"]


def test_valid_hcval_list_passes_and_writes_sha256(tmp_path: Path) -> None:
    dataset_dir, root, registry = _make_dataset(tmp_path, hcval_text="c\n")

    summary = _audit(dataset_dir, registry, root / "img_idx" / "hcval_NUDT-SIRST.txt")

    assert summary["gate_pass"] is True
    assert summary["decision"] == "NUDT_HCVAL_LIST_SOURCE_ACCEPTED"
    assert summary["hcval_count"] == 1
    assert summary["hcval_list_sha256"]
    assert summary["source_accepted_before_new_repo_seed42"] is True
    assert summary["next_allowed_gate"] == "Gate-CGA-v2-P1-dataset-preflight"


def test_zero_byte_image_or_mask_in_hcval_fails(tmp_path: Path) -> None:
    dataset_dir, root, registry = _make_dataset(tmp_path, hcval_text="c\nd\n")
    _write_zero(root / "images" / "c.png")
    _write_zero(root / "masks" / "d.png")

    summary = _audit(dataset_dir, registry, root / "img_idx" / "hcval_NUDT-SIRST.txt")

    assert summary["gate_pass"] is False
    assert "zero_byte_images" in summary["failures"]
    assert "zero_byte_masks" in summary["failures"]


def test_copying_test_list_without_source_note_fails(tmp_path: Path) -> None:
    dataset_dir, root, registry = _make_dataset(tmp_path, hcval_text="b\n")

    summary = _audit(dataset_dir, registry, root / "img_idx" / "hcval_NUDT-SIRST.txt", source_note="")

    assert summary["gate_pass"] is False
    assert "source_note_required" in summary["failures"]


def test_preflight_reports_missing_required_hcval_and_zero_byte_files(tmp_path: Path) -> None:
    dataset_dir, root, registry = _make_dataset(tmp_path, hcval_text=None)

    missing = run_preflight(dataset_dir, "NUDT-SIRST", registry)
    assert missing["gate_pass"] is False
    assert missing["missing_required_splits"] == ["hcval"]
    assert missing["missing_hcval_list"] is True
    assert missing["hcval_list_sha256"] is None
    assert missing["next_allowed_gate"] == "Gate-CGA-v2-P1A-NUDT-HCVal-List-Source-Audit"

    (root / "img_idx" / "hcval_NUDT-SIRST.txt").write_text("c\n", encoding="utf-8")
    _write_zero(root / "images" / "c.png")
    zero = run_preflight(dataset_dir, "NUDT-SIRST", registry)
    assert zero["gate_pass"] is False
    assert zero["decision"] == "STOP_DATASET_DUE_TO_ZERO_BYTE_FILES"
    assert zero["zero_byte_images"] == 1
    assert zero["zero_byte_examples"] == ["c"]
