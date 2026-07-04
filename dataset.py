"""Dataset registry and loaders for dataset-name-selectable CGA-v2 runs."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset


REGISTRY_PATH = Path(__file__).resolve().parent / "configs" / "datasets.yaml"
REQUIRED_SPLITS = {"train", "test"}


def split_spec(entry: dict[str, Any], split: str) -> dict[str, Any]:
    raw = entry.get("splits", {}).get(split)
    if not raw:
        raise KeyError(f"Dataset {entry.get('name', '<unknown>')!r} has no split {split!r} in registry")
    if isinstance(raw, dict):
        spec = dict(raw)
    else:
        spec = {"file": raw}
    spec.setdefault("required", split in REQUIRED_SPLITS)
    return spec


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset_registry(registry_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(registry_path) if registry_path is not None else REGISTRY_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("datasets"), dict):
        raise ValueError(f"Invalid dataset registry schema: {path}")
    return data


def dataset_registry_sha256(registry_path: str | Path | None = None) -> str:
    path = Path(registry_path) if registry_path is not None else REGISTRY_PATH
    return sha256_file(path)


def get_dataset_entry(dataset_name: str, registry_path: str | Path | None = None) -> dict[str, Any]:
    registry = load_dataset_registry(registry_path)
    try:
        entry = dict(registry["datasets"][dataset_name])
    except KeyError as exc:
        names = ", ".join(sorted(registry["datasets"]))
        raise KeyError(f"Unknown DATASET_NAME={dataset_name!r}; registered datasets: {names}") from exc
    entry["name"] = dataset_name
    return entry


def dataset_root(dataset_dir: str | Path, dataset_name: str, registry_path: str | Path | None = None) -> Path:
    entry = get_dataset_entry(dataset_name, registry_path)
    return Path(dataset_dir) / str(entry["root_name"])


def split_path(
    dataset_dir: str | Path,
    dataset_name: str,
    split: str,
    registry_path: str | Path | None = None,
) -> Path:
    entry = get_dataset_entry(dataset_name, registry_path)
    split_file = split_spec(entry, split)["file"]
    return dataset_root(dataset_dir, dataset_name, registry_path) / str(entry["list_dir"]) / str(split_file)


def split_exists(
    dataset_dir: str | Path,
    dataset_name: str,
    split: str,
    registry_path: str | Path | None = None,
) -> bool:
    try:
        return split_path(dataset_dir, dataset_name, split, registry_path).is_file()
    except KeyError:
        return False


def normalize_item_id(item: str, entry: dict[str, Any]) -> str:
    fmt = entry.get("item_format", {})
    value = item.strip().replace("\\", "/")
    if not value or value.startswith("#"):
        return ""
    if fmt.get("strip_extension_from_list_item", False):
        path = PurePosixPath(value)
        value = str(path.with_suffix("")) if path.suffix else str(path)
    prefix = str(fmt.get("id_prefix", ""))
    suffix = str(fmt.get("id_suffix", ""))
    return f"{prefix}{value}{suffix}"


def read_split_items(
    dataset_dir: str | Path,
    dataset_name: str,
    split: str = "train",
    registry_path: str | Path | None = None,
    *,
    required: bool | None = None,
) -> list[str]:
    entry = get_dataset_entry(dataset_name, registry_path)
    path = split_path(dataset_dir, dataset_name, split, registry_path)
    if required is None:
        required = bool(split_spec(entry, split).get("required", split in REQUIRED_SPLITS))
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {dataset_name} {split} list: {path}")
        return []
    items = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        item_id = normalize_item_id(raw, entry)
        if item_id:
            items.append(item_id)
    return items


def resolve_item_paths(
    dataset_dir: str | Path,
    dataset_name: str,
    item_id: str,
    registry_path: str | Path | None = None,
) -> tuple[Path, Path]:
    entry = get_dataset_entry(dataset_name, registry_path)
    root = dataset_root(dataset_dir, dataset_name, registry_path)
    fmt = entry.get("item_format", {})
    image_path = root / str(entry["image_dir"]) / f"{item_id}{fmt.get('image_suffix', '.png')}"
    mask_path = root / str(entry["mask_dir"]) / f"{item_id}{fmt.get('mask_suffix', '.png')}"
    return image_path, mask_path


def read_gray_image(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def _to_float_image(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 255.0


def _to_binary_mask(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.float32) > 127.0).astype(np.float32)


def _pad_to_at_least(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant")
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
    return image, mask


def _random_crop(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _pad_to_at_least(image, mask, size)
    h, w = image.shape[:2]
    top = 0 if h == size else random.randint(0, h - size)
    left = 0 if w == size else random.randint(0, w - size)
    return image[top : top + size, left : left + size], mask[top : top + size, left : left + size]


def _pad_to_stride(image: np.ndarray, mask: np.ndarray, stride: int = 16) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    if pad_h == 0 and pad_w == 0:
        return image, mask
    image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant")
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
    return image, mask


class TrainSetLoader(Dataset):
    """Generic training loader selected only by DATASET_DIR and DATASET_NAME."""

    __test__ = False

    def __init__(
        self,
        dataset_dir: str | Path,
        dataset_name: str,
        patch_size: int = 256,
        split: str = "train",
        registry_path: str | Path | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.dataset_name = dataset_name
        self.patch_size = int(patch_size)
        self.split = split
        self.registry_path = registry_path
        self.items = read_split_items(dataset_dir, dataset_name, split, registry_path, required=True)
        if not self.items:
            raise ValueError(f"Empty {dataset_name} {split} split")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item_id = self.items[index]
        image_path, mask_path = resolve_item_paths(self.dataset_dir, self.dataset_name, item_id, self.registry_path)
        image = _to_float_image(read_gray_image(image_path))
        mask = _to_binary_mask(read_gray_image(mask_path))
        image, mask = _random_crop(image, mask, self.patch_size)
        return torch.from_numpy(image[None]).float(), torch.from_numpy(mask[None]).float()


class TestSetLoader(Dataset):
    """Generic test/hcval loader; pads to the model stride and returns original size."""

    __test__ = False

    def __init__(
        self,
        dataset_dir: str | Path,
        train_dataset_name: str,
        dataset_name: str | None = None,
        split: str = "test",
        registry_path: str | Path | None = None,
        stride: int = 16,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.train_dataset_name = train_dataset_name
        self.dataset_name = dataset_name or train_dataset_name
        self.split = split
        self.registry_path = registry_path
        self.stride = int(stride)
        self.items = read_split_items(dataset_dir, self.dataset_name, split, registry_path, required=True)
        if not self.items:
            raise ValueError(f"Empty {self.dataset_name} {split} split")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        item_id = self.items[index]
        image_path, mask_path = resolve_item_paths(self.dataset_dir, self.dataset_name, item_id, self.registry_path)
        image = _to_float_image(read_gray_image(image_path))
        mask = _to_binary_mask(read_gray_image(mask_path))
        original_size = torch.tensor([image.shape[0], image.shape[1]], dtype=torch.long)
        image, mask = _pad_to_stride(image, mask, self.stride)
        return torch.from_numpy(image[None]).float(), torch.from_numpy(mask[None]).float(), original_size, item_id
