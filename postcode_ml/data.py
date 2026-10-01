"""Strict competition data loading and reproducible, group-safe folds."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class ImageRow:
    image_id: str
    load_pct: float | None = None


@dataclass(frozen=True)
class Dataset:
    train_csv: Path
    groups_csv: Path
    test_csv: Path
    sample_csv: Path
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    train_paths: tuple[Path, ...]
    test_paths: tuple[Path, ...]
    train_rows: tuple[ImageRow, ...]
    test_rows: tuple[ImageRow, ...]
    y: np.ndarray
    groups: np.ndarray


def _valid_id(value: str) -> bool:
    """IDs are filename stems, never paths (including Windows-style paths)."""
    if not value or value != value.strip() or value in {".", ".."}:
        return False
    if value.endswith(".") or re.search(r'[<>:"/\\|?*\x00-\x1f]', value):
        return False
    # Windows reserves these stems even when an extension is appended.
    stem = value.split(".", 1)[0].upper()
    return stem not in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                        *(f"LPT{i}" for i in range(1, 10))}


def _read_table(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"{path}: expected exactly columns {fields}, got {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: table is empty")
    for row in rows:
        if set(row) != set(fields) or any(value is None or value == "" for value in row.values()):
            raise ValueError(f"{path}: missing or extra CSV values")
        if not _valid_id(row["image_id"]):
            raise ValueError(f"{path}: unsafe or empty image_id {row['image_id']!r}")
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate image IDs")
    return rows


def _labels(rows: list[dict[str, str]], path: Path) -> np.ndarray:
    try:
        values = np.asarray([float(row["load_pct"]) for row in rows], dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: load_pct must be numeric") from exc
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise ValueError(f"{path}: load_pct must be finite and within [0, 100]")
    return values


def _image_paths(csv_path: Path, ids: tuple[str, ...], verify_images: bool) -> tuple[Path, ...]:
    image_dir = (csv_path.parent / "images").resolve()
    result = []
    for image_id in ids:
        path = image_dir / f"{image_id}.jpg"
        if not path.is_file() or not path.resolve().is_relative_to(image_dir):
            raise ValueError(f"Missing image or image outside images directory: {path}")
        if verify_images:
            try:
                with Image.open(path) as source:
                    source.verify()
                with Image.open(path) as source:
                    ImageOps.exif_transpose(source).convert("RGB").load()
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                raise ValueError(f"Cannot decode image: {path}: {exc}") from exc
        result.append(path)
    return tuple(result)


def _immutable(values: np.ndarray) -> np.ndarray:
    # A bytes backing store also prevents callers from switching WRITEABLE back on.
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _choose(candidates: list[tuple], errors: list[str], kind: str, root: Path) -> tuple:
    if len(candidates) > 1:
        locations = ", ".join(str(candidate[0]) for candidate in candidates)
        raise ValueError(f"Multiple valid {kind} datasets: {locations}. Supply a more specific data root.")
    if not candidates:
        details = "\n".join(errors[:12]) or "No matching CSV files with an images directory found."
        raise ValueError(f"No valid {kind} dataset under {root}.\n{details}")
    return candidates[0]


def load_dataset(
    root: str | Path = "/kaggle/input", *, expected_train: int | None = 716,
    expected_test: int | None = 307, verify_images: bool = False,
) -> Dataset:
    """Discover complete unpacked datasets recursively; reject ambiguous choices.

    Expected counts deliberately exclude the incomplete 190-image baseline test
    copy. For custom/new labelled datasets, explicitly set expected_train=None.
    Images must live in ``images/`` beside their corresponding CSV file.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Data root does not exist or is not a directory: {root}")
    train_candidates, test_candidates, train_errors, test_errors = [], [], [], []
    for path in sorted(root.rglob("train.csv")):
        groups_path = path.with_name("train_groups.csv")
        try:
            rows = _read_table(path, ["image_id", "load_pct"])
            if expected_train is not None and len(rows) != expected_train:
                raise ValueError(f"{path}: expected {expected_train} train rows, found {len(rows)}")
            y = _labels(rows, path)
            group_rows = _read_table(groups_path, ["image_id", "group_id"])
            ids = tuple(row["image_id"] for row in rows)
            mapping = {row["image_id"]: row["group_id"] for row in group_rows}
            if set(mapping) != set(ids):
                raise ValueError(f"{groups_path}: groups must cover exactly the train IDs")
            if any(not value.strip() or value != value.strip() for value in mapping.values()):
                raise ValueError(f"{groups_path}: group_id must be nonempty, without outer whitespace")
            paths = _image_paths(path, ids, verify_images)
            train_candidates.append((path, groups_path, ids, y, np.asarray([mapping[i] for i in ids]), paths))
        except (OSError, ValueError) as exc:
            train_errors.append(str(exc))
    for path in sorted(root.rglob("test.csv")):
        sample_path = path.with_name("sample_submission.csv")
        try:
            rows = _read_table(path, ["image_id"])
            if expected_test is not None and len(rows) != expected_test:
                raise ValueError(f"{path}: expected {expected_test} test rows, found {len(rows)}")
            sample_rows = _read_table(sample_path, ["image_id", "load_pct"])
            _labels(sample_rows, sample_path)
            ids = tuple(row["image_id"] for row in rows)
            sample_ids = tuple(row["image_id"] for row in sample_rows)
            if set(sample_ids) != set(ids):
                raise ValueError(f"{sample_path}: sample submission IDs must match test.csv exactly")
            paths = _image_paths(path, ids, verify_images)
            test_candidates.append((path, sample_path, ids, sample_ids, paths))
        except (OSError, ValueError) as exc:
            test_errors.append(str(exc))
    train_csv, groups_csv, train_ids, y, groups, train_paths = _choose(
        train_candidates, train_errors, "train", root)
    test_csv, sample_csv, test_ids, sample_ids, test_paths = _choose(
        test_candidates, test_errors, "test", root)
    if set(train_ids) & set(test_ids):
        raise ValueError("Train and test image IDs overlap")
    return Dataset(
        train_csv, groups_csv, test_csv, sample_csv, train_ids, test_ids, sample_ids,
        train_paths, test_paths,
        tuple(ImageRow(i, float(v)) for i, v in zip(train_ids, y)),
        tuple(ImageRow(i) for i in test_ids), _immutable(y), _immutable(groups),
    )


def make_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Return fold IDs aligned to train rows using coarse 20-point label bins.

    The 80--100 bin includes 100. Group constraints take precedence over exact
    class balance. Group IDs are validation constraints, never model features.
    """
    y, groups = np.asarray(y, dtype=np.float64), np.asarray(groups)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups) or not len(y):
        raise ValueError("Labels and groups must be nonempty 1-D arrays of equal length")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
        raise ValueError("Labels must be finite and within [0, 100]")
    if n_splits < 2 or len(np.unique(groups)) < n_splits:
        raise ValueError("Need at least n_splits independent groups, with n_splits >= 2")
    if any(not str(group).strip() for group in groups):
        raise ValueError("Group IDs must be nonempty")
    bins = np.minimum((y // 20).astype(np.int64), 4)
    folds = np.full(len(y), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train, valid) in enumerate(splitter.split(np.zeros((len(y), 1)), bins, groups)):
        if set(groups[train]) & set(groups[valid]):
            raise RuntimeError("Group leakage in fold assignment")
        folds[valid] = fold
    if np.any(folds < 0) or len(np.unique(folds)) != n_splits:
        raise ValueError("Could not build nonempty folds for this dataset")
    return folds


def load_saved_folds(path: str | Path, dataset: Dataset, n_splits: int = 5) -> np.ndarray:
    """Restore a validated assignment in train order, independent of sklearn.

    Saving only the seed is insufficient across scikit-learn versions. This
    loads the exported ``folds.csv`` and checks its IDs, labels, groups and split
    boundaries against the current training data before any model is trained.
    """
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    path = Path(path)
    rows = _read_table(path, ["image_id", "load_pct", "group_id", "fold"])
    mapping = {row["image_id"]: row for row in rows}
    expected = set(dataset.train_ids)
    if len(expected) != len(dataset.train_ids) or set(mapping) != expected:
        raise ValueError(f"{path}: saved folds must cover exactly the train IDs")
    aligned = [mapping[image_id] for image_id in dataset.train_ids]
    labels = _labels(aligned, path)
    if not np.array_equal(labels, dataset.y):
        raise ValueError(f"{path}: saved load_pct values do not match the current training labels")
    groups = np.asarray([row["group_id"] for row in aligned])
    if not np.array_equal(groups, dataset.groups):
        raise ValueError(f"{path}: saved group_id values do not match the current training groups")
    values = [row["fold"] for row in aligned]
    if any(not re.fullmatch(r"[0-9]+", value) for value in values):
        raise ValueError(f"{path}: fold must be an integer in [0, {n_splits - 1}]")
    parsed = [int(value) for value in values]
    if any(value >= n_splits for value in parsed):
        raise ValueError(f"{path}: fold must be an integer in [0, {n_splits - 1}]")
    folds = np.asarray(parsed, dtype=np.int64)
    if not np.array_equal(np.unique(folds), np.arange(n_splits)):
        raise ValueError(f"{path}: expected exactly {n_splits} nonempty folds numbered from zero")
    group_folds = {}
    for group, fold in zip(groups, folds, strict=True):
        previous = group_folds.setdefault(str(group), int(fold))
        if previous != fold:
            raise ValueError(f"{path}: group leakage: group {group!r} appears in multiple folds")
    return _immutable(folds)
