"""Fold-local regression heads, the supplied baseline, and strict submissions."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .data import Dataset, _labels, _read_table


def baseline_features(paths: Iterable[str | Path]) -> np.ndarray:
    """Exactly reproduce the organisers' 233-dimensional NumPy/Pillow features."""
    result = []
    for path in paths:
        with Image.open(path) as source:
            im = ImageOps.exif_transpose(source).convert("RGB")
            aspect = im.width / im.height
            small = np.asarray(im.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float64) / 255
            spatial = small.reshape(8, 4, 8, 4, 3).mean(axis=(1, 3)).ravel()
            gray = small.mean(axis=2)
            texture = gray.reshape(4, 8, 4, 8).std(axis=(1, 3)).ravel()
            hist = np.concatenate([
                np.histogram(small[:, :, c], bins=8, range=(0, 1))[0] / 1024 for c in range(3)
            ])
        result.append(np.concatenate((spatial, texture, hist, [aspect])))
    return np.asarray(result, dtype=np.float64).reshape(-1, 233)


def _xy(x: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Features must be a nonempty finite 2-D matrix")
    if y is not None:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or len(y) != len(x) or not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
            raise ValueError("Labels must match feature rows and be finite within [0, 100]")
    return x, y


@dataclass
class OfficialRidge:
    """The original baseline fit, including its small-scale threshold."""
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x) - self.mean) / self.scale) @ self.weights + self.intercept


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {"name": f"ridge_a{alpha}", "kind": "ridge", "alpha": float(alpha)}
        for alpha in (1, 10, 100)
    ] + [
        {"name": f"svr_c{c}", "kind": "svr", "C": float(c), "epsilon": 2.0, "gamma": "scale"}
        for c in (10, 100)
    ]


def refined_candidate_specs(dimension: int) -> list[dict[str, Any]]:
    """Bounded search motivated by the verified first-run cached experiment."""
    if dimension < 1:
        raise ValueError("Feature dimension must be positive")
    return candidate_specs() + [
        {"name": f"svr_c{c}_g{gamma}_e1", "kind": "svr", "C": float(c),
         "epsilon": 1.0, "gamma": gamma / dimension}
        for c in (100, 300, 1000) for gamma in (0.0625, 0.25, 1.0)
    ]


def fit_candidate(spec: dict[str, Any], x: np.ndarray, y: np.ndarray) -> Any:
    """Fit a serializable predictor. Every transform learns from x only."""
    x, y = _xy(x, y)
    kind = spec["kind"]
    if kind == "official_ridge":
        alpha = float(spec.get("alpha", 100.0))
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be positive and finite")
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-8] = 1
        z = (x - mean) / scale
        intercept = float(y.mean())
        weights = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
        return OfficialRidge(mean, scale, weights, intercept)
    if kind == "median":
        model = DummyRegressor(strategy="median")
    elif kind == "ridge":
        # LSQR avoids allocating a D x D matrix for concatenated image embeddings.
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(spec["alpha"]), solver="lsqr", tol=1e-8))
    elif kind == "svr":
        model = make_pipeline(StandardScaler(), SVR(
            C=float(spec["C"]), epsilon=float(spec.get("epsilon", 2.0)), gamma=spec.get("gamma", "scale")))
    else:
        raise ValueError(f"Unknown candidate kind: {kind}")
    return model.fit(x, y)


def predict_candidate(model: Any, x: np.ndarray) -> np.ndarray:
    x, _ = _xy(x)
    predictions = np.asarray(model.predict(x), dtype=np.float64)
    if predictions.shape != (len(x),) or not np.isfinite(predictions).all():
        raise ValueError("Model returned nonfinite predictions or a wrong shape")
    return np.clip(predictions, 0.0, 100.0)


def metrics(y: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    y, predictions = np.asarray(y, dtype=np.float64), np.asarray(predictions, dtype=np.float64)
    if y.ndim != 1 or predictions.shape != y.shape or not len(y):
        raise ValueError("Metrics require nonempty aligned 1-D arrays")
    if not np.isfinite(y).all() or not np.isfinite(predictions).all():
        raise ValueError("Metrics require finite labels and predictions")
    errors = np.abs(y - predictions)
    return {"mae": float(errors.mean()), "within_10_pct": float(100 * (errors <= 10).mean()), "count": len(y)}


def _ranges(y: np.ndarray, predictions: np.ndarray) -> dict[str, dict | None]:
    result = {}
    for low in range(0, 100, 20):
        high = low + 20
        mask = (y >= low) & ((y <= high) if high == 100 else (y < high))
        key = f"[{low},{high}{']' if high == 100 else ')'}"
        result[key] = metrics(y[mask], predictions[mask]) if mask.any() else None
    return result


def evaluate_candidates(
    x: np.ndarray, y: np.ndarray, fold_ids: np.ndarray,
    candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Compute out-of-fold predictions; selection is based on TRAIN labels only.

    The same folds are shared by all candidates. Comparing many candidates on
    these folds can still overfit validation, so keep the search deliberately small.
    """
    x, y = _xy(x, y)
    folds = np.asarray(fold_ids)
    if folds.shape != y.shape or not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("Fold IDs must be an integer vector matching labels")
    unique = np.unique(folds)
    if len(unique) < 2 or not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("Fold IDs must be contiguous from zero with at least two folds")
    specs = candidate_specs() if candidates is None else candidates
    if not specs or len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("Candidates must be nonempty with unique names")
    results = []
    for spec in specs:
        oof = np.full(len(y), np.nan)
        per_fold = []
        for fold in unique:
            train, valid = folds != fold, folds == fold
            model = fit_candidate(spec, x[train], y[train])
            oof[valid] = predict_candidate(model, x[valid])
            per_fold.append({"fold": int(fold), "train_count": int(train.sum()),
                             **metrics(y[valid], oof[valid])})
        results.append({"name": spec["name"], "spec": dict(spec), "oof": oof,
                        "metrics": metrics(y, oof), "per_fold": per_fold, "ranges": _ranges(y, oof)})
    return results


def evaluate_baselines(x: np.ndarray, y: np.ndarray, fold_ids: np.ndarray) -> list[dict[str, Any]]:
    return evaluate_candidates(x, y, fold_ids, [
        {"name": "median", "kind": "median"},
        {"name": "official_ridge_a100", "kind": "official_ridge", "alpha": 100.0},
    ])


def validate_submission(path: str | Path, expected_ids: Iterable[str]) -> np.ndarray:
    """Validate exact columns, ID coverage, uniqueness and finite [0,100] values."""
    path = Path(path)
    rows = _read_table(path, ["image_id", "load_pct"])
    values = _labels(rows, path)
    expected = tuple(expected_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Expected IDs must be nonempty and unique")
    found = {row["image_id"] for row in rows}
    if found != set(expected):
        raise ValueError(f"Submission ID mismatch: missing={len(set(expected) - found)}, unknown={len(found - set(expected))}")
    mapping = {row["image_id"]: value for row, value in zip(rows, values)}
    return np.asarray([mapping[i] for i in expected], dtype=np.float64)


def write_submission(path: str | Path, dataset: Dataset, predictions: np.ndarray) -> Path:
    """Accept predictions in test order, write rows in sample-submission order."""
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (len(dataset.test_ids),):
        raise ValueError("Expected one prediction for every test ID")
    if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 100)):
        raise ValueError("Submission predictions must be finite within [0, 100]")
    if len(set(dataset.test_ids)) != len(dataset.test_ids) or set(dataset.test_ids) != set(dataset.sample_ids):
        raise ValueError("Test and sample IDs must be unique and match")
    if len(set(dataset.sample_ids)) != len(dataset.sample_ids):
        raise ValueError("Sample IDs must be unique")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping = dict(zip(dataset.test_ids, predictions))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "load_pct"])
        writer.writerows((image_id, f"{float(mapping[image_id]):.6f}") for image_id in dataset.sample_ids)
    validate_submission(path, dataset.test_ids)
    return path
