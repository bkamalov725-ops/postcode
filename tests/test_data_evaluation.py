"""Meaningful safeguards for grouped validation and competition file contracts."""
from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from PIL import Image

from postcode_ml.data import load_dataset, make_folds
from postcode_ml.evaluation import (
    baseline_features, candidate_specs, evaluate_baselines, evaluate_candidates,
    fit_candidate, metrics, predict_candidate, validate_submission, write_submission,
)


def write_table(path: Path, fields: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


class DataContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.train = self.root / "train" / "train.csv"
        self.test = self.root / "nested" / "test" / "test.csv"
        self.train_rows = [(f"train_{i}", float(i * 10)) for i in range(10)]
        self.test_rows = [(f"test_{i}",) for i in range(3)]
        write_table(self.train, ["image_id", "load_pct"], self.train_rows)
        write_table(self.train.with_name("train_groups.csv"), ["image_id", "group_id"],
                    [(image_id, f"g{i // 2}") for i, (image_id, _) in enumerate(self.train_rows)])
        write_table(self.test, ["image_id"], self.test_rows)
        write_table(self.test.with_name("sample_submission.csv"), ["image_id", "load_pct"],
                    [(row[0], 50) for row in reversed(self.test_rows)])
        for csv_path, rows in ((self.train, self.train_rows), (self.test, self.test_rows)):
            image_dir = csv_path.parent / "images"
            image_dir.mkdir()
            for row in rows:
                Image.new("RGB", (20, 30), (128, 128, 128)).save(image_dir / f"{row[0]}.jpg")

    def load(self, **kwargs):
        return load_dataset(self.root, expected_train=10, expected_test=3, **kwargs)

    def test_discovery_uses_complete_test_and_keeps_all_orders(self) -> None:
        write_table(self.root / "baseline" / "test.csv", ["image_id"], [("old_test",)])
        data = self.load(verify_images=True)
        self.assertEqual(data.test_csv, self.test)
        self.assertEqual(data.sample_ids, ("test_2", "test_1", "test_0"))
        self.assertEqual(data.train_rows[2].load_pct, 20)
        self.assertEqual(data.groups.tolist(), [f"g{i // 2}" for i in range(10)])
        with self.assertRaises(FrozenInstanceError):
            data.train_csv = self.test
        with self.assertRaises(ValueError):
            data.y[0] = 10
        with self.assertRaises(ValueError):
            data.y.setflags(write=True)

    def test_rejects_unsafe_or_duplicate_ids_and_nonfinite_labels(self) -> None:
        for bad in ("../escape", "..\\escape", "C:escape", "CON", "", " a", "a."):
            with self.subTest(image_id=bad):
                write_table(self.train, ["image_id", "load_pct"], [(bad, 5), *self.train_rows[1:]])
                with self.assertRaises(ValueError):
                    self.load()
        write_table(self.train, ["image_id", "load_pct"], [self.train_rows[1], *self.train_rows[1:]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load()
        for bad in ("nan", "inf", "-inf", -1, 101, "no"):
            with self.subTest(label=bad):
                write_table(self.train, ["image_id", "load_pct"], [("train_0", bad), *self.train_rows[1:]])
                with self.assertRaises(ValueError):
                    self.load()

    def test_group_and_sample_coverage_are_exact(self) -> None:
        write_table(self.train.with_name("train_groups.csv"), ["image_id", "group_id"],
                    [(row[0], "g") for row in self.train_rows[:-1]])
        with self.assertRaisesRegex(ValueError, "groups must cover exactly"):
            self.load()
        write_table(self.train.with_name("train_groups.csv"), ["image_id", "group_id"],
                    [(row[0], "g") for row in self.train_rows])
        write_table(self.test.with_name("sample_submission.csv"), ["image_id", "load_pct"],
                    [("test_0", 50), ("test_1", 50), ("unknown", 50)])
        with self.assertRaisesRegex(ValueError, "sample submission IDs must match"):
            self.load()

    def test_headers_and_row_width_are_strict(self) -> None:
        write_table(self.train, ["load_pct", "image_id"], [(value, image_id) for image_id, value in self.train_rows])
        with self.assertRaisesRegex(ValueError, "expected exactly columns"):
            self.load()
        write_table(self.train, ["image_id", "load_pct"], [("train_0", 0, "extra"), *self.train_rows[1:]])
        with self.assertRaisesRegex(ValueError, "extra CSV"):
            self.load()

    def test_missing_and_corrupt_images_fail(self) -> None:
        path = self.train.parent / "images" / "train_0.jpg"
        path.write_bytes(b"not an image")
        self.load(verify_images=False)
        with self.assertRaisesRegex(ValueError, "Cannot decode"):
            self.load(verify_images=True)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing image"):
            self.load()

    def test_233_baseline_features_have_expected_values(self) -> None:
        data = self.load()
        features = baseline_features(data.train_paths[:1])
        self.assertEqual(features.shape, (1, 233))
        np.testing.assert_allclose(features[0, :192], 128 / 255)
        np.testing.assert_allclose(features[0, 192:208], 0, atol=1e-15)
        np.testing.assert_array_equal(features[0, 208:232], np.tile([0, 0, 0, 0, 1, 0, 0, 0], 3))
        self.assertAlmostEqual(features[0, -1], 2 / 3)

    def test_submission_is_reordered_and_validated(self) -> None:
        data = self.load()
        path = write_submission(self.root / "output" / "submission.csv", data, np.array([10.5, 50, 100]))
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["image_id"] for row in rows], list(data.sample_ids))
        np.testing.assert_array_equal(validate_submission(path, data.test_ids), [10.5, 50, 100])
        for values in (np.array([0, 1]), np.array([0, 1, 101]), np.array([0, 1, np.nan])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                write_submission(path, data, values)

    def test_submission_rejects_invalid_files(self) -> None:
        path = self.root / "submission.csv"
        cases = [
            (["image_id", "load_pct", "extra"], [("a", 10, 1), ("b", 20, 2)]),
            (["image_id", "load_pct"], [("a", 10), ("a", 20)]),
            (["image_id", "load_pct"], [("a", 10), ("c", 20)]),
            (["image_id", "load_pct"], [("a", 10)]),
        ] + [(["image_id", "load_pct"], [("a", bad), ("b", 20)])
             for bad in ("", "NaN", "Infinity", -0.01, 100.01)]
        for fields, rows in cases:
            with self.subTest(fields=fields, rows=rows):
                write_table(path, fields, rows)
                with self.assertRaises(ValueError):
                    validate_submission(path, ["a", "b"])


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(5)
        self.x = self.rng.normal(size=(40, 12))
        self.y = np.tile([0.0, 20, 40, 60, 100], 8)
        self.groups = np.repeat(np.arange(20).astype(str), 2)
        self.folds = make_folds(self.y, self.groups)

    def test_folds_are_deterministic_and_have_no_group_overlap(self) -> None:
        np.testing.assert_array_equal(self.folds, make_folds(self.y, self.groups))
        self.assertEqual(set(self.folds), set(range(5)))
        for group in np.unique(self.groups):
            self.assertEqual(len(set(self.folds[self.groups == group])), 1)
        with self.assertRaises(ValueError):
            make_folds(self.y, np.full(len(self.y), "same"))

    def test_oof_median_is_fit_without_validation_labels(self) -> None:
        result = evaluate_baselines(self.x, self.y, self.folds)[0]
        for fold in range(5):
            np.testing.assert_array_equal(result["oof"][self.folds == fold], np.median(self.y[self.folds != fold]))
        self.assertEqual(result["metrics"]["count"], len(self.y))
        self.assertEqual(sum(row["count"] for row in result["per_fold"]), len(self.y))
        self.assertEqual(sum(row["count"] for row in result["ranges"].values() if row), len(self.y))

    def test_official_ridge_matches_direct_baseline_calculation(self) -> None:
        self.x[:, 0] = 1.0  # Zero variance must not produce NaN.
        results = evaluate_baselines(self.x, self.y, self.folds)
        expected = np.empty(len(self.y))
        for fold in range(5):
            train, valid = self.folds != fold, self.folds == fold
            x, y = self.x[train], self.y[train]
            mean, scale = x.mean(0), x.std(0)
            scale[scale < 1e-8] = 1
            z = (x - mean) / scale
            weights = np.linalg.solve(z.T @ z + 100 * np.eye(x.shape[1]), z.T @ (y - y.mean()))
            expected[valid] = np.clip(((self.x[valid] - mean) / scale) @ weights + y.mean(), 0, 100)
        np.testing.assert_allclose(results[1]["oof"], expected, rtol=0, atol=1e-10)

    def test_embedding_pipeline_fits_scaler_on_train_and_roundtrips(self) -> None:
        spec = {"name": "ridge_a10", "kind": "ridge", "alpha": 10.0}
        train, valid = self.folds != 0, self.folds == 0
        changed = self.x.copy()
        changed[valid] += 10000
        model = fit_candidate(spec, changed[train], self.y[train])
        np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, self.x[train].mean(0))
        expected = predict_candidate(model, changed[valid])
        result = evaluate_candidates(changed, self.y, self.folds, [spec])[0]
        np.testing.assert_allclose(result["oof"][valid], expected)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(predict_candidate(restored, changed[valid]), expected)

    def test_all_heads_work_with_wide_embeddings(self) -> None:
        wide = self.rng.normal(size=(40, 1200))
        wide[:, 0] = 1
        for spec in candidate_specs():
            with self.subTest(spec=spec):
                model = fit_candidate(spec, wide[:30], self.y[:30])
                predictions = predict_candidate(model, wide[30:])
                self.assertEqual(predictions.shape, (10,))
                self.assertTrue(np.isfinite(predictions).all())
                self.assertTrue(np.all((predictions >= 0) & (predictions <= 100)))

    def test_metrics_include_exactly_ten_and_validate_inputs(self) -> None:
        result = metrics(np.array([0, 50, 100]), np.array([10, 50, 80]))
        self.assertAlmostEqual(result["mae"], 10)
        self.assertAlmostEqual(result["within_10_pct"], 200 / 3)
        for bad in (np.full(40, -1), np.zeros(40), np.zeros(39), self.folds.astype(float)):
            with self.subTest(folds=bad), self.assertRaises(ValueError):
                evaluate_candidates(self.x, self.y, bad)


if __name__ == "__main__":
    unittest.main()
