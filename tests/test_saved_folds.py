"""Saved CV assignments must survive row reordering without leaking groups."""
import csv
import hashlib
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from postcode_ml.data import load_saved_folds
from postcode_ml.experiment import _prepare_folds, main, run


class SavedFoldsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "original_folds.csv"
        self.dataset = SimpleNamespace(
            train_ids=tuple(f"image_{i}" for i in range(10)),
            y=np.arange(10, dtype=np.float64) * 10,
            groups=np.asarray([f"group_{i // 2}" for i in range(10)]),
        )
        self.folds = np.repeat(np.arange(5), 2)
        self.rows = [dict(image_id=image_id, load_pct=float(y), group_id=group, fold=int(fold))
                     for image_id, y, group, fold in zip(
                         self.dataset.train_ids, self.dataset.y, self.dataset.groups, self.folds, strict=True)]

    def write(self, rows=None, fields=None):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or ["image_id", "load_pct", "group_id", "fold"])
            writer.writeheader()
            writer.writerows(self.rows if rows is None else rows)

    def test_reordered_rows_are_aligned_to_dataset_and_immutable(self):
        self.write(list(reversed(self.rows)))
        loaded = load_saved_folds(self.path, self.dataset)
        np.testing.assert_array_equal(loaded, self.folds)
        self.assertEqual(loaded.dtype, np.int64)
        with self.assertRaises(ValueError):
            loaded[0] = 4

    def test_missing_duplicate_unknown_ids_rejected(self):
        cases = [self.rows[:-1], self.rows + [dict(self.rows[0])],
                 [dict(self.rows[0], image_id="unknown"), *self.rows[1:]]]
        for rows in cases:
            with self.subTest(rows=rows):
                self.write(rows)
                with self.assertRaises(ValueError):
                    load_saved_folds(self.path, self.dataset)

    def test_mismatched_labels_and_groups_rejected(self):
        for change in ({"load_pct": 0.0001}, {"load_pct": "NaN"}, {"load_pct": 101},
                       {"group_id": "group_changed"}, {"group_id": " group_0"}):
            with self.subTest(change=change):
                self.write([dict(self.rows[0], **change), *self.rows[1:]])
                with self.assertRaises(ValueError):
                    load_saved_folds(self.path, self.dataset)

    def test_invalid_fold_numbers_and_empty_fold_rejected(self):
        for fold in (-1, 5, "0.0", "1.5", "", "NaN", "Infinity", " 0", "1e0"):
            with self.subTest(fold=fold):
                self.write([dict(self.rows[0], fold=fold), *self.rows[1:]])
                with self.assertRaises(ValueError):
                    load_saved_folds(self.path, self.dataset)
        self.write([dict(row, fold=min(row["fold"], 3)) for row in self.rows])
        with self.assertRaisesRegex(ValueError, "nonempty folds"):
            load_saved_folds(self.path, self.dataset)

    def test_group_leakage_rejected_even_when_all_fold_numbers_exist(self):
        self.write([dict(self.rows[0], fold=1), *self.rows[1:]])
        with self.assertRaisesRegex(ValueError, "group leakage"):
            load_saved_folds(self.path, self.dataset)

    def test_csv_contract_rejects_wrong_headers_and_extra_columns(self):
        self.write(fields=["image_id", "group_id", "load_pct", "fold"])
        with self.assertRaisesRegex(ValueError, "expected exactly columns"):
            load_saved_folds(self.path, self.dataset)
        self.write([dict(row, extra="bad") for row in self.rows],
                   fields=["image_id", "load_pct", "group_id", "fold", "extra"])
        with self.assertRaisesRegex(ValueError, "expected exactly columns"):
            load_saved_folds(self.path, self.dataset)

    def test_reuse_does_not_call_splitter_and_records_both_hashes(self):
        self.write(list(reversed(self.rows)))
        source_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        output = self.root / "output"
        output.mkdir()
        with patch("postcode_ml.experiment.make_folds") as splitter:
            folds, metadata = _prepare_folds(self.dataset, output, seed=999, folds_csv=self.path)
        splitter.assert_not_called()
        np.testing.assert_array_equal(folds, self.folds)
        self.assertEqual(metadata["source"], "saved_csv")
        self.assertEqual(metadata["source_sha256"], source_hash)
        self.assertEqual(metadata["exported_sha256"], hashlib.sha256((output / "folds.csv").read_bytes()).hexdigest())
        np.testing.assert_array_equal(load_saved_folds(output / "folds.csv", self.dataset), self.folds)

    def test_generate_preserves_seed_and_records_exported_hash(self):
        output = self.root / "output"
        output.mkdir()
        with patch("postcode_ml.experiment.make_folds", return_value=self.folds) as splitter:
            folds, metadata = _prepare_folds(self.dataset, output, seed=13)
        splitter.assert_called_once_with(self.dataset.y, self.dataset.groups, n_splits=5, seed=13)
        np.testing.assert_array_equal(folds, self.folds)
        self.assertEqual(metadata["source"], "StratifiedGroupKFold")
        self.assertEqual(metadata["seed"], 13)
        self.assertTrue(metadata["scikit_learn_version"])
        self.assertEqual(metadata["exported_sha256"], hashlib.sha256((output / "folds.csv").read_bytes()).hexdigest())


class RetrainingCliTests(unittest.TestCase):
    def test_custom_expected_train_reaches_loader_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments = ["run_experiment.py", "--output", temporary, "--expected-train", "720"]
            with patch("sys.argv", arguments), patch("postcode_ml.experiment.run") as runner:
                main()
            args = runner.call_args.args[0]
            self.assertEqual(args.expected_train, 720)
            # Stop at the data boundary: this test never trains or extracts features.
            with patch("postcode_ml.experiment.copy_source"), \
                 patch("postcode_ml.experiment.load_dataset", side_effect=RuntimeError("stop-before-training")) as loader, \
                 self.assertRaisesRegex(RuntimeError, "stop-before-training"):
                run(args)
            loader.assert_called_once_with(args.data_root, expected_train=720, verify_images=True)

    def test_default_expected_train_is_716(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("sys.argv", ["run_experiment.py", "--output", temporary]), \
                 patch("postcode_ml.experiment.run") as runner:
                main()
            self.assertEqual(runner.call_args.args[0].expected_train, 716)

    def test_nonpositive_or_fractional_expected_train_rejected(self):
        for value in ("0", "-1", "1.5"):
            with self.subTest(value=value), \
                 patch("sys.argv", ["run_experiment.py", "--expected-train", value]), \
                 patch("sys.stderr", new_callable=StringIO), \
                 patch("postcode_ml.experiment.run") as runner, \
                 self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
