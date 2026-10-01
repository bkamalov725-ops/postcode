"""Prediction batches cannot damage an existing submission or input data."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from make_submission import generate_submission
from postcode_ml.evaluation import validate_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.test_csv, self.images = self.root / "test.csv", self.root / "images"
        self.images.mkdir()
        self.test_csv.write_text("image_id\na\nb\nc\n", encoding="utf-8")
        for image_id in "abc":
            (self.images / f"{image_id}.jpg").write_bytes(b"predictor test image")
        self.output = self.root / "submission.csv"
        self.predictor = Mock(model_version="test-v1")
        self.predictor.predict.side_effect = lambda paths: [float(ord(path.stem) - ord("a")) for path in paths]
        self.factory = Mock(return_value=self.predictor)

    def generate(self, **kwargs):
        options = dict(bundle=self.root / "model", test_csv=self.test_csv, images_dir=self.images,
                       output=self.output, expected_count=3, batch_size=2, predictor_factory=self.factory)
        options.update(kwargs)
        return generate_submission(**options)

    def test_streams_batches_with_one_cached_predictor(self):
        self.generate()
        self.factory.assert_called_once()
        self.assertEqual(self.predictor.predict.call_count, 2)
        self.assertEqual(validate_submission(self.output, "abc").tolist(), [0, 1, 2])
        self.assertFalse(list(self.root.glob(".submission.csv.*.tmp")))

    def test_default_requires_all_307_images_before_model_is_loaded(self):
        with self.assertRaisesRegex(ValueError, "Expected 307"):
            generate_submission(self.root / "model", self.test_csv, self.images, self.output,
                                predictor_factory=self.factory)
        self.factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_bad_predictions_and_cancellation_preserve_old_csv(self):
        for failure in ([1, 2], [float("nan")], [101], KeyboardInterrupt(), RuntimeError("model failed")):
            with self.subTest(failure=repr(failure)):
                self.output.write_bytes(b"previous submission")
                self.predictor.predict.side_effect = [[10, 20], failure]
                with self.assertRaises((ValueError, RuntimeError, KeyboardInterrupt)):
                    self.generate()
                self.assertEqual(self.output.read_bytes(), b"previous submission")
                self.assertFalse(list(self.root.glob(".submission.csv.*.tmp")))

    def test_refuses_overwriting_csv_photo_or_model_inputs(self):
        for output in (self.test_csv, self.images / "a.jpg", self.root / "model" / "manifest.json"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "overwrite"):
                self.generate(output=output)
        self.factory.assert_not_called()
        self.assertEqual(self.test_csv.read_text(encoding="utf-8"), "image_id\na\nb\nc\n")

    def test_unsafe_duplicate_case_colliding_and_missing_ids_fail_before_model_load(self):
        for content in ("image_id\na\na\nc\n", "image_id\na\nA\nc\n", "image_id\n../a\nb\nc\n",
                        "image_id\nunknown\nb\nc\n"):
            self.test_csv.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.generate()
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
