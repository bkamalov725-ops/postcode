"""Validate data/model import before any existing output can be changed."""
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile

from PIL import Image

from scripts.prepare_data import prepare_data
from scripts.prepare_model import prepare_model


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive, self.output = self.root / "data.zip", self.root / "data"
        image = io.BytesIO()
        Image.new("RGB", (11, 17), (90, 110, 120)).save(image, format="JPEG")
        self.files = {
            "train/train.csv": b"image_id,load_pct\na,25\nb,100\n",
            "train/train_groups.csv": b"image_id,group_id\na,g1\nb,g2\n",
            "test/test.csv": b"image_id\nc\n",
            "test/sample_submission.csv": b"image_id,load_pct\nc,50\n",
            "train/images/a.jpg": image.getvalue(),
            "train/images/b.jpg": image.getvalue(),
            "test/images/c.jpg": image.getvalue(),
        }

    def write_zip(self, files=None):
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, content in (self.files if files is None else files).items():
                archive.writestr(name, content)

    def prepare(self):
        return prepare_data(self.archive, self.output, expected_train=2, expected_test=1)

    def test_canonical_import_and_identical_reuse(self):
        self.write_zip()
        self.assertEqual(self.prepare(), 7)
        self.assertEqual(self.prepare(), 0)
        self.assertEqual((self.output / "train/train.csv").read_bytes(), self.files["train/train.csv"])
        self.assertFalse(list(self.root.glob(".prepare-data-*")))

    def test_legacy_layout_maps_to_canonical(self):
        files = {("postcode/" + name if name.startswith("train/") else "postcode/test/" + name): data
                 for name, data in self.files.items()}
        files["postcode/baseline/test/test.csv"] = b"ignored incomplete dataset"
        self.write_zip(files)
        self.assertEqual(self.prepare(), 7)
        self.assertTrue((self.output / "test/images/c.jpg").is_file())

    def test_mixed_layout_or_duplicate_members_are_rejected(self):
        self.write_zip({**self.files, "postcode/train/train.csv": self.files["train/train.csv"]})
        with self.assertRaisesRegex(ValueError, "Duplicate|layout"):
            self.prepare()
        self.write_zip()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.archive, "a") as archive:
                archive.writestr("train/train.csv", self.files["train/train.csv"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_existing_conflict_prevents_all_new_files(self):
        destination = self.output / "test/test.csv"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"keep me")
        self.write_zip()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(destination.read_bytes(), b"keep me")
        self.assertFalse((self.output / "train").exists())
        self.assertFalse(list(self.root.glob(".prepare-data-*")))

    def test_corrupt_archive_preserves_existing_output_and_leaves_no_partial_files(self):
        self.write_zip()
        content = bytearray(self.archive.read_bytes())
        offset = content.find(self.files["test/images/c.jpg"])
        content[offset + 100] ^= 1
        self.archive.write_bytes(content)
        self.output.mkdir()
        marker = self.output / "user-notes.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(zipfile.BadZipFile):
            self.prepare()
        self.assertEqual(list(self.output.iterdir()), [marker])
        self.assertFalse(list(self.root.glob(".prepare-data-*")))

    def test_unsafe_paths_are_rejected_even_outside_selected_dataset(self):
        for path in ("../outside.txt", "/outside.txt", "C:/outside.txt", "other/..\\outside.txt", "other/NUL", "other/file. "):
            with self.subTest(path=path):
                self.write_zip({**self.files, path: b"bad"})
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    self.prepare()
        self.assertFalse(self.output.exists())

    def test_archive_symlink_is_rejected(self):
        self.write_zip()
        entry = zipfile.ZipInfo("other/link")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr(entry, "../../outside")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.prepare()

    def test_csv_counts_id_coverage_duplicates_and_finite_labels_are_checked(self):
        cases = {
            "train/train.csv": [b"image_id,load_pct\na,nan\nb,100\n", b"image_id,load_pct\na,20\na,30\n"],
            "train/train_groups.csv": [b"image_id,group_id\na,g1\n"],
            "test/test.csv": [b"image_id\nc\nd\n"],
            "test/sample_submission.csv": [b"image_id,load_pct\nunknown,50\n", b"image_id,load_pct\nc,inf\n"],
        }
        for name, contents in cases.items():
            for content in contents:
                with self.subTest(name=name, content=content):
                    self.write_zip({**self.files, name: content})
                    with self.assertRaises(ValueError):
                        self.prepare()
                    self.assertFalse(self.output.exists())

    def test_missing_extra_and_corrupt_images_are_rejected(self):
        for files in ({k: v for k, v in self.files.items() if k != "train/images/a.jpg"},
                      {**self.files, "train/images/unknown.jpg": self.files["train/images/a.jpg"]},
                      {**self.files, "train/images/a.jpg": b"not JPEG"}):
            self.write_zip(files)
            with self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output.exists())


class PrepareModelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive, self.output = self.root / "model.zip", self.root / "model"
        self.manifest = {"format_version": 1, "components": [{"encoder": "convnext_tiny"}], "weights": [1]}
        self.files = {
            "manifest.json": json.dumps(self.manifest).encode(),
            # This intentionally is not valid pickle; extraction must never load it.
            "heads.pkl": b"not pickle",
            "requirements-lock.txt": b"torch==2.10.0\n",
            "encoders/convnext_tiny/model.pt": b"trusted-weights",
            "encoders/convnext_tiny/encoder.json": json.dumps({"name": "convnext_tiny",
                "weights_sha256": hashlib.sha256(b"trusted-weights").hexdigest()}).encode(),
            "encoders/unused/model.pt": b"omit unused encoder",
            "postcode_ml/inference.py": b"omit old application code",
        }

    def write_zip(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, content in self.files.items():
                archive.writestr("model_bundle/" + name, content)

    def test_extracts_selected_model_without_unpickling_and_reuses_identical_output(self):
        self.write_zip()
        self.assertEqual(prepare_model(self.archive, self.output, trust_model_source=True), 5)
        self.assertEqual(prepare_model(self.archive, self.output, trust_model_source=True), 0)
        self.assertEqual((self.output / "heads.pkl").read_bytes(), b"not pickle")
        self.assertFalse((self.output / "encoders/unused").exists())
        self.assertFalse((self.output / "postcode_ml").exists())

    def test_requires_explicit_trust_and_rejects_missing_weights(self):
        self.write_zip()
        with self.assertRaisesRegex(ValueError, "trust-model-source"):
            prepare_model(self.archive, self.output)
        del self.files["encoders/convnext_tiny/model.pt"]
        self.write_zip()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            prepare_model(self.archive, self.output, trust_model_source=True)
        self.assertFalse(self.output.exists())

    def test_checksum_failure_is_detected_before_publish(self):
        self.files["encoders/convnext_tiny/model.pt"] = b"changed-weights"
        self.write_zip()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            prepare_model(self.archive, self.output, trust_model_source=True)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".prepare-model-*")))

    def test_convnext_allows_null_source_files_but_dino_requires_offline_source(self):
        metadata = json.loads(self.files["encoders/convnext_tiny/encoder.json"])
        metadata["source_files"] = None
        self.files["encoders/convnext_tiny/encoder.json"] = json.dumps(metadata).encode()
        self.write_zip()
        self.assertEqual(prepare_model(self.archive, self.output, trust_model_source=True), 5)
        self.manifest["components"] = [{"encoder": "dinov2_vitb14"}]
        self.files["manifest.json"] = json.dumps(self.manifest).encode()
        metadata["name"] = "dinov2_vitb14"
        self.files["encoders/dinov2_vitb14/encoder.json"] = json.dumps(metadata).encode()
        self.files["encoders/dinov2_vitb14/model.pt"] = b"trusted-weights"
        self.write_zip()
        with self.assertRaisesRegex(ValueError, "offline source files are missing"):
            prepare_model(self.archive, self.root / "dino-model", trust_model_source=True)


if __name__ == "__main__":
    unittest.main()
