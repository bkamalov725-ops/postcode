"""Persistent, offline inference behavior without downloading any model weights."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from postcode_ml.evaluation import OfficialRidge
from postcode_ml.inference import Predictor, predict_images
from postcode_ml import vision


def constant_head(dimensions, value=42):
    return OfficialRidge(np.zeros(dimensions), np.ones(dimensions), np.zeros(dimensions), float(value))


def write_bundle(root, *, neural=False, value=42):
    encoder, variant, dimensions = ("dinov2_vitb14", "global", 1536) if neural else ("baseline", "baseline", 233)
    manifest = {"format_version": 1, "components": [{"encoder": encoder, "variant": variant}],
                "weights": [1.0], "encoder_configs": {}}
    if neural:
        manifest["encoder_configs"][encoder] = asdict(vision.EncoderConfig(image_size=224))
        artifact = root / "encoders" / encoder
        artifact.mkdir(parents=True)
        (artifact / "model.pt").write_bytes(b"test-only encoder")
        (artifact / "encoder.json").write_text(json.dumps({"weights_sha256": "test-only"}), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "heads.pkl").write_bytes(pickle.dumps([constant_head(dimensions, value)]))
    return manifest


class PredictorTests(unittest.TestCase):
    def test_persistent_baseline_uses_in_memory_heads_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_bundle(root)
            path = root / "image.png"
            Image.new("RGB", (25, 40), "white").save(path)
            predictor = Predictor(root)
            expected = predict_images(root, [path, path])
            # Requests remain independent of disk model files after startup.
            (root / "heads.pkl").unlink()
            np.testing.assert_array_equal(predictor.predict([path, path]), expected)
            np.testing.assert_array_equal(predictor.predict([path]), [42.])
            with self.assertRaisesRegex(ValueError, "At least one"):
                predictor.predict([])

    def test_selected_encoder_loaded_once_and_requests_serialized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_bundle(root, neural=True)
            state = {"active": 0, "maximum": 0}
            state_lock = threading.Lock()
            model = object()

            def extract(paths, config, loaded):
                self.assertIs(loaded, model)
                with state_lock:
                    state["active"] += 1
                    state["maximum"] = max(state["maximum"], state["active"])
                time.sleep(0.02)
                with state_lock:
                    state["active"] -= 1
                return {"global": np.zeros((len(paths), 1536), dtype=np.float32)}, {}

            with patch.object(vision, "load_encoder", return_value=model) as loader, \
                 patch.object(vision, "extract_features_with_model", side_effect=extract):
                predictor = Predictor(root)
                with ThreadPoolExecutor(max_workers=3) as pool:
                    predictions = list(pool.map(lambda _: predictor.predict([root / "image.jpg"]), range(3)))
            loader.assert_called_once()
            self.assertTrue(loader.call_args.kwargs["offline_only"])
            self.assertEqual(state["maximum"], 1)
            for result in predictions:
                np.testing.assert_array_equal(result, [42.])

    def test_invalid_manifest_and_head_counts_rejected_at_startup(self):
        for change in ({"weights": []}, {"weights": [-1]}, {"weights": [0]},
                       {"weights": [float("nan")]}, {"weights": [float("inf")]},
                       {"weights": [True]}, {"components": []}, {"format_version": 9},
                       {"components": [{"encoder": "../outside", "variant": "global"}]},
                       {"components": [{"encoder": "baseline", "variant": "spatial"}]}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = write_bundle(root)
                manifest.update(change)
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    Predictor(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_bundle(root)
            (root / "heads.pkl").write_bytes(pickle.dumps([]))
            with self.assertRaisesRegex(ValueError, "head count"):
                Predictor(root)
            (root / "heads.pkl").write_bytes(pickle.dumps([constant_head(3)]))
            with self.assertRaisesRegex(ValueError, "Invalid baseline head"):
                Predictor(root)

    def test_version_tracks_heads_and_ignores_directory_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            write_bundle(first)
            write_bundle(second)
            initial = Predictor(first).model_version
            self.assertEqual(initial, Predictor(second).model_version)
            write_bundle(second, value=43)
            self.assertNotEqual(initial, Predictor(second).model_version)

    def test_invalid_features_fail_instead_of_silent_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_bundle(root, neural=True)
            with patch.object(vision, "load_encoder", return_value=object()):
                predictor = Predictor(root)
            for matrix in (np.zeros((1, 7)), np.full((1, 1536), np.nan)):
                with patch.object(vision, "extract_features_with_model", return_value=({"global": matrix}, {})), \
                     self.assertRaisesRegex(ValueError, "Invalid extracted features"):
                    predictor.predict([root / "image.jpg"])

    def test_missing_artifact_offline_only_cannot_download(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(torch.hub, "load") as loader:
            with self.assertRaisesRegex(FileNotFoundError, "Missing offline encoder"):
                vision.load_encoder(vision.EncoderConfig(device="cpu"), Path(temporary), offline_only=True)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
