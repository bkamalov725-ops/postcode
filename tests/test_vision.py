"""CPU-only encoder contract checks; never download models or weights."""

from dataclasses import asdict
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from postcode_ml import vision


class FakeDino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward_features(self, pixels):
        count = (pixels.shape[-1] // 14) ** 2
        mean = pixels.mean((1, 2, 3)).reshape(-1, 1, 1)
        return {"x_norm_clstoken": mean[:, 0].repeat(1, 4),
                "x_norm_patchtokens": mean.repeat(1, count, 4)}


class VisionTests(unittest.TestCase):
    def test_letterbox_extreme_shapes_preserves_content_and_zero_padding(self):
        for width, height in [(3, 3000), (3000, 3), (600, 900), (900, 600), (50, 50)]:
            with self.subTest(width=width, height=height):
                pixels, masks = vision.preprocess_image(Image.new("RGB", (width, height), "white"), 224)
                self.assertEqual(tuple(pixels.shape), (3, 224, 224))
                self.assertEqual(tuple(masks.shape), (5, 256))
                self.assertTrue(torch.isfinite(pixels).all())
                self.assertTrue((masks.sum(1) > 0).all())
                self.assertTrue(torch.allclose(masks[0], masks[1:].sum(0), atol=1e-6))
                occupied = (pixels.abs().sum(0) > 0)
                ys, xs = torch.where(occupied)
                expected_width = max(1, round(width * 224 / max(width, height)))
                expected_height = max(1, round(height * 224 / max(width, height)))
                self.assertEqual(int(xs.max() - xs.min() + 1), expected_width)
                self.assertEqual(int(ys.max() - ys.min() + 1), expected_height)
                self.assertAlmostEqual(float(masks[0].sum() * 14 * 14), expected_width * expected_height, places=2)

    def test_exif_orientation_applied_before_resizing(self):
        original = Image.new("RGB", (40, 80), "white")
        original.getexif()[274] = 6
        pixels, _ = vision.preprocess_image(original, 224)
        occupied = pixels.abs().sum(0) > 0
        ys, xs = torch.where(occupied)
        self.assertEqual(int(xs.max() - xs.min() + 1), 224)
        self.assertEqual(int(ys.max() - ys.min() + 1), 112)

    def test_rgb_conversion_and_file_input_match(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Image.new("L", (23, 40), 128)
            path = Path(temp) / "gray.png"
            image.save(path)
            direct, direct_masks = vision.preprocess_image(image, 224)
            loaded, loaded_masks = vision.preprocess_image(path, 224)
            self.assertTrue(torch.equal(direct, loaded))
            self.assertTrue(torch.equal(direct_masks, loaded_masks))

    def test_dino_pooling_excludes_padding_and_orders_quadrants(self):
        # 4x4 patches: two left columns are content, two right columns padding.
        weights = vision.region_weights(56, 14, (0, 0, 28, 56)).unsqueeze(0)
        values = torch.tensor([[1, 2, 999, 999], [1, 2, 999, 999],
                               [3, 4, 999, 999], [3, 4, 999, 999]], dtype=torch.float32)
        outputs = {"x_norm_clstoken": torch.tensor([[7.0]]), "x_norm_patchtokens": values.reshape(1, 16, 1)}
        pooled = vision.pool_dino_features(outputs, weights)
        torch.testing.assert_close(pooled["global"], torch.tensor([[7.0, 2.5]]))
        torch.testing.assert_close(pooled["spatial"], torch.tensor([[7.0, 2.5, 1., 2., 3., 4.]]))

    def test_dino_real_dimensions_and_half_precision_pooling(self):
        outputs = {"x_norm_clstoken": torch.ones(2, 768, dtype=torch.float16),
                   "x_norm_patchtokens": torch.ones(2, 16, 768, dtype=torch.float16)}
        weights = vision.region_weights(56, 14, (0, 0, 56, 56)).repeat(2, 1, 1)
        result = vision.pool_dino_features(outputs, weights)
        self.assertEqual(tuple(result["global"].shape), (2, 1536))
        self.assertEqual(tuple(result["spatial"].shape), (2, 4608))
        self.assertEqual(result["spatial"].dtype, torch.float32)

    def test_convnext_dimensions_and_padding(self):
        weights = vision.region_weights(64, 32, (0, 0, 32, 64)).unsqueeze(0)
        features = torch.ones(1, 768, 2, 2)
        features[:, :, :, 1] = 999
        result = vision.pool_convnext_features(features, weights, torch.nn.Identity())
        self.assertEqual(tuple(result["global"].shape), (1, 768))
        self.assertEqual(tuple(result["spatial"].shape), (1, 3840))
        self.assertTrue(torch.equal(result["spatial"], torch.ones(1, 3840)))

    def test_invalid_config_and_pool_shape_rejected(self):
        for config in [vision.EncoderConfig(image_size=225), vision.EncoderConfig(batch_size=0),
                       vision.EncoderConfig(name="other"), vision.EncoderConfig(revision="main"),
                       vision.EncoderConfig(name="convnext_tiny", image_size=392)]:
            with self.assertRaises(ValueError):
                config.validate()
        with self.assertRaises(ValueError):
            vision.pool_dino_features({"x_norm_clstoken": torch.ones(1, 2), "x_norm_patchtokens": torch.ones(1, 4, 2)}, torch.ones(1, 5, 5))

    def _make_artifact(self, root, config):
        artifact = root / "encoder"
        source = artifact / "source"
        source.mkdir(parents=True)
        (source / "hubconf.py").write_text("# fake test source\n", encoding="utf-8")
        torch.save(FakeDino().state_dict(), artifact / "model.pt")
        metadata = {"artifact_version": vision.ARTIFACT_VERSION, "name": config.name,
                    "source_revision": config.revision, "source_repository": "https://github.com/facebookresearch/dinov2",
                    "weights_sha256": vision._sha256(artifact / "model.pt"), "source_files": vision._source_manifest(source),
                    "preprocessing": vision._preprocessing_metadata(config)}
        (artifact / "encoder.json").write_text(json.dumps(metadata), encoding="utf-8")
        return artifact

    def test_offline_loading_and_extraction_preserve_rows(self):
        config = vision.EncoderConfig(image_size=224, batch_size=2, device="cpu")
        self.assertEqual(json.loads(json.dumps(asdict(config)))["revision"], vision.DINO_REVISION)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = self._make_artifact(root, config)
            paths = []
            for index, color in enumerate(["black", "gray", "white"]):
                path = root / f"{index}.png"
                Image.new("RGB", (224, 224), color).save(path)
                paths.append(path)
            with patch.object(torch.hub, "load", return_value=FakeDino()) as loader:
                arrays, metadata = vision.extract_features(paths, config, artifact)
            self.assertEqual(loader.call_args.kwargs, {"source": "local", "pretrained": False})
            self.assertEqual(arrays["global"].shape, (3, 8))
            self.assertEqual(arrays["spatial"].shape, (3, 24))
            self.assertTrue(np.all(np.diff(arrays["global"][:, 0]) > 0))
            self.assertFalse(metadata["mixed_precision"])
            self.assertEqual(metadata["feature_dimensions"], {"global": 8, "spatial": 24})
            self.assertEqual(metadata["source_revision"], vision.DINO_REVISION)
            json.dumps(metadata)

    def test_invalid_artifact_cannot_trigger_network_fallback(self):
        config = vision.EncoderConfig(image_size=224, device="cpu")
        with tempfile.TemporaryDirectory() as temp:
            artifact = self._make_artifact(Path(temp), config)
            (artifact / "source" / "hubconf.py").write_text("# changed\n", encoding="utf-8")
            with patch.object(torch.hub, "load") as loader, self.assertRaisesRegex(ValueError, "checksum"):
                vision.load_encoder(config, artifact)
            loader.assert_not_called()

    def test_oom_retries_same_rows_with_smaller_batch(self):
        config = vision.EncoderConfig(image_size=224, batch_size=4, device="cuda:0")
        calls = []

        def forward(model, paths, encoder_config, device):
            calls.append(list(paths))
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return {"global": np.asarray(paths, dtype=np.float32).reshape(-1, 1)}

        with tempfile.TemporaryDirectory() as temp:
            artifact = self._make_artifact(Path(temp), config)
            with patch.object(vision, "_device", return_value=torch.device("cuda:0")), \
                 patch.object(vision, "load_encoder", return_value=FakeDino()), \
                 patch.object(vision, "_forward_batch", side_effect=forward), \
                 patch.object(torch.cuda, "device", return_value=nullcontext()), \
                 patch.object(torch.cuda, "empty_cache"):
                arrays, metadata = vision.extract_features([0, 1, 2, 3, 4], config, artifact)
            self.assertEqual(calls, [[0, 1, 2, 3], [0, 1], [2, 3], [4]])
            np.testing.assert_array_equal(arrays["global"][:, 0], np.arange(5))
            self.assertEqual(metadata["actual_batch_size"], 2)

    def test_generated_directories_are_checked_before_removal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(vision._checked_artifact_child(root, "source"), root.resolve() / "source")
            with self.assertRaises(ValueError):
                vision._checked_artifact_child(root, "../outside")

    def test_corrupted_image_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken.jpg"
            path.write_bytes(b"not an image")
            with self.assertRaises(OSError):
                vision.preprocess_image(path)


if __name__ == "__main__":
    unittest.main()
