import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from postcode_ml.evaluation import baseline_features, fit_candidate, predict_candidate
from postcode_ml.inference import predict_images


class BundleTests(unittest.TestCase):
    def test_offline_baseline_bundle_matches_batch_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, color in enumerate(("red", "blue", "green")):
                path = root / f"{index}.jpg"
                Image.new("RGB", (100 + index * 30, 200), color).save(path)
                paths.append(path)
            x = baseline_features(paths)
            head = fit_candidate({"kind": "official_ridge", "alpha": 100}, x, np.array([5., 40., 95.]))
            (root / "manifest.json").write_text(json.dumps({
                "components": [{"encoder": "baseline", "variant": "baseline"}],
                "weights": [1.0], "encoder_configs": {}}))
            with (root / "heads.pkl").open("wb") as stream:
                pickle.dump([head], stream)
            np.testing.assert_allclose(predict_images(root, paths), predict_candidate(head, x))

    def test_incomplete_offline_encoder_never_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "encoders" / "dinov2_vitb14").mkdir(parents=True)
            (root / "manifest.json").write_text(json.dumps({
                "components": [{"encoder": "dinov2_vitb14", "variant": "global"}],
                "weights": [1.0], "encoder_configs": {"dinov2_vitb14": {"name": "dinov2_vitb14"}}}))
            with (root / "heads.pkl").open("wb") as stream:
                pickle.dump([None], stream)
            with patch("postcode_ml.vision.extract_features") as extract:
                with self.assertRaisesRegex(FileNotFoundError, "Missing offline encoder"):
                    predict_images(root, [root / "some.jpg"])
                extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
