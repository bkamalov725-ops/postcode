"""One offline prediction implementation for CLI batches and persistent services.

Bundles contain executable Python source and pickle objects. Only load bundles
produced by a trusted training run, never arbitrary uploaded model files. Image
uploads must be validated separately by the API before calling this module.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import pickle
from pathlib import Path
import threading

import numpy as np

from .evaluation import OfficialRidge, baseline_features, predict_candidate


FEATURE_DIMENSIONS = {
    "baseline": {"baseline": 233},
    "dinov2_vitb14": {"global": 1536, "spatial": 4608},
    "convnext_tiny": {"global": 768, "spatial": 3840},
}


def _validate_manifest(manifest: dict) -> np.ndarray:
    if not isinstance(manifest, dict) or manifest.get("format_version", 1) != 1:
        raise ValueError("Unsupported model bundle format")
    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("Model bundle must contain at least one component")
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("Invalid model component")
        encoder, variant = component.get("encoder"), component.get("variant")
        if not isinstance(encoder, str) or encoder not in FEATURE_DIMENSIONS:
            raise ValueError(f"Unsupported component encoder: {encoder}")
        if not isinstance(variant, str) or variant not in FEATURE_DIMENSIONS[encoder]:
            raise ValueError(f"Unsupported feature variant for {encoder}: {variant}")
    raw_weights = manifest.get("weights")
    if not isinstance(raw_weights, list) or len(raw_weights) != len(components):
        raise ValueError("Component count and ensemble weights must match")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_weights):
        raise ValueError("Ensemble weights must be numbers")
    weights = np.asarray(raw_weights, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError("Ensemble weights must be finite, nonnegative and have a positive sum")
    if not isinstance(manifest.get("encoder_configs", {}), dict):
        raise ValueError("encoder_configs must be a mapping")
    return weights


def _validate_head(head, encoder: str, variant: str) -> None:
    if not callable(getattr(head, "predict", None)):
        raise ValueError("Every model head must have a predict method")
    expected = FEATURE_DIMENSIONS[encoder][variant]
    dimensions = getattr(head, "n_features_in_", None)
    if isinstance(head, OfficialRidge):
        for name in ("mean", "scale", "weights"):
            value = np.asarray(getattr(head, name))
            if value.shape != (expected,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid baseline head {name}")
        if np.any(np.asarray(head.scale) <= 0) or not np.isfinite(head.intercept):
            raise ValueError("Invalid baseline head scale or intercept")
        dimensions = len(head.mean)
    if dimensions is not None and dimensions != expected:
        raise ValueError(f"Head feature dimension mismatch: {dimensions} instead of {expected} for {encoder}/{variant}")


class Predictor:
    """Keep trusted regression heads and selected offline encoders in memory.

    Construct once at service startup, then call ``predict(paths)`` repeatedly.
    A lock serializes predictions so a single GPU encoder cannot be used by
    overlapping requests. Loading failure is fatal; no network or random-weight
    fallback is allowed. ``model_version`` fingerprints the validated manifest,
    heads and encoder manifests (which include verified weight/source hashes).

    ``device`` may be cpu, cuda, cuda:N or auto. CPU keeps float32, CUDA preserves
    the exported encoder's mixed-precision setting. Small CPU/GPU numerical
    differences are expected, just as in the original Kaggle extraction path.
    """

    def __init__(self, bundle_dir, device: str = "cpu"):
        self.bundle_dir = Path(bundle_dir)
        self._lock = threading.RLock()
        self._encoders = {}
        self._configs = {}
        manifest = json.loads((self.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        self._weights = _validate_manifest(manifest)
        self._components = tuple(dict(component) for component in manifest["components"])
        encoder_metadata = {}
        # Check all required artifacts before invoking any loader or unpickling.
        for component in self._components:
            encoder = component["encoder"]
            if encoder == "baseline" or encoder in self._configs:
                continue
            encoder_dir = self.bundle_dir / "encoders" / encoder
            if not all((encoder_dir / name).is_file() for name in ("encoder.json", "model.pt")):
                raise FileNotFoundError(f"Missing offline encoder: {encoder_dir}")
            from .vision import EncoderConfig
            configs = manifest.get("encoder_configs", {})
            if not isinstance(configs.get(encoder), dict):
                raise ValueError(f"Missing encoder config: {encoder}")
            config = EncoderConfig(**configs[encoder])
            if config.name != encoder:
                raise ValueError(f"Encoder config does not match component: {encoder}")
            config = replace(config, device=device)
            config.validate()
            self._configs[encoder] = config
            encoder_metadata[encoder] = json.loads((encoder_dir / "encoder.json").read_text(encoding="utf-8"))
        heads_bytes = (self.bundle_dir / "heads.pkl").read_bytes()
        # This is intentionally a trusted-local-artifact interface, not a model
        # upload parser. Checksums provide identity/integrity, not authenticity.
        heads = pickle.loads(heads_bytes)
        if not isinstance(heads, (list, tuple)) or len(heads) != len(self._components):
            raise ValueError("Component count and regression head count must match")
        for component, head in zip(self._components, heads, strict=True):
            _validate_head(head, component["encoder"], component["variant"])
        self._heads = tuple(heads)
        for encoder, config in self._configs.items():
            from .vision import load_encoder
            # Preflight guarantees the existing-artifact branch, which verifies
            # weight and source hashes rather than downloading replacements.
            self._encoders[encoder] = load_encoder(config, self.bundle_dir / "encoders" / encoder, offline_only=True)
        identity = {"manifest": manifest, "heads_sha256": hashlib.sha256(heads_bytes).hexdigest(),
                    "encoders": encoder_metadata}
        self.model_version = "sha256:" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()

    def predict(self, image_paths) -> np.ndarray:
        """Return one finite clipped percentage per path, preserving input order."""
        paths = [Path(path) for path in image_paths]
        if not paths:
            raise ValueError("At least one image is required")
        with self._lock:
            feature_sets = {}
            for component in self._components:
                encoder = component["encoder"]
                if encoder in feature_sets:
                    continue
                if encoder == "baseline":
                    feature_sets[encoder] = {"baseline": baseline_features(paths)}
                else:
                    from .vision import extract_features_with_model
                    feature_sets[encoder], _ = extract_features_with_model(
                        paths, self._configs[encoder], self._encoders[encoder])
            predictions = []
            for component, head in zip(self._components, self._heads, strict=True):
                encoder, variant = component["encoder"], component["variant"]
                x = feature_sets[encoder][variant]
                if x.shape != (len(paths), FEATURE_DIMENSIONS[encoder][variant]) or not np.isfinite(x).all():
                    raise ValueError(f"Invalid extracted features for {encoder}/{variant}")
                predictions.append(predict_candidate(head, x))
            result = np.average(np.stack(predictions), axis=0, weights=self._weights)
            if not np.isfinite(result).all():
                raise ValueError("Model produced non-finite predictions")
            return np.clip(result, 0.0, 100.0)


def predict_images(bundle_dir, image_paths, *, device="cpu"):
    """Compatibility wrapper; use a persistent Predictor for repeated requests."""
    return Predictor(bundle_dir, device=device).predict(image_paths)
