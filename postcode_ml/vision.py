"""Frozen image encoders with identical training and offline inference transforms.

The first extraction downloads official pretrained weights. It exports both the
weights and (for DINOv2) the pinned upstream Python source. Further extractions
from the artifact do not require Internet access. Random weights are never a
fallback for a failed pretrained download.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
import torch


DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class EncoderConfig:
    name: str = "dinov2_vitb14"
    image_size: int = 392
    batch_size: int = 24
    device: str = "auto"
    amp: bool = True
    revision: str = DINO_REVISION

    def validate(self) -> None:
        if self.name not in {"dinov2_vitb14", "convnext_tiny"}:
            raise ValueError(f"Unsupported encoder: {self.name}")
        stride = 14 if self.name == "dinov2_vitb14" else 32
        if self.image_size < stride or self.image_size % stride:
            raise ValueError(f"image_size must be a positive multiple of {stride}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.name == "dinov2_vitb14" and not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("DINOv2 source revision must be a full immutable Git commit SHA")


def _device(config: EncoderConfig) -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are CPU and CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; enable a Kaggle GPU or select device='cpu'")
    return device


def region_weights(image_size: int, stride: int, content_box: Sequence[int]) -> torch.Tensor:
    """Fractional patch overlap for the full image and its four content quadrants.

    The regions are defined in the resized *content*, not the padded square.
    Boundary patches may contribute to two regions proportionally. Padding has
    zero weight, although it still participates in the encoder's receptive field
    and DINO attention (including its CLS token).
    """
    if image_size % stride or stride <= 0:
        raise ValueError("image_size must be divisible by a positive stride")
    left, top, right, bottom = map(float, content_box)
    if not (0 <= left < right <= image_size and 0 <= top < bottom <= image_size):
        raise ValueError("content_box must be a nonempty rectangle inside the image")
    middle_x, middle_y = (left + right) / 2, (top + bottom) / 2
    regions = [(left, top, right, bottom), (left, top, middle_x, middle_y),
               (middle_x, top, right, middle_y), (left, middle_y, middle_x, bottom),
               (middle_x, middle_y, right, bottom)]
    starts = torch.arange(image_size // stride, dtype=torch.float32) * stride
    ends = starts + stride
    masks = []
    for x0, y0, x1, y1 in regions:
        x_overlap = (torch.minimum(ends, torch.tensor(x1)) - torch.maximum(starts, torch.tensor(x0))).clamp_min(0)
        y_overlap = (torch.minimum(ends, torch.tensor(y1)) - torch.maximum(starts, torch.tensor(y0))).clamp_min(0)
        masks.append((y_overlap[:, None] * x_overlap[None, :] / (stride * stride)).flatten())
    return torch.stack(masks)


def preprocess_image(image: Image.Image | str | Path, image_size: int = 392, stride: int = 14) -> tuple[torch.Tensor, torch.Tensor]:
    """EXIF transpose, RGB, aspect-preserving letterbox, ImageNet normalization.

    Returns CHW float32 pixels and 5xN fractional region masks. No crop, stretch,
    mirroring, or other stochastic augmentation is used for frozen features.
    """
    if image_size < stride or stride <= 0 or image_size % stride:
        raise ValueError("Invalid image size / encoder stride")
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            opened.load()
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
    else:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    width, height = rgb.size
    if width < 1 or height < 1:
        raise ValueError("Image must be nonempty")
    scale = image_size / max(width, height)
    resized_width = max(1, min(image_size, round(width * scale)))
    resized_height = max(1, min(image_size, round(height * scale)))
    resized = rgb.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    left, top = (image_size - resized_width) // 2, (image_size - resized_height) // 2
    # Float canvas gives padding exactly zero after normalization (no uint8 rounding).
    canvas = np.empty((image_size, image_size, 3), dtype=np.float32)
    canvas[:] = IMAGENET_MEAN
    canvas[top:top + resized_height, left:left + resized_width] = np.asarray(resized, dtype=np.float32) / 255.0
    canvas = (canvas - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    pixels = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1)))
    masks = region_weights(image_size, stride, (left, top, left + resized_width, top + resized_height))
    return pixels, masks


def pool_dino_features(outputs: Mapping[str, torch.Tensor], weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """CLS + valid patch mean, optionally concatenating four regional patch means."""
    patches = outputs["x_norm_patchtokens"].float()
    cls = outputs["x_norm_clstoken"].float()
    if patches.ndim != 3 or weights.shape != (patches.shape[0], 5, patches.shape[1]):
        raise ValueError("DINO patch tokens and region masks have incompatible shapes")
    weights = weights.to(device=patches.device, dtype=torch.float32)
    regional = torch.bmm(weights, patches) / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    global_features = torch.cat((cls, regional[:, 0]), dim=1)
    return {"global": global_features, "spatial": torch.cat((cls, regional.flatten(1)), dim=1)}


def pool_convnext_features(features: torch.Tensor, weights: torch.Tensor, normalization: torch.nn.Module) -> dict[str, torch.Tensor]:
    patches = features.float().flatten(2).transpose(1, 2)
    if weights.shape != (patches.shape[0], 5, patches.shape[1]):
        raise ValueError("ConvNeXt feature map and region masks have incompatible shapes")
    weights = weights.to(device=patches.device, dtype=torch.float32)
    regional = torch.bmm(weights, patches) / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    batch, regions, channels = regional.shape
    regional = normalization(regional.reshape(batch * regions, channels, 1, 1)).reshape(batch, regions, channels)
    return {"global": regional[:, 0], "spatial": regional.flatten(1)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_manifest(path: Path) -> dict[str, str]:
    return {file.relative_to(path).as_posix(): _sha256(file) for file in sorted(path.rglob("*"))
            if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc"}


def _checked_artifact_child(artifact_dir: Path, name: str) -> Path:
    """Resolve a generated direct child before any recursive removal or move."""
    if name not in {"source", "source.tmp"}:
        raise ValueError("Unexpected generated directory name")
    parent = artifact_dir.resolve(strict=True)
    child = parent / name
    if child.is_symlink() or (hasattr(child, "is_junction") and child.is_junction()):
        raise ValueError(f"Refusing to modify a linked source directory: {child}")
    resolved = child.resolve()
    if resolved.parent != parent or resolved.name != name:
        raise ValueError(f"Generated source path escaped its artifact directory: {resolved}")
    return resolved


def _preprocessing_metadata(config: EncoderConfig) -> dict:
    return {"version": 1, "image_size": config.image_size, "exif_transpose": True, "color": "RGB",
            "resize": "bicubic longest side; preserve aspect ratio; centered letterbox",
            "padding": "exact ImageNet mean before normalization", "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
            "pooling": "fractional valid-content overlap; 2x2 regions relative to content box; row-major quadrants",
            "padding_limitation": "Padding still participates in receptive fields and DINO attention; only pooling excludes it.",
            "global": "CLS + valid patch mean" if config.name == "dinov2_vitb14" else "valid spatial mean + pretrained classifier LayerNorm",
            "spatial": "global + four regional feature vectors"}


def _read_artifact(artifact_dir: Path, config: EncoderConfig) -> dict:
    manifest_path = artifact_dir / "encoder.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_version") != ARTIFACT_VERSION or metadata.get("name") != config.name:
        raise ValueError(f"Incompatible encoder artifact: {manifest_path}")
    if metadata.get("preprocessing") != _preprocessing_metadata(config):
        raise ValueError("Encoder artifact preprocessing differs from requested config; use a separate artifact directory")
    if config.name == "dinov2_vitb14" and metadata.get("source_revision") != config.revision:
        raise ValueError("Encoder artifact has a different DINOv2 source revision")
    if _sha256(artifact_dir / "model.pt") != metadata.get("weights_sha256"):
        raise ValueError("Encoder weights checksum mismatch")
    if config.name == "dinov2_vitb14" and _source_manifest(artifact_dir / "source") != metadata.get("source_files"):
        raise ValueError("DINOv2 source snapshot checksum mismatch")
    return metadata


def load_encoder(config: EncoderConfig, artifact_dir: str | Path, *, offline_only: bool = False) -> torch.nn.Module:
    """Load a validated offline artifact, or create it from official pretrained weights.

    ``encoder.json`` is written last as the completion marker. An existing invalid
    artifact raises an error rather than silently downloading/replacing weights.
    The artifact includes the complete state dict (even the unused classifier).
    Set ``offline_only=True`` in services to prohibit even an initial download.
    """
    config.validate()
    device = _device(config)
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "encoder.json"
    source_path = None
    torchvision_version = None
    if manifest_path.is_file():
        _read_artifact(artifact_dir, config)
        if config.name == "dinov2_vitb14":
            model = torch.hub.load(str(artifact_dir / "source"), config.name, source="local", pretrained=False)
        else:
            from torchvision.models import convnext_tiny
            model = convnext_tiny(weights=None)
        state = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
    else:
        if offline_only:
            raise FileNotFoundError(f"Missing offline encoder: {artifact_dir}")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if config.name == "dinov2_vitb14":
            # SHA is verified/pinned in source. skip_validation avoids GitHub's
            # optional branch-list API lookup/rate limit for an immutable commit.
            model = torch.hub.load(f"facebookresearch/dinov2:{config.revision}", config.name,
                                   pretrained=True, trust_repo=True, skip_validation=True)
            source_path = Path(torch.hub.get_dir()) / f"facebookresearch_dinov2_{config.revision}"
            if not (source_path / "hubconf.py").is_file():
                raise RuntimeError("Official DINOv2 source cache not found; cannot export a self-contained artifact")
        else:
            import torchvision
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
            torchvision_version = torchvision.__version__
            model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model = model.cpu().eval()
        weights_temp = artifact_dir / "model.pt.tmp"
        torch.save(model.state_dict(), weights_temp)
        os.replace(weights_temp, artifact_dir / "model.pt")
        source_files = None
        if source_path is not None:
            # Copy to an empty temporary directory; never overlay an old source
            # tree left by an interrupted run with potentially stale modules.
            if source_path.is_symlink() or (hasattr(source_path, "is_junction") and source_path.is_junction()):
                raise ValueError("Refusing to copy a linked DINOv2 source directory")
            for source_child in source_path.rglob("*"):
                if source_child.is_symlink() or (hasattr(source_child, "is_junction") and source_child.is_junction()):
                    raise ValueError(f"Refusing to follow a link in DINOv2 source: {source_child}")
            source_temp = _checked_artifact_child(artifact_dir, "source.tmp")
            source_target = _checked_artifact_child(artifact_dir, "source")
            if source_temp.exists():
                shutil.rmtree(source_temp)
            shutil.copytree(source_path, source_temp, ignore=shutil.ignore_patterns(".git", ".github", "__pycache__", "*.pyc"))
            if source_target.exists():
                shutil.rmtree(source_target)
            source_temp.rename(source_target)
            source_files = _source_manifest(source_target)
        metadata = {"artifact_version": ARTIFACT_VERSION, "name": config.name,
                    "source_repository": "https://github.com/facebookresearch/dinov2" if source_path else "https://github.com/pytorch/vision",
                    "source_revision": config.revision if source_path else None,
                    "weights": "official DINOv2 pretrained" if source_path else "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
                    "weights_sha256": _sha256(artifact_dir / "model.pt"), "source_files": source_files,
                    "torch_version": torch.__version__, "torchvision_version": torchvision_version,
                    "preprocessing": _preprocessing_metadata(config)}
        metadata_temp = artifact_dir / "encoder.json.tmp"
        metadata_temp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(metadata_temp, manifest_path)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval()


def _forward_batch(model: torch.nn.Module, image_paths: Sequence[str | Path | Image.Image], config: EncoderConfig, device: torch.device) -> dict[str, np.ndarray]:
    stride = 14 if config.name == "dinov2_vitb14" else 32
    prepared = [preprocess_image(path, config.image_size, stride) for path in image_paths]
    pixels = torch.stack([item[0] for item in prepared]).to(device)
    weights = torch.stack([item[1] for item in prepared]).to(device)
    context = torch.autocast(device_type="cuda", dtype=torch.float16) if config.amp and device.type == "cuda" else nullcontext()
    with torch.inference_mode(), context:
        if config.name == "dinov2_vitb14":
            outputs = model.forward_features(pixels)
        else:
            outputs = model.features(pixels)
    # Pooling and accumulation explicitly use float32, outside mixed precision.
    with torch.inference_mode():
        pooled = pool_dino_features(outputs, weights) if config.name == "dinov2_vitb14" else pool_convnext_features(outputs, weights, model.classifier[0])
        result = {key: value.cpu().numpy().astype(np.float32, copy=False) for key, value in pooled.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Encoder produced non-finite features; rerun with amp=False")
    return result


def extract_features_with_model(
    image_paths: Sequence[str | Path | Image.Image], config: EncoderConfig,
    model: torch.nn.Module, *, progress: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    """Extract with an already loaded encoder without loading or releasing weights.

    The caller owns the model, must keep it in evaluation mode on the configured
    device, and must serialize concurrent use. This shares preprocessing, pooling,
    mixed precision and CUDA OOM retries with the training extraction path.
    """
    config.validate()
    paths = list(image_paths)
    if not paths:
        raise ValueError("At least one image is required")
    device = _device(config)
    arrays: dict[str, np.ndarray] = {}
    batch_size = min(config.batch_size, len(paths))
    start = 0
    while start < len(paths):
        stop = min(start + batch_size, len(paths))
        oom_message = None
        try:
            batch = _forward_batch(model, paths[start:stop], config, device)
        except RuntimeError as error:
            if device.type != "cuda" or "out of memory" not in str(error).lower():
                raise
            # Leave the except scope before empty_cache: its traceback would
            # otherwise retain the failed batch's CUDA tensors.
            oom_message = str(error)
        if oom_message is not None:
            if batch_size <= 1:
                raise RuntimeError(f"CUDA OOM even at batch_size=1; use a smaller compatible model bundle. {oom_message}")
            batch_size = max(1, batch_size // 2)
            gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            print(f"[{config.name}] CUDA OOM: retrying at batch_size={batch_size}", flush=True)
            continue
        for key, value in batch.items():
            if key not in arrays:
                arrays[key] = np.empty((len(paths), value.shape[1]), dtype=np.float32)
            arrays[key][start:stop] = value
        start = stop
        if progress:
            print(f"[{config.name}] features {start}/{len(paths)}", flush=True)
    metadata = {"config": asdict(config), "resolved_device": str(device), "actual_batch_size": batch_size,
                "count": len(paths), "feature_dimensions": {key: value.shape[1] for key, value in arrays.items()},
                "mixed_precision": bool(config.amp and device.type == "cuda")}
    return arrays, metadata


def extract_features(image_paths: Sequence[str | Path], config: EncoderConfig, artifact_dir: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Extract row-aligned float32 global/spatial vectors, reducing CUDA OOM batches.

    Call once with train+test paths if desired; no labels are used here. Encoder
    outputs are raw features: fitting any regressor/scaler belongs inside each
    training fold. Existing exported artifacts are always loaded offline.
    """
    config.validate()
    paths = list(image_paths)
    if not paths:
        raise ValueError("At least one image is required")
    device = _device(config)
    model = load_encoder(config, artifact_dir)
    try:
        arrays, runtime = extract_features_with_model(paths, config, model, progress=True)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    metadata = json.loads((Path(artifact_dir) / "encoder.json").read_text(encoding="utf-8"))
    return arrays, {**metadata, **runtime}
