"""Regenerate a competition submission using one cached, offline Predictor."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import tempfile

from postcode_ml.data import _read_table
from postcode_ml.evaluation import validate_submission


def generate_submission(bundle: str | Path, test_csv: str | Path, images_dir: str | Path,
                        output: str | Path, *, device: str = "cpu", expected_count: int = 307,
                        batch_size: int = 4, threads: int = 4, predictor_factory=None) -> Path:
    """Publish only a completely validated CSV, preserving old output on failure.

    The model bundle is trusted executable model material (pickle/Python code).
    Never point this command at a bundle uploaded by an untrusted API user.
    """
    if expected_count <= 0 or batch_size <= 0 or threads <= 0:
        raise ValueError("Expected count, batch size and threads must be positive")
    test_csv, images_dir = Path(test_csv).resolve(), Path(images_dir).resolve()
    bundle, output = Path(bundle).resolve(), Path(output).absolute()
    if output.is_symlink():
        raise ValueError("Output must not be a symbolic link")
    output = output.resolve()
    ids = [row["image_id"] for row in _read_table(test_csv, ["image_id"])]
    if len(ids) != expected_count:
        raise ValueError(f"Expected {expected_count} test IDs, got {len(ids)}")
    if len({i.casefold() for i in ids}) != len(ids):
        raise ValueError("Test image IDs collide on a case-insensitive filesystem")
    paths = [images_dir / f"{image_id}.jpg" for image_id in ids]
    for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(images_dir):
            raise ValueError(f"Missing image or image outside images directory: {path}")
    protected = {test_csv, *(path.resolve() for path in paths)}
    if (output in protected or output.is_relative_to(bundle) or output.is_relative_to(images_dir)
            or (output.exists() and any(os.path.samefile(output, p) for p in protected))):
        raise ValueError("Output must not overwrite an input, image, or model bundle file")
    if predictor_factory is None:
        import torch
        from postcode_ml.inference import Predictor
        torch.set_num_threads(threads)
        predictor_factory = Predictor
    predictor = predictor_factory(bundle, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "load_pct"])
            for start in range(0, len(ids), batch_size):
                batch_ids, batch_paths = ids[start:start + batch_size], paths[start:start + batch_size]
                values = list(predictor.predict(batch_paths))
                if len(values) != len(batch_ids):
                    raise ValueError("Model returned a different number of predictions than input images")
                for image_id, value in zip(batch_ids, values, strict=True):
                    value = float(value)
                    if not math.isfinite(value) or not 0 <= value <= 100:
                        raise ValueError("Model predictions must be finite in [0, 100]")
                    writer.writerow([image_id, f"{value:.6f}"])
                print(f"Predicted {min(start + batch_size, len(ids))}/{len(ids)}", flush=True)
            stream.flush()
            os.fsync(stream.fileno())
        validate_submission(temporary, ids)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {output}: {len(ids)} unique IDs; model {predictor.model_version}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("runtime/model_bundle"))
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads used by PyTorch")
    parser.add_argument("--expected-count", type=int, default=307,
                        help="Override only for a deliberately custom test dataset")
    args = parser.parse_args()
    generate_submission(args.bundle, args.test_csv, args.images_dir, args.output, device=args.device,
                        batch_size=args.batch_size, threads=args.threads, expected_count=args.expected_count)


if __name__ == "__main__":
    main()
