"""Measure offline model startup and sequential warm CPU/GPU photo latency."""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from postcode_ml.inference import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("runtime/model_bundle"))
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    args = parser.parse_args()
    if args.count < 1 or args.threads < 1:
        parser.error("count and threads must be positive")
    torch.set_num_threads(args.threads)
    with args.test_csv.open(encoding="utf-8-sig", newline="") as stream:
        ids = [row["image_id"] for row in csv.DictReader(stream)][:args.count]
    if not ids:
        parser.error("test table is empty")
    started = time.perf_counter()
    predictor = Predictor(args.bundle, device=args.device)
    startup = time.perf_counter() - started
    paths = [args.images_dir / (key + ".jpg") for key in ids]
    # Keep initial kernel/cache warmup separate from both startup and warm timing.
    started = time.perf_counter()
    predictor.predict(paths[:1])
    first = time.perf_counter() - started
    rows = []
    for key, path in zip(ids, paths, strict=True):
        started = time.perf_counter()
        prediction = float(predictor.predict([path])[0])
        elapsed = time.perf_counter() - started
        rows.append({"image_id": key, "load_pct": prediction, "seconds": elapsed})
        print(f"{key}: {prediction:.6f}, {elapsed:.3f} seconds", flush=True)
    seconds = np.asarray([row["seconds"] for row in rows])
    result = {"model_version": predictor.model_version, "device": args.device,
              "threads": args.threads, "logical_cpus": os.cpu_count(),
              "processor": platform.processor(), "platform": platform.platform(),
              "python": sys.version, "packages": {key: importlib.metadata.version(key)
                  for key in ("numpy", "Pillow", "scikit-learn", "torch", "torchvision")},
              "startup_seconds": startup, "first_photo_seconds": first,
              "warm_count": len(rows), "warm_median_seconds": float(np.median(seconds)),
              "warm_p95_seconds": float(np.percentile(seconds, 95)),
              "warm_max_seconds": float(seconds.max()), "predictions": rows,
              "note": "Sequential requests, first N CSV images; excludes HTTP and queueing."}
    if args.reference:
        with args.reference.open(encoding="utf-8-sig", newline="") as stream:
            reference = {row["image_id"]: float(row["load_pct"]) for row in csv.DictReader(stream)}
        errors = [abs(row["load_pct"] - reference[row["image_id"]]) for row in rows]
        result["reference_max_abs_difference_pp"] = max(errors)
        result["reference_mean_abs_difference_pp"] = float(np.mean(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
