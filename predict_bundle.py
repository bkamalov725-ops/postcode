"""Predict with an extracted, trusted model bundle without network downloads."""
import argparse
import json
from pathlib import Path

from postcode_ml.inference import predict_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    predictions = predict_images(args.bundle, args.images, device=args.device)
    print(json.dumps([{"image": str(p), "load_pct": float(v)}
                      for p, v in zip(args.images, predictions)], ensure_ascii=False))
