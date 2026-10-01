"""Run the offline model as a persistent XML API."""
import argparse
import os

import torch
import uvicorn

from postcode_ml.service import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", default=os.getenv("POSTCODE_BUNDLE_DIR", "runtime/attention518"))
    parser.add_argument("--db-path", default=os.getenv("POSTCODE_DB_PATH", "runtime/assessments.sqlite3"))
    parser.add_argument("--device", default=os.getenv("POSTCODE_DEVICE", "cpu"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--threads", type=int, default=int(os.getenv("POSTCODE_THREADS", "4")))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be a positive integer")
    torch.set_num_threads(args.threads)
    app = create_app(args.bundle_dir, args.db_path, args.device)
    # One process keeps one encoder in memory. Predictor serializes inference;
    # request parsing, SQLite and health requests remain asynchronous.
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
