"""First competition experiment: fixed grouped CV and frozen image encoders."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import pickle
import platform
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np

from .data import load_dataset, load_saved_folds, make_folds
from .evaluation import (baseline_features, evaluate_baselines, evaluate_candidates,
                         fit_candidate, metrics, predict_candidate, write_submission,
                         validate_submission, refined_candidate_specs)


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=json_default, allow_nan=False), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_table(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset_manifest(dataset):
    entries = []
    for split, ids, paths in (("train", dataset.train_ids, dataset.train_paths),
                              ("test", dataset.test_ids, dataset.test_paths)):
        for image_id, path in zip(ids, paths, strict=True):
            entries.append({"split": split, "image_id": image_id, "sha256": file_hash(path)})
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"image_fingerprint": digest, "images": entries,
            "tables": {name: file_hash(getattr(dataset, name)) for name in
                       ("train_csv", "groups_csv", "test_csv", "sample_csv")}}


def _prepare_folds(dataset, output, seed, folds_csv=None):
    """Validate/reuse a saved split or generate it, then record its provenance."""
    if folds_csv is not None:
        source = Path(folds_csv).expanduser().resolve()
        fold_ids = load_saved_folds(source, dataset, n_splits=5)
        provenance = {"source": "saved_csv", "path": str(source),
                      "source_sha256": file_hash(source), "n_splits": 5}
    else:
        fold_ids = make_folds(dataset.y, dataset.groups, n_splits=5, seed=seed)
        provenance = {"source": "StratifiedGroupKFold", "seed": seed, "n_splits": 5,
                      "scikit_learn_version": importlib.metadata.version("scikit-learn")}
    exported = Path(output) / "folds.csv"
    write_table(exported, ["image_id", "load_pct", "group_id", "fold"],
                [{"image_id": key, "load_pct": float(y), "group_id": str(g), "fold": int(f)}
                 for key, y, g, f in zip(dataset.train_ids, dataset.y, dataset.groups, fold_ids, strict=True)])
    provenance["exported_file"] = "folds.csv"
    provenance["exported_sha256"] = file_hash(exported)
    return fold_ids, provenance


def encoder_worker(payload):
    # Spawned process owns one GPU and leaves no live CUDA model in the parent.
    import torch
    from .vision import EncoderConfig, extract_features, _read_artifact
    torch.set_num_threads(2)
    config = EncoderConfig(**payload["config"])
    cache = Path(payload["cache"])
    signature = hashlib.sha256(json.dumps({"config": payload["config"],
                                          "data": payload["fingerprint"],
                                          "vision_code": file_hash(Path(__file__).with_name("vision.py"))},
                                         sort_keys=True).encode()).hexdigest()
    metadata_path = cache.with_suffix(".json")
    if cache.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifact_dir = Path(payload["artifact_dir"])
        if metadata.get("signature") == signature and (artifact_dir / "encoder.json").is_file():
            _read_artifact(artifact_dir, config)
            print(f"[{config.name}] reusing validated feature cache", flush=True)
            return {"name": config.name, "cache": str(cache), "metadata": metadata}
    started = time.perf_counter()
    features, metadata = extract_features(payload["paths"], config,
                                          artifact_dir=Path(payload["artifact_dir"]))
    for name, values in features.items():
        if values.ndim != 2 or len(values) != len(payload["paths"]) or not np.isfinite(values).all():
            raise ValueError(f"Invalid features from {config.name}/{name}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **features)
    metadata = dict(metadata, signature=signature, elapsed_seconds=time.perf_counter() - started,
                    config=payload["config"], image_count=len(payload["paths"]))
    save_json(metadata_path, metadata)
    return {"name": config.name, "cache": str(cache), "metadata": metadata}


def export_archives(output):
    output = Path(output)
    with zipfile.ZipFile(output / "postcode_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            relative = path.relative_to(output)
            if (path.is_file() and path.suffix != ".zip" and "model_bundle" not in relative.parts
                    and "__pycache__" not in relative.parts):
                archive.write(path, str(relative))
    bundle = output / "model_bundle"
    if (bundle / "manifest.json").exists():
        with zipfile.ZipFile(output / "postcode_model.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, str(Path("model_bundle") / path.relative_to(bundle)))


def copy_source(destination):
    root = Path(__file__).resolve().parent.parent
    destination.mkdir(parents=True, exist_ok=True)
    package = destination / "postcode_ml"
    package.mkdir(exist_ok=True)
    for path in (root / "postcode_ml").glob("*.py"):
        shutil.copy2(path, package / path.name)
    for name in ("run_experiment.py", "run_improved.py", "predict_bundle.py", "requirements.txt"):
        if (root / name).is_file():
            shutil.copy2(root / name, destination / name)


def run(args):
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / "model_bundle"
    bundle.mkdir(exist_ok=True)
    packages = ("numpy", "Pillow", "scikit-learn", "torch", "torchvision")
    environment = {"python": sys.version, "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in packages}}
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "environment": environment,
              "status": "running", "seed": args.seed, "requested_encoders": args.encoders,
              "validation_note": "Fixed stratified grouped 5-fold selection CV; not an untouched holdout. "
              "All scaler/head fitting uses training folds only. No test targets are available.",
              "encoder_errors": {}, "encoders": {}, "candidates": []}
    save_json(output / "report.json", report)
    copy_source(output / "source")
    print("Checking all CSVs and decoding every image...", flush=True)
    dataset = load_dataset(args.data_root, expected_train=getattr(args, "expected_train", 716), verify_images=True)
    manifest = dataset_manifest(dataset)
    save_json(output / "dataset_manifest.json", manifest)
    fold_ids, report["folds"] = _prepare_folds(
        dataset, output, args.seed, folds_csv=getattr(args, "folds_csv", None))
    report["data"] = {"train_count": len(dataset.y), "test_count": len(dataset.test_ids),
                      "group_count": len(set(dataset.groups)),
                      "fingerprint": manifest["image_fingerprint"]}
    save_json(output / "report.json", report)
    print("Computing the official 233-feature baseline on the fixed folds...", flush=True)
    baseline_train = baseline_features(dataset.train_paths)
    baseline_test = baseline_features(dataset.test_paths)
    feature_sets = {"baseline": {"baseline": (baseline_train, baseline_test)}}
    candidates = []
    for result in evaluate_baselines(baseline_train, dataset.y, fold_ids):
        result.update(encoder="baseline", variant="baseline", key="baseline__" + result["name"])
        candidates.append(result)
        print(f"{result['key']}: MAE={result['metrics']['mae']:.4f}", flush=True)
    save_json(output / "baseline_report.json", candidates)

    encoder_configs = {}
    if not args.baseline_only:
        import torch
        from .vision import EncoderConfig
        gpu_count = torch.cuda.device_count()
        environment["gpus"] = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        if gpu_count == 0 and not args.allow_cpu:
            raise RuntimeError("No GPU detected. Enable Kaggle GPU accelerator, then rerun. "
                               "For a local baseline use --baseline-only.")
        payloads = []
        for index, name in enumerate(args.encoders):
            device = f"cuda:{index % gpu_count}" if gpu_count else "cpu"
            config = EncoderConfig(name=name, image_size=getattr(args, 'dino_image_size', 392) if name.startswith("dinov2") else 384,
                                   batch_size=args.batch_size, device=device, amp=gpu_count > 0)
            encoder_configs[name] = asdict(config)
            payloads.append({"config": asdict(config), "fingerprint": manifest["image_fingerprint"],
                             "paths": [str(p) for p in (*dataset.train_paths, *dataset.test_paths)],
                             "cache": str(output / "features" / (name + ".npz")),
                             "artifact_dir": str(bundle / "encoders" / name)})
        extracted = []
        if gpu_count >= 2 and len(payloads) >= 2:
            print("Extracting frozen features in independent GPU processes...", flush=True)
            with ProcessPoolExecutor(max_workers=min(gpu_count, len(payloads)),
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(encoder_worker, item): item["config"]["name"] for item in payloads}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        extracted.append(future.result())
                    except Exception:
                        report["encoder_errors"][name] = traceback.format_exc()
                        print(report["encoder_errors"][name], flush=True)
        else:
            for item in payloads:
                name = item["config"]["name"]
                try:
                    extracted.append(encoder_worker(item))
                except Exception:
                    report["encoder_errors"][name] = traceback.format_exc()
                    print(report["encoder_errors"][name], flush=True)
        # Keep candidate ordering stable even when GPUs finish in a different order.
        for extracted_item in sorted(extracted, key=lambda item: item["name"]):
            name = extracted_item["name"]
            report["encoders"][name] = extracted_item["metadata"]
            feature_sets[name] = {}
            with np.load(extracted_item["cache"], allow_pickle=False) as saved:
                for variant in sorted(saved.files):
                    matrix = saved[variant]
                    x, test_x = matrix[:len(dataset.y)], matrix[len(dataset.y):]
                    feature_sets[name][variant] = (x, test_x)
                    print(f"Evaluating {name}/{variant}: {matrix.shape[1]} features", flush=True)
                    specs = refined_candidate_specs(x.shape[1]) if getattr(args, 'search_profile', 'original') == 'refined' else None
                    for result in evaluate_candidates(x, dataset.y, fold_ids, specs):
                        result.update(encoder=name, variant=variant,
                                      key=f"{name}__{variant}__{result['name']}")
                        candidates.append(result)
                        print(f"  {result['key']}: MAE={result['metrics']['mae']:.4f}", flush=True)

    candidates.sort(key=lambda result: (result["metrics"]["mae"], result["key"]))
    best = candidates[0]
    components = [best]
    weights = [1.0]
    selected_oof = best["oof"]
    selection_name = best["key"]
    # One predeclared ensemble experiment; no large search of mixture weights.
    neural_winners = []
    for encoder in sorted(report["encoders"]):
        neural_winners.append(min((c for c in candidates if c["encoder"] == encoder),
                                  key=lambda c: c["metrics"]["mae"]))
    if len(neural_winners) == 2:
        ensemble_oof = np.mean([c["oof"] for c in neural_winners], axis=0)
        ensemble_metrics = metrics(dataset.y, ensemble_oof)
        report["ensemble_trial"] = {"components": [c["key"] for c in neural_winners],
                                    "weights": [0.5, 0.5], "metrics": ensemble_metrics}
        if ensemble_metrics["mae"] < best["metrics"]["mae"] - 0.05:
            components, weights = neural_winners, [0.5, 0.5]
            selected_oof = ensemble_oof
            selection_name = "mean_of_two_encoders"
    report["candidates"] = [{k: v for k, v in c.items() if k != "oof"} for c in candidates]
    report["selection"] = {"name": selection_name, "components": [c["key"] for c in components],
                           "weights": weights, "metrics": metrics(dataset.y, selected_oof)}
    print("Selected:", selection_name, report["selection"]["metrics"], flush=True)
    (output / "submissions").mkdir(exist_ok=True)
    final_heads, final_predictions = [], []
    for component in components:
        train_x, test_x = feature_sets[component["encoder"]][component["variant"]]
        head = fit_candidate(component["spec"], train_x, dataset.y)
        predictions = predict_candidate(head, test_x)
        final_heads.append(head)
        final_predictions.append(predictions)
        write_submission(output / "submissions" / (component["key"] + ".csv"), dataset, predictions)
    final_predictions = np.average(final_predictions, axis=0, weights=weights)
    write_submission(output / "submission.csv", dataset, final_predictions)
    validate_submission(output / "submission.csv", dataset.test_ids)
    with (bundle / "heads.pkl").open("wb") as stream:
        pickle.dump(final_heads, stream, protocol=4)
    save_json(bundle / "manifest.json", {"format_version": 1, "selection": selection_name,
              "components": [{k: c[k] for k in ("key", "encoder", "variant", "spec")} for c in components],
              "weights": weights, "encoder_configs": encoder_configs, "environment": environment,
              "trust_note": "Only load this pickle from a trusted run. Never accept arbitrary pickle uploads.",
              "inference_note": "GPU extraction uses autocast; CPU float32 can have small numerical differences."})
    copy_source(bundle)
    (bundle / "requirements-lock.txt").write_text("\n".join(
        f"{key}=={value}" for key, value in environment["packages"].items()) + "\n", encoding="utf-8")
    save_json(bundle / "example_predictions.json", [{"image_id": key, "load_pct": float(value)}
              for key, value in zip(dataset.test_ids[:5], final_predictions[:5], strict=True)])
    write_table(output / "leaderboard.csv", ["candidate", "mae", "within_10_pct"],
                [{"candidate": c["key"], "mae": c["metrics"]["mae"],
                  "within_10_pct": c["metrics"]["within_10_pct"]} for c in candidates])
    fields = ["image_id", "load_pct", "fold", "selected"] + [c["key"] for c in candidates]
    write_table(output / "oof_predictions.csv", fields,
                [dict(image_id=key, load_pct=float(dataset.y[i]), fold=int(fold_ids[i]),
                      selected=float(selected_oof[i]), **{c["key"]: float(c["oof"][i]) for c in candidates})
                 for i, key in enumerate(dataset.train_ids)])
    worst = np.argsort(-np.abs(dataset.y - selected_oof))[:50]
    write_table(output / "worst_errors.csv", ["image_id", "load_pct", "prediction", "absolute_error", "fold"],
                [{"image_id": dataset.train_ids[i], "load_pct": float(dataset.y[i]),
                  "prediction": float(selected_oof[i]), "absolute_error": float(abs(dataset.y[i] - selected_oof[i])),
                  "fold": int(fold_ids[i])} for i in worst])
    # Verify serialized final heads before publishing downloadable artifacts.
    with (bundle / "heads.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    restored_predictions = np.average([predict_candidate(head, feature_sets[c["encoder"]][c["variant"]][1])
                                       for head, c in zip(restored, components, strict=True)], axis=0, weights=weights)
    np.testing.assert_allclose(restored_predictions, final_predictions, rtol=1e-6, atol=1e-6)
    report["status"] = "partial_encoder_failure" if report["encoder_errors"] else "complete"
    report["elapsed_seconds"] = time.perf_counter() - started
    save_json(output / "report.json", report)
    save_json(output / "diagnostics.json", {"status": report["status"], "environment": environment,
                                           "encoder_errors": report["encoder_errors"]})
    print("Packaging results, features and offline model weights...", flush=True)
    export_archives(output)
    print(f"Results: {output / 'postcode_results.zip'}", flush=True)
    print(f"Weights: {output / 'postcode_model.zip'}", flush=True)
    if report["encoder_errors"]:
        raise RuntimeError("One or more encoders failed. Send postcode_results.zip or kaggle_run.log for diagnosis.")


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)
    def flush(self):
        self.console.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--expected-train", type=int, default=716,
                        help="Expected labelled image count; override when retraining on confirmed additional labels")
    parser.add_argument("--output", type=Path, default=Path("outputs/first_run"))
    parser.add_argument("--encoders", nargs="+", choices=["dinov2_vitb14", "convnext_tiny"],
                        default=["dinov2_vitb14", "convnext_tiny"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds-csv", type=Path,
                        help="Reuse a validated exported folds.csv instead of regenerating folds from --seed")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--dino-image-size", type=int, choices=[392, 518], default=392)
    parser.add_argument("--search-profile", choices=['original', 'refined'], default='original')
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if len(set(args.encoders)) != len(args.encoders) or args.batch_size < 1:
        parser.error("Encoder names must be unique and batch size must be positive")
    if args.expected_train < 1:
        parser.error("--expected-train must be a positive integer")
    args.output.mkdir(parents=True, exist_ok=True)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with (args.output / "run.log").open("a", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(old_stdout, log), Tee(old_stderr, log)
        try:
            run(args)
        except Exception:
            trace = traceback.format_exc()
            print(trace, file=sys.stderr)
            save_json(args.output / "failure.json", {"error": trace})
            report_path = args.output / "report.json"
            if report_path.exists():
                failed_report = json.loads(report_path.read_text(encoding="utf-8"))
                failed_report["status"] = "failed"
                failed_report["failure"] = trace
                save_json(report_path, failed_report)
            export_archives(args.output)
            raise
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
