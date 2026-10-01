"""Import the selected offline model from a trusted Kaggle output ZIP.

Extraction never loads pickle or executes model code. Subsequent inference does,
so --trust-model-source confirms that this is your own training output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import zipfile

if __package__:
    from .prepare_data import _cleanup_stage, _extract, _publish, _safe_parts, _sha256, _validated_members
else:
    from prepare_data import _cleanup_stage, _extract, _publish, _safe_parts, _sha256, _validated_members


def _json_member(archive, entry):
    if entry.file_size > 4 * 1024**2:
        raise ValueError(f"JSON metadata exceeds size limit: {entry.filename}")
    return json.loads(archive.read(entry).decode("utf-8"))


def _encoder_names(manifest) -> set[str]:
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("Unsupported model manifest format")
    components, weights = manifest.get("components"), manifest.get("weights")
    if not isinstance(components, list) or not components or not isinstance(weights, list) or len(components) != len(weights):
        raise ValueError("Model components and weights must be nonempty and aligned")
    if any(not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Model weights must be finite, nonnegative and have a positive sum")
    names = set()
    for component in components:
        name = component.get("encoder") if isinstance(component, dict) else None
        if not isinstance(name, str) or len(_safe_parts(name)) != 1 or "/" in name:
            raise ValueError("Invalid encoder name in model manifest")
        if name != "baseline":
            names.add(name)
    return names


def prepare_model(archive_path: str | Path, output: str | Path, *, trust_model_source: bool = False) -> int:
    if not trust_model_source:
        raise ValueError("Model bundles contain executable pickle/Python code; use --trust-model-source only for your own trusted training output")
    output = Path(output).absolute()
    if output.is_symlink():
        raise ValueError("Output directory must not be a symbolic link")
    output = output.resolve()
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"Output is not a directory: {output}")
    with zipfile.ZipFile(archive_path) as archive:
        members = {entry.filename: entry for entry in _validated_members(archive, max_total=4 * 1024**3)}
        prefixes = [prefix for prefix in ("model_bundle/", "") if prefix + "manifest.json" in members]
        if len(prefixes) != 1:
            raise ValueError("Expected exactly one model manifest at root or model_bundle/")
        prefix = prefixes[0]
        manifest = _json_member(archive, members[prefix + "manifest.json"])
        encoders = _encoder_names(manifest)
        required = {"manifest.json", "heads.pkl", "requirements-lock.txt"}
        metadata = {}
        for name in sorted(encoders):
            meta_path = f"encoders/{name}/encoder.json"
            if prefix + meta_path not in members:
                raise ValueError(f"Missing offline encoder metadata: {name}")
            meta = _json_member(archive, members[prefix + meta_path])
            if not isinstance(meta, dict) or meta.get("name") != name:
                raise ValueError(f"Encoder metadata mismatch: {name}")
            digest = meta.get("weights_sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Missing or invalid encoder checksum: {name}")
            required |= {meta_path, f"encoders/{name}/model.pt"}
            source_files = meta.get("source_files", {})
            if source_files is None:
                source_files = {}
            if not isinstance(source_files, dict):
                raise ValueError(f"Invalid encoder source checksums: {name}")
            if name.startswith("dinov2") and not source_files:
                raise ValueError("DINOv2 offline source files are missing")
            meta["source_files"] = source_files
            for relative, checksum in source_files.items():
                _safe_parts(relative)
                if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                    raise ValueError("Invalid encoder source checksum")
                required.add(f"encoders/{name}/source/{relative}")
            metadata[name] = meta
        missing = {relative for relative in required if prefix + relative not in members}
        if missing:
            raise ValueError(f"Incomplete model bundle: {sorted(missing)[:10]}")
        selected = [(members[prefix + relative], relative) for relative in sorted(required)]
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_prefix = ".prepare-model-"
        stage = Path(tempfile.mkdtemp(prefix=temp_prefix, dir=output.parent)).resolve()
        try:
            _extract(archive, selected, stage, max_file=2 * 1024**3)
            for name, meta in metadata.items():
                encoder = stage / "encoders" / name
                if _sha256(encoder / "model.pt") != meta["weights_sha256"]:
                    raise ValueError(f"Encoder weights checksum mismatch: {name}")
                for relative, checksum in meta.get("source_files", {}).items():
                    if _sha256(encoder / "source" / relative) != checksum:
                        raise ValueError(f"Encoder source checksum mismatch: {name}/{relative}")
            return _publish(stage, output, sorted(required))
        finally:
            _cleanup_stage(stage, output.parent, temp_prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runtime/model_bundle"))
    parser.add_argument("--trust-model-source", action="store_true",
                        help="Confirm this ZIP came from your own trusted training run")
    args = parser.parse_args()
    count = prepare_model(args.archive, args.output, trust_model_source=args.trust_model_source)
    print(f"Selected offline model verified; imported {count} new files to {args.output}")


if __name__ == "__main__":
    main()
