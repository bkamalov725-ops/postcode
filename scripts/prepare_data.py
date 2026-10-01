"""Validate and safely import the canonical or original competition ZIP.

No existing file is overwritten. The complete archive is verified in a temporary
directory before publishing any files, so damaged ZIPs cannot leave usable-looking
partial datasets. Repeating an import of identical files is safe.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from uuid import uuid4
import zipfile


TABLES = {
    "train/train.csv": ["image_id", "load_pct"],
    "train/train_groups.csv": ["image_id", "group_id"],
    "test/test.csv": ["image_id"],
    "test/sample_submission.csv": ["image_id", "load_pct"],
}
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}


def _safe_parts(name: str) -> tuple[str, ...]:
    """Reject paths that Windows or POSIX could resolve differently."""
    parts = tuple(name.rstrip("/").split("/"))
    if not parts or any(
        not p or p in {".", ".."} or p != p.strip() or p.endswith(".")
        or re.search(r'[<>:"\\|?*\x00-\x1f]', p)
        or p.split(".", 1)[0].upper() in _RESERVED for p in parts
    ):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return parts


def _validated_members(archive: zipfile.ZipFile, *, max_total: int = 2 * 1024**3):
    """Validate every ZIP member, including entries that will not be imported."""
    members, names, total = [], set(), 0
    for entry in archive.infolist():
        if entry.orig_filename != entry.filename:
            raise ValueError("Archive path contains a NUL byte")
        parts = _safe_parts(entry.filename)
        key = "/".join(parts).casefold()
        if key in names:
            raise ValueError(f"Duplicate archive path: {entry.filename}")
        names.add(key)
        mode = stat.S_IFMT(entry.external_attr >> 16)
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR) or (mode == stat.S_IFDIR and not entry.is_dir()):
            raise ValueError(f"Symlink or special archive member: {entry.filename}")
        if entry.flag_bits & 1:
            raise ValueError("Encrypted archives are not supported")
        if entry.file_size < 0 or entry.compress_size < 0:
            raise ValueError("Invalid archive member size")
        total += entry.file_size
        if total > max_total:
            raise ValueError("Archive exceeds the uncompressed size limit")
        if not entry.is_dir():
            members.append(entry)
    return members


def _destination(root: Path, relative: str) -> Path:
    path = root.joinpath(*_safe_parts(relative))
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Output path escapes its directory: {relative}")
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f"Output path is a symbolic link: {cursor}")
        cursor = cursor.parent
    return path


def _extract(archive: zipfile.ZipFile, selected, stage: Path, *, max_file: int):
    for entry, relative in selected:
        if entry.file_size > max_file:
            raise ValueError(f"Archive member exceeds the size limit: {entry.filename}")
        destination = _destination(stage, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        copied = 0
        with archive.open(entry) as source, destination.open("xb") as target:
            while block := source.read(1024 * 1024):
                copied += len(block)
                if copied > entry.file_size or copied > max_file:
                    raise ValueError(f"Archive member exceeds declared size: {entry.filename}")
                target.write(block)
        if copied != entry.file_size:
            raise ValueError(f"Archive member has an invalid size: {entry.filename}")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _publish(stage: Path, output: Path, relatives: list[str]) -> int:
    """Preflight all collisions, then publish complete files without replacement."""
    pending = []
    for relative in relatives:
        source, destination = _destination(stage, relative), _destination(output, relative)
        if destination.exists():
            if (not destination.is_file() or source.stat().st_size != destination.stat().st_size
                    or _sha256(source) != _sha256(destination)):
                raise FileExistsError(f"Different file already exists: {destination}; choose a fresh output directory")
        else:
            for parent in destination.parents:
                if parent == output.parent:
                    break
                if parent.exists() and not parent.is_dir():
                    raise FileExistsError(f"Output directory is occupied by a file: {parent}")
            pending.append((source, destination))
    for source, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A hardlink from the private staging directory also retains its private
        # Windows ACL. Create a fresh sibling with the destination directory's
        # permissions, then atomically publish it without replacing other files.
        sibling = destination.with_name(f'.{destination.name}.{uuid4().hex}.tmp')
        try:
            with source.open('rb') as src, sibling.open('xb') as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            os.link(sibling, destination)
        finally:
            sibling.unlink(missing_ok=True)
    return len(pending)


def _cleanup_stage(stage: Path, parent: Path, prefix: str) -> None:
    """Validate the final absolute target before recursive removal on Windows."""
    if (stage.is_symlink() or stage.resolve().parent != parent.resolve()
            or not stage.name.startswith(prefix)):
        raise RuntimeError(f"Refusing to clean an unexpected staging directory: {stage}")
    shutil.rmtree(stage)


def _rows(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"{path.name}: expected exactly columns {fields}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: table is empty")
    ids = set()
    for row in rows:
        if set(row) != set(fields) or any(v is None or not v.strip() or v != v.strip() for v in row.values()):
            raise ValueError(f"{path.name}: missing, extra or whitespace-padded CSV values")
        image_id = row["image_id"]
        if len(_safe_parts(image_id)) != 1:
            raise ValueError("image_id must be a filename stem")
        if image_id.casefold() in ids:
            raise ValueError(f"{path.name}: duplicate image ID {image_id}")
        ids.add(image_id.casefold())
        if "load_pct" in fields:
            value = float(row["load_pct"])
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"{path.name}: load_pct must be finite in [0, 100]")
    return rows


def _validate_dataset(stage: Path, relatives: list[str], expected_train: int, expected_test: int):
    missing = set(TABLES) - set(relatives)
    if missing:
        raise ValueError(f"Missing required tables: {sorted(missing)}")
    tables = {relative: _rows(stage / relative, fields) for relative, fields in TABLES.items()}
    train = {row["image_id"] for row in tables["train/train.csv"]}
    test = {row["image_id"] for row in tables["test/test.csv"]}
    if len(train) != expected_train or len(test) != expected_test:
        raise ValueError(f"Expected {expected_train} train and {expected_test} test IDs; got {len(train)} and {len(test)}")
    if {i.casefold() for i in train} & {i.casefold() for i in test}:
        raise ValueError("Train and test image IDs overlap")
    if train != {row["image_id"] for row in tables["train/train_groups.csv"]}:
        raise ValueError("Groups must cover exactly the train IDs")
    if test != {row["image_id"] for row in tables["test/sample_submission.csv"]}:
        raise ValueError("Sample submission IDs must match test.csv exactly")
    expected = set(TABLES) | {f"train/images/{i}.jpg" for i in train} | {f"test/images/{i}.jpg" for i in test}
    if set(relatives) != expected:
        raise ValueError(f"Image/table ID mismatch: missing={len(expected - set(relatives))}, extra={len(set(relatives) - expected)}")
    from PIL import Image
    for relative in sorted(expected - set(TABLES)):
        try:
            with Image.open(stage / relative) as image:
                image.verify()
            with Image.open(stage / relative) as image:
                image.load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Cannot decode image: {relative}") from exc


def prepare_data(archive_path: str | Path, output: str | Path, *, expected_train: int = 716,
                 expected_test: int = 307) -> int:
    if expected_train <= 0 or expected_test <= 0:
        raise ValueError("Expected dataset counts must be positive")
    output = Path(output).absolute()
    if output.is_symlink():
        raise ValueError("Output directory must not be a symbolic link")
    output = output.resolve()
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"Output is not a directory: {output}")
    selected, keys, layouts = [], set(), set()
    with zipfile.ZipFile(archive_path) as archive:
        for entry in _validated_members(archive):
            name, layout, relative = entry.filename, None, None
            if name.startswith(("train/", "test/")):
                layout, relative = "canonical", name
            elif name.startswith("postcode/train/"):
                layout, relative = "legacy", "train/" + name[len("postcode/train/"):]
            elif name.startswith("postcode/test/test/"):
                layout, relative = "legacy", "test/" + name[len("postcode/test/test/"):]
            if relative is None:
                continue
            parts = relative.split("/")
            if relative not in TABLES and not (len(parts) == 3 and parts[1] == "images" and parts[2].endswith(".jpg")):
                continue
            layouts.add(layout)
            if relative.casefold() in keys:
                raise ValueError(f"Duplicate dataset output path: {relative}")
            keys.add(relative.casefold())
            selected.append((entry, relative))
        if len(layouts) != 1:
            raise ValueError("Expected exactly one canonical or legacy dataset layout")
        output.parent.mkdir(parents=True, exist_ok=True)
        prefix = ".prepare-data-"
        stage = Path(tempfile.mkdtemp(prefix=prefix, dir=output.parent)).resolve()
        try:
            _extract(archive, selected, stage, max_file=64 * 1024**2)
            relatives = [relative for _, relative in selected]
            _validate_dataset(stage, relatives, expected_train, expected_test)
            return _publish(stage, output, relatives)
        finally:
            _cleanup_stage(stage, output.parent, prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--expected-train", type=int, default=716)
    parser.add_argument("--expected-test", type=int, default=307)
    args = parser.parse_args()
    count = prepare_data(args.archive, args.output, expected_train=args.expected_train, expected_test=args.expected_test)
    print(f"Dataset verified; imported {count} new files to {args.output}")


if __name__ == "__main__":
    main()
