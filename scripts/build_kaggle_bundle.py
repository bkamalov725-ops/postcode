"""Build a private Kaggle dataset and a self-contained launch notebook (stdlib only)."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import textwrap
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Explicitly allow this validation artifact; never sweep arbitrary CSV/data into
# the embedded code archive.
BUNDLE_FILES = ("run_experiment.py", "run_improved.py", "predict_bundle.py", "requirements.txt",
                "validation/folds_kaggle.csv")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_rows(path: Path, columns: list[str], expected: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows in {path}, found {len(rows)}")
    ids = [row["image_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate image_id in {path}")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) for identifier in ids):
        raise ValueError(f"Invalid image_id in {path}")
    return rows


def dataset_files(data_root: Path) -> list[tuple[Path, str]]:
    """Use the original full test folder, never baseline/test."""
    train, test = data_root / "train", data_root / "test" / "test"
    train_rows = read_rows(train / "train.csv", ["image_id", "load_pct"], 716)
    groups = read_rows(train / "train_groups.csv", ["image_id", "group_id"], 716)
    test_rows = read_rows(test / "test.csv", ["image_id"], 307)
    sample = read_rows(test / "sample_submission.csv", ["image_id", "load_pct"], 307)
    if {row["image_id"] for row in groups} != {row["image_id"] for row in train_rows}:
        raise ValueError("Training and grouping IDs differ")
    if {row["image_id"] for row in sample} != {row["image_id"] for row in test_rows}:
        raise ValueError("Test and sample submission IDs differ")
    files = [(train / "train.csv", "train/train.csv"),
             (train / "train_groups.csv", "train/train_groups.csv"),
             (test / "test.csv", "test/test.csv"),
             (test / "sample_submission.csv", "test/sample_submission.csv")]
    for folder, prefix, rows in [(train, "train", train_rows), (test, "test", test_rows)]:
        for row in rows:
            name = row["image_id"] + ".jpg"
            source = folder / "images" / name
            if not source.is_file():
                raise FileNotFoundError(source)
            files.append((source, f"{prefix}/images/{name}"))
    return sorted(files, key=lambda item: item[1])


def add_bytes(archive: zipfile.ZipFile, filename: str, data: bytes) -> None:
    # Fixed metadata makes repeated builds of unchanged inputs reproducible.
    info = zipfile.ZipInfo(filename, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def build_dataset(data_root: Path, output_dir: Path) -> dict:
    files = dataset_files(data_root)
    target = output_dir / "postcode_dataset.zip"
    temporary = target.with_suffix(".zip.tmp")
    manifest = {"train_images": 716, "test_images": 307, "files": {}}
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for index, (source, name) in enumerate(files, 1):
                data = source.read_bytes()
                manifest["files"][name] = {"sha256": sha256(data), "bytes": len(data)}
                add_bytes(archive, name, data)
                if index % 200 == 0:
                    print(f"Dataset: packed {index}/{len(files)} files", flush=True)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    manifest["archive"] = target.name
    with target.open("rb") as handle:
        manifest["archive_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset ready: {target} ({target.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    return manifest


def code_bundle() -> tuple[bytes, dict[str, str]]:
    sources = [REPO_ROOT / relative for relative in BUNDLE_FILES]
    package_sources = sorted((REPO_ROOT / "postcode_ml").rglob("*.py"))
    if not package_sources:
        raise FileNotFoundError("No Python sources found in postcode_ml")
    sources.extend(package_sources)
    hashes, stream = {}, io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            data = source.read_bytes()
            relative = source.relative_to(REPO_ROOT).as_posix()
            hashes[relative] = sha256(data)
            add_bytes(archive, relative, data)
    return stream.getvalue(), hashes


def cell(kind: str, source: str, cell_id: str) -> dict:
    result = {"cell_type": kind, "id": cell_id, "metadata": {},
              "source": textwrap.dedent(source).strip() + "\n"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook(output_dir: Path, improved: bool = False) -> dict:
    bundled, source_hashes = code_bundle()
    bundle_hash = sha256(bundled)
    setup = '''
        import base64, hashlib, importlib, io, json, os, sys, traceback, zipfile
        from pathlib import Path
        from IPython.display import FileLink, display

        CODE_DIR = Path("/kaggle/working/postcode-code")
        OUTPUT_DIR = Path("/kaggle/working/postcode_output")
        os.chdir("/kaggle/working")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RUN_LOG = OUTPUT_DIR / "kaggle_run.log"
        RUN_LOG.write_text("Postcode Kaggle environment check\\n", encoding="utf-8")

        def log(message):
            print(message, flush=True)
            with RUN_LOG.open("a", encoding="utf-8") as handle:
                handle.write(str(message) + "\\n")

        READY = False
        try:
            raw = base64.b64decode(BUNDLED_CODE)
            if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
                raise RuntimeError("Code archive checksum mismatch. Import the original notebook again.")
            CODE_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if set(archive.namelist()) != set(SOURCE_SHA256):
                    raise RuntimeError("Unexpected files in the embedded archive")
                for name, expected in SOURCE_SHA256.items():
                    target = (CODE_DIR / name).resolve()
                    if CODE_DIR.resolve() not in target.parents:
                        raise RuntimeError("Unsafe archive member")
                    data = archive.read(name)
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise RuntimeError("Source checksum mismatch: " + name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
            (OUTPUT_DIR / "bundled_sources.json").write_text(
                json.dumps({"bundle_sha256": BUNDLE_SHA256, "files": SOURCE_SHA256}, indent=2),
                encoding="utf-8")
            log("Embedded code verified: " + BUNDLE_SHA256)
            log("Python: " + sys.version.replace("\\n", " "))
            missing = []
            for name in ["numpy", "PIL", "sklearn", "torch", "torchvision"]:
                try:
                    module = importlib.import_module(name)
                    log(name + ": " + str(getattr(module, "__version__", "unknown")))
                except Exception as error:
                    missing.append(name)
                    log("IMPORT FAILED " + name + ": " + repr(error))
            if missing:
                raise RuntimeError("Missing or incompatible packages: " + ", ".join(missing)
                                   + ". Download kaggle_run.log and send it back. Do not reinstall torch.")
            import torch
            log("CUDA available: " + str(torch.cuda.is_available()))
            log("CUDA runtime: " + str(torch.version.cuda))
            if not torch.cuda.is_available():
                raise RuntimeError("Enable a GPU accelerator in Kaggle notebook settings, then Run All.")
            for index in range(torch.cuda.device_count()):
                gpu = torch.cuda.get_device_properties(index)
                log(f"GPU {index}: {gpu.name}, {gpu.total_memory / 1024**3:.1f} GiB")
            inputs = sorted(str(path) for path in Path("/kaggle/input").glob("*"))
            log("Attached inputs: " + repr(inputs))
            if not inputs:
                raise RuntimeError("Attach the private postcode_dataset dataset with Add Input, then Run All.")
            log("Pretrained weights require Internet enabled in notebook settings on the first run.")
            READY = True
        except Exception:
            log(traceback.format_exc())
            log("Setup failed. Download the diagnostic log below or from Output/postcode_output.")
            display(FileLink(str(RUN_LOG.relative_to(Path.cwd()))))
            raise
    '''
    setup_source = (
        "BUNDLED_CODE = " + repr(base64.b64encode(bundled).decode("ascii")) + "\n"
        + "BUNDLE_SHA256 = " + repr(bundle_hash) + "\n"
        + "SOURCE_SHA256 = " + repr(source_hashes) + "\n\n"
        + textwrap.dedent(setup).strip() + "\n")
    cells = [
        cell("markdown", '''
            # Postcode: эксперимент на GPU

            Оценка загрузки кузова по фотографии. Код уже встроен в этот ноутбук;
            подключение GitLab и токен доступа не нужны.

            Добавьте **приватный** датасет из `postcode_dataset.zip`, выберите
            **GPU T4 ×2**, если доступно, и включите **Internet** для весов.
            Затем нажмите **Run All**. Для сохранения результатов используйте
            **Save Version → Save & Run All** и дождитесь завершения версии.

            Модели сравниваются на сохранённом разбиении первого успешного запуска:
            `validation/folds_kaggle.csv` уже встроен в ноутбук. Группы похожих
            фотографий остаются в одной части, независимо от версии scikit-learn.
            Значения из `sample_submission.csv` не используются как разметка.
            Метрики нового эксперимента появятся после выполнения.
        ''', "intro"),
        cell("code", setup_source, "setup"),
        cell("markdown", '''
            ## Запуск

            Статус появится ниже по мере выполнения. Первый запуск скачивает веса.
            При ошибке скачайте `kaggle_run.log` из упавшей ячейки или вкладки Output.
        ''', "run-help"),
        cell("code", '''
            import subprocess

            if READY:
                command = [sys.executable, "-u", "run_experiment.py",
                           "--data-root", "/kaggle/input", "--output", str(OUTPUT_DIR),
                           "--folds-csv", str(CODE_DIR / "validation" / "folds_kaggle.csv"),
                           "--encoders", "dinov2_vitb14", "convnext_tiny"]
                log("Command: " + " ".join(command))
                try:
                    environment = os.environ.copy()
                    environment["PYTHONUNBUFFERED"] = "1"
                    with RUN_LOG.open("a", encoding="utf-8") as handle:
                        process = subprocess.Popen(
                            command, cwd=CODE_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1, env=environment)
                        try:
                            for line in process.stdout:
                                print(line, end="", flush=True)
                                handle.write(line)
                                handle.flush()
                            return_code = process.wait()
                        except BaseException:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            raise
                    log("Process exit code: " + str(return_code))
                    if return_code:
                        raise RuntimeError("Experiment failed with exit code " + str(return_code)
                                           + ". Download kaggle_run.log below and send it back.")
                    else:
                        log("Run completed. Download the result and model archives below.")
                except Exception:
                    log(traceback.format_exc())
                    display(FileLink(str(RUN_LOG.relative_to(Path.cwd()))))
                    raise
            else:
                log("Run skipped because setup failed. Download the diagnostic log below.")
        ''', "run"),
        cell("markdown", '''
            ## Скачать результат

            Передайте **postcode_results.zip** и **kaggle_run.log** для анализа.
            Сохраните **postcode_model.zip** для повторного прогнозирования.
            **submission.csv** — файл для платформы хакатона; сначала проверим метрики.

            В сохранённой версии можно скачать файлы через вкладку Output,
            папка `postcode_output`. Не публикуйте материалы соревнования.
        ''', "download-help"),
        cell("code", '''
            for filename in ["postcode_results.zip", "postcode_model.zip", "submission.csv", "kaggle_run.log"]:
                path = OUTPUT_DIR / filename
                if path.is_file():
                    print(f"{filename}: {path.stat().st_size / 1024**2:.2f} MiB")
                    display(FileLink(str(path.relative_to(Path.cwd()))))
                else:
                    print("Not produced: " + filename)
        ''', "download"),
    ]
    if improved:
        for item in cells:
            item['source'] = item['source'].replace('postcode_output', 'postcode_output_v2')
            if item['id'] == 'run':
                item['source'] = item['source'].replace('"run_experiment.py"', '"run_improved.py"')
                item['source'] = re.sub(r',\s*"--encoders", "dinov2_vitb14", "convnext_tiny"', '', item['source'])
            if item['id'] == 'intro':
                item['source'] += ('\n## Версия 2\n'
                    'Сравниваются DINOv2 при 392 и 518 пикселях, по одному запуску на GPU. '
                    'При одной GPU запуски последовательные. В каждом проверяются улучшенные '
                    'настройки SVR; лучший вариант выбирается по одной и той же групповой валидации. '
                    'Датасет прежний. Веса нейросети фиксированы; регрессор обучается заново.\n')
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "postcode": {"bundle_sha256": bundle_hash, "source_sha256": source_hashes}},
        "nbformat": 4, "nbformat_minor": 5}
    content = json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"
    filename = 'postcode_kaggle_v2.ipynb' if improved else 'postcode_kaggle.ipynb'
    for target in [output_dir / filename, REPO_ROOT / "notebooks" / filename]:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"Notebook ready: {target}", flush=True)
    manifest = {"bundle_sha256": bundle_hash, "source_sha256": source_hashes}
    (output_dir / ('code_manifest_v2.json' if improved else 'code_manifest.json')).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT.parent / "kaggle_launch")
    parser.add_argument("--notebook-only", action="store_true", help="Refresh code without rebuilding data ZIP")
    parser.add_argument("--improved", action="store_true", help="Build v2: compare two DINO resolutions with refined heads")
    args = parser.parse_args()
    code_bundle()  # Check code before creating the large data archive.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.notebook_only:
        build_dataset(args.data_root.resolve(), args.output_dir.resolve())
    build_notebook(args.output_dir.resolve(), improved=args.improved)


if __name__ == "__main__":
    main()
