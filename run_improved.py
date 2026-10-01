"""Compare two DINO resolutions on the same grouped folds, one run per GPU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent


def choose_winner(runs):
    candidates = []
    for folder in runs:
        folder = Path(folder)
        report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
        if report['status'] != 'complete':
            raise RuntimeError(f'Incomplete run: {folder}')
        candidates.append({'directory': folder.name, 'selection': report['selection'],
                           'folds_sha256': report['folds']['exported_sha256']})
    if len({r['folds_sha256'] for r in candidates}) != 1:
        raise ValueError('Cannot compare different validation folds')
    if any(not 0 <= r['selection']['metrics']['mae'] <= 100 for r in candidates):
        raise ValueError('Invalid validation MAE')
    return min(candidates, key=lambda r:r['selection']['metrics']['mae']), candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/postcode_output'))
    parser.add_argument('--folds-csv', type=Path, default=ROOT/'validation/folds_kaggle.csv')
    args = parser.parse_args()
    import torch
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError('Enable Kaggle GPU T4; this runner requires at least one GPU')
    args.output.mkdir(parents=True, exist_ok=True)
    pending = [392, 518]
    active, completed = {}, []
    try:
        while pending or active:
            free = [i for i in range(min(gpu_count, 2)) if i not in active]
            for gpu in free:
                if not pending:
                    break
                size = pending.pop(0)
                folder = args.output/'runs'/f'dino_{size}'
                folder.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                visible = env.get('CUDA_VISIBLE_DEVICES')
                devices = visible.split(',') if visible else list(map(str,range(gpu_count)))
                env.update(CUDA_VISIBLE_DEVICES=devices[gpu], OMP_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                           TORCH_HOME=str(args.output.parent.resolve()/'postcode_torch_cache'/f'dino_{size}'))
                command = [sys.executable, '-u', str(ROOT/'run_experiment.py'),
                           '--data-root', str(args.data_root.resolve()), '--output', str(folder.resolve()),
                           '--folds-csv', str(args.folds_csv.resolve()), '--encoders', 'dinov2_vitb14',
                           '--dino-image-size', str(size), '--search-profile', 'refined', '--batch-size', '12']
                log_path = folder/'launcher.log'
                log = log_path.open('w', encoding='utf-8')
                try:
                    process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                active[gpu] = (process, log, log_path, folder, 0)
                print(f'Started DINO {size}px on GPU {gpu}', flush=True)
            for gpu, (process, log, log_path, folder, offset) in list(active.items()):
                with log_path.open(encoding='utf-8', errors='replace') as stream:
                    stream.seek(offset)
                    new = stream.read()
                    offset = stream.tell()
                if new:
                    print(f'[{folder.name}]\n{new}', end='', flush=True)
                active[gpu] = (process, log, log_path, folder, offset)
                status = process.poll()
                if status is not None:
                    log.close()
                    if status != 0:
                        raise RuntimeError(f'{folder.name} failed with exit code {status}; inspect runs/{folder.name}/launcher.log')
                    completed.append(folder)
                    del active[gpu]
            if active:
                time.sleep(1)
    finally:
        for process, log, *_ in active.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            log.close()
    winner, results = choose_winner(completed)
    comparison = {'winner': winner, 'runs': results,
                  'note': 'Selected by the same grouped training CV, not public test labels. No guarantee of public/private improvement.'}
    content = json.dumps(comparison, ensure_ascii=False, indent=2)
    (args.output/'comparison.json').write_text(content, encoding='utf-8')
    selected = args.output/'runs'/winner['directory']
    for name in ('submission.csv','report.json','postcode_results.zip','postcode_model.zip'):
        shutil.copy2(selected/name, args.output/name)
    with zipfile.ZipFile(args.output/'postcode_results.zip','a',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('resolution_comparison.json',content)
        for folder in completed:
            archive.write(folder/'report.json',f'comparison/{folder.name}/report.json')
            archive.write(folder/'oof_predictions.csv',f'comparison/{folder.name}/oof_predictions.csv')
    print('SELECTED:', winner['directory'], winner['selection']['metrics'], flush=True)
    print('Download submission.csv, postcode_results.zip, postcode_model.zip, kaggle_run.log', flush=True)


if __name__ == '__main__':
    main()
