"""Small, reproducible SVR search on verified Kaggle embeddings; no GPU required."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from postcode_ml.data import load_dataset, load_saved_folds, make_folds
from postcode_ml.evaluation import evaluate_candidates, fit_candidate, predict_candidate, metrics, write_submission
from postcode_ml.experiment import dataset_manifest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('../postcode_results.zip'))
    parser.add_argument('--bundle', type=Path, default=Path('runtime/model_bundle'))
    parser.add_argument('--data-root', type=Path, default=Path('..'))
    parser.add_argument('--output', type=Path, default=Path('outputs/refine_cached'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve prior experiments')
    data = load_dataset(args.data_root)
    args.output.mkdir(parents=True)
    with zipfile.ZipFile(args.results) as archive:
        original_manifest = json.loads(archive.read('dataset_manifest.json'))
        actual_manifest = dataset_manifest(data)
        if actual_manifest != original_manifest:
            raise ValueError('Cached features do not match the current images/tables/order')
        folds_path = args.output / 'folds.csv'
        folds_path.write_bytes(archive.read('folds.csv'))
        folds = load_saved_folds(folds_path, data)
        with np.load(io.BytesIO(archive.read('features/dinov2_vitb14.npz')), allow_pickle=False) as saved:
            features = {key: saved[key].copy() for key in ('global', 'spatial')}
        old_rows = list(csv.DictReader(io.StringIO(archive.read('oof_predictions.csv').decode('utf-8'))))
        old_map = {row['image_id']: float(row['selected']) for row in old_rows}
        old_oof = np.array([old_map[key] for key in data.train_ids])
        encoder_meta = json.loads(archive.read('features/dinov2_vitb14.json'))
    local_meta = json.loads((args.bundle / 'encoders/dinov2_vitb14/encoder.json').read_text(encoding='utf-8'))
    if encoder_meta['weights_sha256'] != local_meta['weights_sha256']:
        raise ValueError('Offline encoder does not match cached features')
    n = len(data.y)
    if any(v.shape[0] != n + len(data.test_ids) or not np.isfinite(v).all() for v in features.values()):
        raise ValueError('Invalid cached feature matrix')
    baseline_spec = {'name': 'original', 'kind': 'svr', 'C': 100., 'epsilon': 2., 'gamma': 'scale'}
    # Predeclare a bounded search; all decisions use labelled train data only.
    candidates = []
    for variant, matrix in features.items():
        dimension = matrix.shape[1]
        specs = [baseline_spec] + [
            {'name': f'c{c}_g{g}_e1', 'kind': 'svr', 'C': float(c), 'epsilon': 1., 'gamma': g/dimension}
            for c in (100, 300, 1000) for g in (0.25, 1., 4.)
        ]
        for spec in specs:
            result = evaluate_candidates(matrix[:n], data.y, folds, [spec])[0]
            result.update(variant=variant, key=f'{variant}__{spec["name"]}')
            candidates.append(result)
            print(result['key'], round(result['metrics']['mae'], 5), flush=True)
            save_json(args.output / 'progress.json', [{k:v for k,v in r.items() if k!='oof'} for r in candidates])
    original = next(r for r in candidates if r['key']=='spatial__original')
    np.testing.assert_allclose(original['oof'], old_oof, atol=1e-6, rtol=1e-7)
    candidates.sort(key=lambda r: r['metrics']['mae'])
    best = candidates[0]
    # A different grouped partition is a stability check, not an untouched holdout.
    second_folds = make_folds(data.y, data.groups, seed=2026)
    second = {}
    for candidate in (original, best):
        r = evaluate_candidates(features[candidate['variant']][:n], data.y, second_folds, [candidate['spec']])[0]
        second[candidate['key']] = {k:v for k,v in r.items() if k!='oof'}
        print('Stability', candidate['key'], r['metrics']['mae'], flush=True)
    wins = sum(b['mae'] < a['mae'] for a,b in zip(original['per_fold'], best['per_fold'], strict=True))
    accepted = (best['metrics']['mae'] < original['metrics']['mae'] - .15 and wins >= 3
                and second[best['key']]['metrics']['mae'] < second[original['key']]['metrics']['mae'] - .05)
    report = {'original_public_mae_user_reported': 9.4549, 'original_cv': original['metrics'],
              'best': {k:v for k,v in best.items() if k!='oof'}, 'fold_wins': wins,
              'stability_seed': 2026, 'stability_results': second, 'recommended': accepted,
              'note': 'Both CV partitions reuse the train labels. This is model selection, not an unbiased holdout or guarantee of public/private improvement.',
              'candidates': [{k:v for k,v in r.items() if k!='oof'} for r in candidates],
              'source_results_sha256': hashlib.file_digest(args.results.open('rb'), 'sha256').hexdigest()}
    save_json(args.output / 'report.json', report)
    with (args.output/'oof.csv').open('w', encoding='utf-8', newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['image_id','load_pct','fold','original','candidate'])
        writer.writerows(zip(data.train_ids,data.y,folds,original['oof'],best['oof'],strict=True))
    if not accepted:
        print('No sufficiently stable improvement; no replacement submission published.', flush=True)
        return
    matrix = features[best['variant']]
    head = fit_candidate(best['spec'], matrix[:n], data.y)
    predictions = predict_candidate(head, matrix[n:])
    write_submission(args.output/'submission.csv', data, predictions)
    bundle = args.output/'model_bundle'
    bundle.mkdir()
    manifest = json.loads((args.bundle/'manifest.json').read_text(encoding='utf-8'))
    manifest.update(selection=best['key'], weights=[1.], components=[{
        'key': best['key'], 'encoder':'dinov2_vitb14', 'variant':best['variant'], 'spec':best['spec']}])
    manifest['refinement'] = {'original_cv': original['metrics'], 'selected_cv':best['metrics'],
                              'source_results_sha256':report['source_results_sha256']}
    save_json(bundle/'manifest.json', manifest)
    with (bundle/'heads.pkl').open('wb') as stream:
        pickle.dump([head],stream,protocol=4)
    shutil.copy2(args.bundle/'requirements-lock.txt',bundle/'requirements-lock.txt')
    shutil.copytree(args.bundle/'encoders/dinov2_vitb14',bundle/'encoders/dinov2_vitb14',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    with (bundle/'heads.pkl').open('rb') as stream:
        restored=pickle.load(stream)
    np.testing.assert_allclose(predict_candidate(restored[0],matrix[n:]),predictions,atol=1e-9)
    print('Saved compatible model bundle and validated submission:',args.output,flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
