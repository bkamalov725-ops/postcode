"""Retrain Attention518 on confirmed labels with five group-held-out folds.

Example: python scripts/retrain_attention518.py --train-csv data/train.csv
--groups-csv data/train_groups.csv --images-dir data/images --output outputs/retrained
--device cuda:0. Add --plan-only to validate data and save the plan without training.
Existing deployed weights are never modified. A fresh pretrained encoder is used in
each fold; task-trained weights do not cross validation boundaries.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib,importlib.util,json,re,shutil,sys
import numpy as np

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path,fields):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=fields:raise ValueError(f'{path}: expected columns {fields}')
        return list(reader)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-dir',type=Path,default=Path('runtime/attention518'))
    p.add_argument('--train-csv',type=Path,required=True)
    p.add_argument('--groups-csv',type=Path,required=True)
    p.add_argument('--images-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();root=a.bundle_dir.resolve();out=a.output.resolve()
    if a.epochs<3:raise ValueError('Use at least 3 epochs for final-three checkpoint averaging')
    if out==root or out.is_relative_to(root) or root.is_relative_to(out):raise ValueError('Output must be separate from the deployed bundle')
    rows=read(a.train_csv,['image_id','load_pct']);gr=read(a.groups_csv,['image_id','group_id'])
    ids=[r['image_id'] for r in rows];mapping={r['image_id']:r['group_id'] for r in gr}
    if not ids or len(set(ids))!=len(ids) or len(mapping)!=len(gr) or set(mapping)!=set(ids):raise ValueError('IDs must be unique and match groups exactly')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',i) for i in ids):raise ValueError('Invalid image IDs')
    y=np.array([float(r['load_pct']) for r in rows]);groups=np.array([mapping[i] for i in ids])
    paths=[(a.images_dir/(i+'.jpg')).resolve() for i in ids]
    name='_postcode_training_source'
    spec=importlib.util.spec_from_file_location(name,root/'postcode_ml/__init__.py',submodule_search_locations=[str(root/'postcode_ml')])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    data=importlib.import_module(name+'.data');network=importlib.import_module(name+'.mae_spatial')
    folds=data.make_folds(y,groups,n_splits=5,seed=a.seed)
    from PIL import Image
    hashes={}
    for image_id,path in zip(ids,paths):
        with Image.open(path) as image:image.verify()
        hashes[image_id]=sha(path)
    fingerprint=hashlib.sha256(json.dumps({'images':hashes,'labels':y.tolist(),'groups':groups.tolist()},sort_keys=True).encode()).hexdigest()
    source_hash=hashlib.sha256(b''.join(f.name.encode()+f.read_bytes() for f in sorted((root/'postcode_ml').glob('*.py')))).hexdigest()
    plan={'config':{'variant':'attention_mae','image_size':518,'train_blocks':2,'epochs':a.epochs,'microbatch':4,'effective_batch':8},'seed':a.seed,'dataset_sha256':fingerprint,'source_sha256':source_hash,'pretrained_sha256':sha(root/'highres/encoder/model.pt'),'count':len(ids),'groups':len(set(groups)),'folds':folds.tolist(),'labels_confirmed_by':'dataset provider','epoch_selection':'fixed budget; mean of final three snapshots','initialization':'pretrained DINO only in each fold'}
    out.mkdir(parents=True,exist_ok=True)
    plan_path=out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text('utf-8'))!=plan:raise ValueError('Different training plan: choose a new output directory')
    plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    if a.plan_only:
        print('Validated data and grouped training plan:',plan_path);return
    oof=np.full(len(ids),np.nan)
    for f in range(5):
        train=np.where(folds!=f)[0];valid=np.where(folds==f)[0]
        assert not set(groups[train]) & set(groups[valid])
        payload=dict(variant='attention_mae',tag=f'retrain_primary_{f}',encoder='dinov2_vitb14',output=str(out/'primary'/f'fold{f}'),encoder_dir=str(root/'highres/encoder'),device=a.device,paths=list(map(str,paths)),labels=y.tolist(),test_paths=[str(paths[valid[0]])],train_indices=train.tolist(),valid_indices=valid.tolist(),seed=20260918+f,epochs=a.epochs,image_size=518,batch_size=4,effective_batch_size=8,train_blocks=2,workers=0,threads=2,code_hash=source_hash,data_hash=fingerprint,pretrained_sha256=plan['pretrained_sha256'])
        network.train_fold(payload)
        with np.load(out/'primary'/f'fold{f}'/'predictions.npz',allow_pickle=False) as result:oof[valid]=result['valid']
    report={'mae':float(abs(y-oof).mean()),'within_10_pct':float((abs(y-oof)<=10).mean()*100),'fold_mae':[float(abs(y[folds==f]-oof[folds==f]).mean()) for f in range(5)],'public_mae':None}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(out/'oof.npz',image_id=np.array(ids),y=y,oof=oof,folds=folds)
    shutil.copytree(root/'highres',out/'highres',dirs_exist_ok=True)
    shutil.copytree(root/'postcode_ml',out/'postcode_ml',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    manifest={'architecture':'attention518_original_v1','encoder_weights':{'sha256':plan['pretrained_sha256']},'trained_on_new_data':True}
    (out/'MODEL_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('New bundle saved separately:',out)

if __name__=='__main__':main()
