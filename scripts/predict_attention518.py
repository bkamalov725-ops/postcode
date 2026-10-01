"""Reproduce the deployed model on an image-ID CSV, without cached predictions."""
from pathlib import Path
import argparse,csv,re,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from postcode_ml.best_model import Predictor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-dir',type=Path,default=Path('runtime/attention518'))
    p.add_argument('--images-dir',type=Path,required=True)
    p.add_argument('--ids-csv',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(4)
    with a.ids_csv.open(encoding='utf-8-sig',newline='') as f:ids=[r['image_id'] for r in csv.DictReader(f)]
    if not ids or len(ids)!=len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+',i) for i in ids):raise ValueError('Invalid IDs')
    if a.output.exists():raise FileExistsError('Choose a new output file')
    values=Predictor(a.bundle_dir,a.device).predict([a.images_dir/(i+'.jpg') for i in ids])
    if len(values)!=len(ids) or not np.isfinite(values).all() or np.any((values<0)|(values>100)):raise ValueError('Invalid predictions')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['image_id','load_pct']);w.writerows((i,f'{v:.6f}') for i,v in zip(ids,values))
    print('Saved',a.output,'rows',len(ids))

if __name__=='__main__':main()
