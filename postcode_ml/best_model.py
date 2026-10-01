"""Persistent adapter for the preserved Attention518 repeat run, without calibration."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import numpy as np
import torch
from torch.utils.data import DataLoader

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Predictor:
    def __init__(self, bundle_dir, device="cpu"):
        root = Path(bundle_dir).resolve()
        manifest = json.loads((root / 'MODEL_MANIFEST.json').read_text('utf-8'))
        if manifest['architecture'] != 'attention518_original_v1':
            raise ValueError('Expected the preserved Attention518 baseline recovery bundle')
        config = json.loads((root / 'plan.json').read_text('utf-8'))['config']
        if config['variant'] != 'attention_mae' or config['image_size'] != 518 or config['train_blocks'] != 2:
            raise ValueError('Unexpected model configuration')
        encoder = root / 'highres/encoder'
        if digest(encoder / 'model.pt') != manifest['encoder_weights']['sha256']:
            raise ValueError('Encoder checksum mismatch')
        self.deltas = []
        weight_hashes = []
        for fold in range(5):
            folder = root / 'primary' / f'fold{fold}'
            marker = json.loads((folder / 'complete.json').read_text('utf-8'))
            fold_config = json.loads((folder / 'config.json').read_text('utf-8'))
            if any(fold_config[k] != config[k] for k in ('variant', 'image_size', 'train_blocks', 'epochs')):
                raise ValueError('Fold configuration mismatch')
            for filename in ('delta.pt', 'config.json'):
                if digest(folder / filename) != marker['files'][filename]:
                    raise ValueError(f'Fold {fold} checksum mismatch: {filename}')
            self.deltas.append(folder / 'delta.pt')
            weight_hashes.append(marker['files']['delta.pt'])
        # A unique package name allows relative imports without shadowing application modules.
        name = "_postcode_bundle_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                name, root / "postcode_ml/__init__.py",
                submodule_search_locations=[str(root / "postcode_ml")],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        network = importlib.import_module(name + '.mae_spatial')
        vision = importlib.import_module(name + '.vision')
        self.device = torch.device('cuda:0' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device)
        cfg = vision.EncoderConfig(name='dinov2_vitb14', image_size=518, batch_size=2, device=str(self.device), amp=False)
        self.model = network.CargoRegressor(vision.load_encoder(cfg, encoder), 2, 'attention_mae').to(self.device)
        self.dataset = network.Photos
        self.predict_loader = network.predict_loader
        self.lock = threading.Lock()
        identity = json.dumps({'config':config,'encoder':manifest['encoder_weights']['sha256'],'folds':weight_hashes},sort_keys=True).encode()
        self.model_version = 'attention518-' + hashlib.sha256(identity).hexdigest()[:12]

    def predict(self, paths):
        if not paths:
            raise ValueError('At least one photo is required')
        loader = DataLoader(self.dataset(paths, image_size=518), batch_size=2, num_workers=0)
        with self.lock:
            predictions = []
            for path in self.deltas:
                self.model.load_delta(torch.load(path, map_location='cpu', weights_only=True))
                predictions.append(self.predict_loader(self.model, loader, self.device)[0])
        return np.stack(predictions).mean(0)
