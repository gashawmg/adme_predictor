# models/multitask_predictors.py
"""Multitask MPNN predictor classes - matching original script GPU usage."""

import os
import glob
import numpy as np
import torch
from typing import List, Dict, Optional

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from sklearn.preprocessing import RobustScaler, StandardScaler

from rdkit import Chem
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models import MPNN

from config.model_config import MULTITASK_DESCRIPTOR_CONFIGS
from utils.io_utils import load_pickle
from utils.helpers import get_version_checkpoints
from core.descriptors import MPNNDescriptorCalculator, LegacyMPNNDescriptorCalculator

# Match original script's device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[multitask_predictors] Using device: {DEVICE}")


class BaseMultitaskPredictor:
    """Base class for multitask predictors - matches original script."""
    
    def __init__(self, group_name: str, target_list: List[str], model_dir: str,
                 checkpoint_version='latest'):
        self.group_name = group_name
        self.target_list = target_list
        self.n_tasks = len(target_list)
        self.model_dir = model_dir
        self.checkpoint_version = checkpoint_version
        self.mpnn_models = []
        self.scalers_y = {}
        self.desc_calc = None
    
    def _load_checkpoint_safe(self, path: str) -> Optional[MPNN]:
        """Load checkpoint - matching original script."""
        try:
            model = MPNN.load_from_checkpoint(path, map_location=DEVICE, weights_only=False)
            model.eval()
            model.to(torch.device(DEVICE))
            return model
        except Exception:
            try:
                model = MPNN.load_from_checkpoint(path, map_location=DEVICE)
                model.eval()
                model.to(torch.device(DEVICE))
                return model
            except Exception as e:
                print(f"      ✗ Failed to load {path}: {e}")
                return None
    
    def _create_trainer(self) -> pl.Trainer:
        """Create trainer - matching original script."""
        return pl.Trainer(
            accelerator='gpu' if DEVICE == 'cuda' else 'cpu',
            devices=1,
            logger=False,
            enable_progress_bar=False,
        )


class LegacyMultitaskMPNNPredictor(BaseMultitaskPredictor):
    """Legacy multitask predictor - matches original script."""
    
    def __init__(self, group_name: str, target_list: List[str], model_dir: str,
                 checkpoint_version='latest'):
        super().__init__(group_name, target_list, model_dir, checkpoint_version)
        self._need_scaler_fit = False
    
    def load_models(self) -> bool:
        """Load model components."""
        print(f"      Loading MultitaskPredictor for {self.group_name}")
        print(f"      Device: {DEVICE}")
        
        try:
            if not os.path.exists(self.model_dir):
                print(f"      ✗ Directory not found: {self.model_dir}")
                return False
            
            # Load scalers
            scalers_path = os.path.join(self.model_dir, 'scalers_y.pkl')
            if not os.path.exists(scalers_path):
                print(f"      ✗ scalers_y.pkl not found")
                return False
            
            self.scalers_y = load_pickle(scalers_path)
            print(f"      ✓ Loaded scalers for: {list(self.scalers_y.keys())}")
            
            # Load descriptor list
            desc_list_path = os.path.join(self.model_dir, 'desc_list.pkl')
            if not os.path.exists(desc_list_path):
                print(f"      ✗ desc_list.pkl not found")
                return False
            
            desc_list = load_pickle(desc_list_path)
            self.desc_calc = LegacyMPNNDescriptorCalculator(desc_list, 'robust')
            
            # Load descriptor scaler
            desc_scaler_path = os.path.join(self.model_dir, 'desc_scaler.pkl')
            if os.path.exists(desc_scaler_path):
                self.desc_calc.scaler = load_pickle(desc_scaler_path)
            else:
                self._need_scaler_fit = True
            
            # Load checkpoints
            ckpts = sorted(glob.glob(os.path.join(self.model_dir, 'fold_*.ckpt')))
            
            if not ckpts:
                print(f"      ✗ No checkpoints found")
                return False
            
            self.mpnn_models = []
            for ckpt in ckpts:
                model = self._load_checkpoint_safe(ckpt)
                if model:
                    self.mpnn_models.append(model)
                    print(f"      ✓ Loaded {os.path.basename(ckpt)} to {DEVICE}")
            
            print(f"      Total models loaded: {len(self.mpnn_models)}")
            return len(self.mpnn_models) > 0
            
        except Exception as e:
            print(f"      ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """Make predictions."""
        if not self.mpnn_models:
            return {t: np.zeros(len(smiles_list)) for t in self.target_list}
        
        # Calculate descriptors
        if self._need_scaler_fit:
            desc = self.desc_calc.fit_transform(smiles_list)
            self._need_scaler_fit = False
        else:
            desc = self.desc_calc.transform(smiles_list)
        
        # Create datapoints
        dps, valid_indices = [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                dps.append(MoleculeDatapoint(mol=mol, x_d=desc[i].astype(np.float32)))
                valid_indices.append(i)
        
        if not dps:
            return {t: np.zeros(len(smiles_list)) for t in self.target_list}
        
        loader = build_dataloader(MoleculeDataset(dps), batch_size=64, shuffle=False, num_workers=0)
        
        # Predict with each model
        all_preds = []
        for model in self.mpnn_models:
            model.eval()
            trainer = self._create_trainer()
            
            with torch.inference_mode():
                batch_preds = trainer.predict(model, loader)
            
            preds = np.vstack([p.cpu().numpy() for p in batch_preds])
            all_preds.append(preds)
        
        # Average predictions
        avg = np.mean(all_preds, axis=0)
        
        # Build results
        results = {}
        for i, target in enumerate(self.target_list):
            if target in self.scalers_y:
                inv = self.scalers_y[target].inverse_transform(avg[:, i].reshape(-1, 1)).ravel()
            else:
                inv = avg[:, i]
            
            full = np.zeros(len(smiles_list))
            for idx, val_idx in enumerate(valid_indices):
                if idx < len(inv):
                    full[val_idx] = inv[idx]
            
            results[target] = full
        
        return results


# Aliases for compatibility
MultitaskMPNNPredictorUnified = LegacyMultitaskMPNNPredictor
MultitaskMPNNPredictorScript1 = LegacyMultitaskMPNNPredictor