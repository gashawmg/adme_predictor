# models/mpnn_predictors.py
"""MPNN predictor classes - matching original training script GPU usage."""

import os
import numpy as np
import torch
from typing import List, Optional

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from sklearn.preprocessing import RobustScaler

from rdkit import Chem
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models import MPNN

from utils.io_utils import load_pickle
from utils.helpers import get_version_checkpoints
from core.descriptors import (
    MPNNDescriptorCalculator,
    MPNNRefinementDescriptorCalculator,
)

# Match original script's device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[mpnn_predictors] Using device: {DEVICE}")


class BaseMPNNPredictor:
    """Base class for MPNN predictors - matches original script."""
    
    def __init__(self, model_dir: str, checkpoint_version='latest'):
        self.model_dir = model_dir
        self.checkpoint_version = checkpoint_version
        self.mpnn_models = []
        self.scaler_y = RobustScaler()
    
    def _load_checkpoint_safe(self, path: str) -> Optional[MPNN]:
        """Load checkpoint - matching original script."""
        try:
            # Match original: map_location=DEVICE
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
        # This matches the original script exactly
        return pl.Trainer(
            accelerator='gpu' if DEVICE == 'cuda' else 'cpu',
            devices=1,
            logger=False,
            enable_progress_bar=False,
        )
    
    def _predict_with_ensemble(self, loader) -> np.ndarray:
        """Predict with model ensemble - matching original script."""
        all_preds = []
        
        for model in self.mpnn_models:
            model.eval()
            trainer = self._create_trainer()
            
            with torch.inference_mode():
                batch_predictions = trainer.predict(model, loader)
            
            preds = np.concatenate([p.flatten().cpu().numpy() for p in batch_predictions])
            all_preds.append(preds)
        
        return np.mean(all_preds, axis=0)


class MPNNPredictorScript1(BaseMPNNPredictor):
    """Script-1 style: RobustScaler, fold-wise augmentation, optional refinement."""
    
    def __init__(self, model_dir: str, use_refinement: bool = False,
                 checkpoint_version='latest'):
        super().__init__(model_dir, checkpoint_version)
        self.use_refinement = use_refinement
        self.desc_calc_integrated = None
        self.refinement_model = None
        self.desc_calc_refinement = None
    
    def load_models(self) -> bool:
        """Load model components."""
        print(f"      Loading Script1 model from {self.model_dir}")
        print(f"      Device: {DEVICE}")
        
        try:
            # Load scaler
            scaler_y_path = os.path.join(self.model_dir, 'scaler_y.pkl')
            if os.path.exists(scaler_y_path):
                self.scaler_y = load_pickle(scaler_y_path)
            
            # Load descriptor calculator
            desc_list_path = os.path.join(self.model_dir, 'desc_list_integrated.pkl')
            desc_scaler_path = os.path.join(self.model_dir, 'desc_scaler_integrated.pkl')
            
            if os.path.exists(desc_list_path):
                desc_list = load_pickle(desc_list_path)
                self.desc_calc_integrated = MPNNDescriptorCalculator(desc_list, 'robust')
                if os.path.exists(desc_scaler_path):
                    self.desc_calc_integrated.scaler = load_pickle(desc_scaler_path)
            
            # Load refinement if needed
            if self.use_refinement:
                refinement_path = os.path.join(self.model_dir, 'refinement_stack.pkl')
                if os.path.exists(refinement_path):
                    self.refinement_model = load_pickle(refinement_path)
                    self.desc_calc_refinement = MPNNRefinementDescriptorCalculator('robust')
                    ref_scaler_path = os.path.join(self.model_dir, 'desc_scaler_refinement.pkl')
                    if os.path.exists(ref_scaler_path):
                        self.desc_calc_refinement.scaler = load_pickle(ref_scaler_path)
            
            # Load checkpoints
            ckpts = get_version_checkpoints(self.model_dir, self.checkpoint_version)
            
            self.mpnn_models = []
            for ckpt in ckpts:
                model = self._load_checkpoint_safe(ckpt)
                if model:
                    self.mpnn_models.append(model)
                    print(f"      ✓ Loaded {os.path.basename(ckpt)} to {DEVICE}")
            
            print(f"      Total models loaded: {len(self.mpnn_models)}")
            return len(self.mpnn_models) > 0
            
        except Exception as e:
            print(f"      ✗ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Make predictions."""
        if not self.mpnn_models:
            return np.zeros(len(smiles_list))
        
        # Calculate descriptors
        if self.desc_calc_integrated is not None:
            integrated_descs = self.desc_calc_integrated.transform(smiles_list)
        else:
            integrated_descs = None
        
        # Create datapoints
        dps, valid_indices = [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                desc = integrated_descs[i].astype(np.float32) if integrated_descs is not None else None
                dps.append(MoleculeDatapoint(mol=mol, x_d=desc))
                valid_indices.append(i)
        
        if not dps:
            return np.zeros(len(smiles_list))
        
        loader = build_dataloader(MoleculeDataset(dps), batch_size=64, shuffle=False, num_workers=0)
        
        # Predict
        avg_preds = self._predict_with_ensemble(loader)
        final_preds = self.scaler_y.inverse_transform(avg_preds.reshape(-1, 1)).ravel()
        
        # Map to full array
        full = np.zeros(len(smiles_list))
        for idx, val in zip(valid_indices, final_preds):
            full[idx] = val
        
        # Apply refinement if available
        if self.use_refinement and self.refinement_model and self.desc_calc_refinement:
            X_refine = self.desc_calc_refinement.transform(smiles_list)
            refinement_correction = self.refinement_model.predict(X_refine)
            full = full + refinement_correction
        
        return full


class MPNNPredictorScript2(BaseMPNNPredictor):
    """Script-2 style: StandardScaler, pre-split augmentation."""
    
    def __init__(self, model_dir: str, use_refinement: bool = False,
                 checkpoint_version='latest'):
        super().__init__(model_dir, checkpoint_version)
        self.use_refinement = use_refinement
        self.desc_calc_integrated = None
    
    def load_models(self) -> bool:
        """Load model components."""
        print(f"      Loading Script2 model from {self.model_dir}")
        print(f"      Device: {DEVICE}")
        
        try:
            scaler_y_path = os.path.join(self.model_dir, 'scaler_y.pkl')
            if os.path.exists(scaler_y_path):
                self.scaler_y = load_pickle(scaler_y_path)
            
            desc_list_path = os.path.join(self.model_dir, 'desc_list_integrated.pkl')
            desc_scaler_path = os.path.join(self.model_dir, 'desc_scaler_integrated.pkl')
            
            if os.path.exists(desc_list_path):
                desc_list = load_pickle(desc_list_path)
                self.desc_calc_integrated = MPNNDescriptorCalculator(desc_list, 'standard')
                if os.path.exists(desc_scaler_path):
                    self.desc_calc_integrated.scaler = load_pickle(desc_scaler_path)
            
            ckpts = get_version_checkpoints(self.model_dir, self.checkpoint_version)
            
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
            print(f"      ✗ Error loading models: {e}")
            return False
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Make predictions."""
        if not self.mpnn_models:
            return np.zeros(len(smiles_list))
        
        integrated_descs = self.desc_calc_integrated.transform(smiles_list) if self.desc_calc_integrated else None
        
        dps, valid_indices = [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                desc = integrated_descs[i].astype(np.float32) if integrated_descs is not None else None
                dps.append(MoleculeDatapoint(mol=mol, x_d=desc))
                valid_indices.append(i)
        
        if not dps:
            return np.zeros(len(smiles_list))
        
        loader = build_dataloader(MoleculeDataset(dps), batch_size=64, shuffle=False, num_workers=0)
        
        avg_preds = self._predict_with_ensemble(loader)
        final_preds = self.scaler_y.inverse_transform(avg_preds.reshape(-1, 1)).ravel()
        
        full = np.zeros(len(smiles_list))
        for idx, val in zip(valid_indices, final_preds):
            full[idx] = val
        
        return full


class MPNNPredictorHybridV5Integrated(BaseMPNNPredictor):
    """Hybrid V5 style MPNN with integrated descriptors."""
    
    def __init__(self, model_dir: str, checkpoint_version='latest'):
        super().__init__(model_dir, checkpoint_version)
        self.desc_calc_integrated = None
    
    def load_models(self) -> bool:
        """Load model components."""
        print(f"      Loading HybridV5Integrated from {self.model_dir}")
        print(f"      Device: {DEVICE}")
        
        try:
            self.scaler_y = load_pickle(os.path.join(self.model_dir, 'scaler_y.pkl'))
            desc_list = load_pickle(os.path.join(self.model_dir, 'desc_list_integrated.pkl'))
            self.desc_calc_integrated = MPNNDescriptorCalculator(desc_list, 'standard')
            self.desc_calc_integrated.scaler = load_pickle(
                os.path.join(self.model_dir, 'desc_scaler_integrated.pkl')
            )
            
            ckpts = get_version_checkpoints(self.model_dir, self.checkpoint_version)
            
            self.mpnn_models = []
            for ckpt in ckpts:
                model = self._load_checkpoint_safe(ckpt)
                if model:
                    self.mpnn_models.append(model)
                    print(f"      ✓ Loaded {os.path.basename(ckpt)} to {DEVICE}")
            
            print(f"      Total models loaded: {len(self.mpnn_models)}")
            return len(self.mpnn_models) > 0
            
        except Exception as e:
            print(f"      ✗ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Make predictions."""
        if not self.mpnn_models:
            return np.zeros(len(smiles_list))
        
        integrated_descs = self.desc_calc_integrated.transform(smiles_list)
        
        dps, valid_indices = [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                dps.append(MoleculeDatapoint(mol=mol, x_d=integrated_descs[i].astype(np.float32)))
                valid_indices.append(i)
        
        if not dps:
            return np.zeros(len(smiles_list))
        
        loader = build_dataloader(MoleculeDataset(dps), batch_size=64, shuffle=False, num_workers=0)
        
        avg_preds = self._predict_with_ensemble(loader)
        final_preds = self.scaler_y.inverse_transform(avg_preds.reshape(-1, 1)).ravel()
        
        full = np.zeros(len(smiles_list))
        for idx, val in zip(valid_indices, final_preds):
            full[idx] = val
        
        return full


class MPNNPredictorHybridV5Refinement(BaseMPNNPredictor):
    """Hybrid V5 style MPNN with refinement stack."""
    
    def __init__(self, model_dir: str, checkpoint_version='latest'):
        super().__init__(model_dir, checkpoint_version)
        self.refinement_model = None
        self.desc_calc_refinement = None
    
    def load_models(self) -> bool:
        """Load model components."""
        print(f"      Loading HybridV5Refinement from {self.model_dir}")
        print(f"      Device: {DEVICE}")
        
        try:
            self.scaler_y = load_pickle(os.path.join(self.model_dir, 'scaler_y.pkl'))
            
            refinement_path = os.path.join(self.model_dir, 'refinement_stack.pkl')
            scaler_path = os.path.join(self.model_dir, 'desc_scaler_refinement.pkl')
            
            if os.path.exists(refinement_path):
                self.refinement_model = load_pickle(refinement_path)
                self.desc_calc_refinement = MPNNRefinementDescriptorCalculator('standard')
                if os.path.exists(scaler_path):
                    self.desc_calc_refinement.scaler = load_pickle(scaler_path)
                print(f"      ✓ Loaded refinement stack")
            
            ckpts = get_version_checkpoints(self.model_dir, self.checkpoint_version)
            
            self.mpnn_models = []
            for ckpt in ckpts:
                model = self._load_checkpoint_safe(ckpt)
                if model:
                    self.mpnn_models.append(model)
                    print(f"      ✓ Loaded {os.path.basename(ckpt)} to {DEVICE}")
            
            print(f"      Total models loaded: {len(self.mpnn_models)}")
            return len(self.mpnn_models) > 0
            
        except Exception as e:
            print(f"      ✗ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Make predictions."""
        if not self.mpnn_models:
            return np.zeros(len(smiles_list))
        
        dps, valid_indices = [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                dps.append(MoleculeDatapoint(mol=mol, x_d=None))
                valid_indices.append(i)
        
        if not dps:
            return np.zeros(len(smiles_list))
        
        loader = build_dataloader(MoleculeDataset(dps), batch_size=64, shuffle=False, num_workers=0)
        
        avg_preds = self._predict_with_ensemble(loader)
        base_preds = self.scaler_y.inverse_transform(avg_preds.reshape(-1, 1)).ravel()
        
        full = np.zeros(len(smiles_list))
        for idx, val in zip(valid_indices, base_preds):
            full[idx] = val
        
        # Apply refinement
        if self.refinement_model and self.desc_calc_refinement:
            X_refine = self.desc_calc_refinement.transform(smiles_list)
            refinement_correction = self.refinement_model.predict(X_refine)
            full = full + refinement_correction
        
        return full