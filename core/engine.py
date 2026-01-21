# core/engine.py
"""Main prediction engine."""

import os
import numpy as np
from typing import Dict, List, Optional

from config.settings import TRAINED_MODELS_DIR, DEVICE, PREDICTION_CONSTRAINTS
from config.model_config import (
    TARGET_CONFIG,
    MULTITASK_CONFIG,
    CHECKPOINT_VERSION_CONFIG,
    get_model_dir,
)


class PredictionEngine:
    """Unified prediction engine for all ADMET targets."""
    
    def __init__(self):
        self.predictors = {}
        self.loaded_targets = set()
        self._multitask_predictors = {}
    
    def get_model_path(self, target: str) -> str:
        """Get the model directory path for a target."""
        config = TARGET_CONFIG.get(target, {})
        
        if config.get('is_multitask'):
            group_name = config.get('multitask_group')
            if group_name and group_name in MULTITASK_CONFIG:
                model_folder = MULTITASK_CONFIG[group_name].get('model_folder', target)
                return os.path.join(TRAINED_MODELS_DIR, model_folder)
        
        return os.path.join(TRAINED_MODELS_DIR, target)
    
    def check_model_exists(self, target: str) -> bool:
        """Check if model files exist for a target."""
        model_dir = self.get_model_path(target)
        
        if not os.path.exists(model_dir):
            print(f"    Model dir not found: {model_dir}")
            return False
        
        files = os.listdir(model_dir)
        has_ckpt = any(f.endswith('.ckpt') for f in files)
        has_pkl = any(f.endswith('.pkl') for f in files)
        
        return has_ckpt or has_pkl
    
    def load_predictor(self, target: str) -> bool:
        """Load a predictor for the given target."""
        if target in self.loaded_targets:
            return True
        
        config = TARGET_CONFIG.get(target)
        if not config:
            print(f"    ⚠ No configuration found for {target}")
            return False
        
        model_dir = self.get_model_path(target)
        print(f"    Looking for model in: {model_dir}")
        
        if not os.path.exists(model_dir):
            print(f"    ⚠ Model directory not found: {model_dir}")
            return False
        
        training_style = config.get('training_style')
        checkpoint_version = CHECKPOINT_VERSION_CONFIG.get(target, 'latest')
        
        try:
            if config.get('is_multitask'):
                return self._load_multitask_predictor(target, config, model_dir)
            
            if training_style == 'optimized_caco_er':
                from models.caco_er_model import OptimizedCacoERModel
                predictor = OptimizedCacoERModel(model_dir)
                if predictor.load_models():
                    self.predictors[target] = predictor
                    self.loaded_targets.add(target)
                    print(f"    ✓ Loaded Caco-ER model")
                    return True
                else:
                    print(f"    ✗ Failed to load Caco-ER model")
                    return False
                    
            elif training_style == 'script1':
                from models.mpnn_predictors import MPNNPredictorScript1
                predictor = MPNNPredictorScript1(
                    model_dir=model_dir,
                    use_refinement=config.get('use_refinement', False),
                    checkpoint_version=checkpoint_version
                )
                if predictor.load_models():
                    self.predictors[target] = predictor
                    self.loaded_targets.add(target)
                    print(f"    ✓ Loaded Script1 model")
                    return True
                else:
                    print(f"    ✗ Failed to load Script1 model")
                    return False
                    
            elif training_style == 'script2':
                from models.mpnn_predictors import MPNNPredictorScript2
                predictor = MPNNPredictorScript2(
                    model_dir=model_dir,
                    use_refinement=config.get('use_refinement', False),
                    checkpoint_version=checkpoint_version
                )
                if predictor.load_models():
                    self.predictors[target] = predictor
                    self.loaded_targets.add(target)
                    print(f"    ✓ Loaded Script2 model")
                    return True
                else:
                    print(f"    ✗ Failed to load Script2 model")
                    return False
                    
            elif training_style == 'hybrid_v5':
                from models.mpnn_predictors import MPNNPredictorHybridV5Refinement
                predictor = MPNNPredictorHybridV5Refinement(
                    model_dir=model_dir,
                    checkpoint_version=checkpoint_version
                )
                if predictor.load_models():
                    self.predictors[target] = predictor
                    self.loaded_targets.add(target)
                    print(f"    ✓ Loaded HybridV5 Refinement model")
                    return True
                else:
                    print(f"    ✗ Failed to load HybridV5 Refinement model")
                    return False
                    
            elif training_style == 'hybrid_v5_integrated':
                from models.mpnn_predictors import MPNNPredictorHybridV5Integrated
                predictor = MPNNPredictorHybridV5Integrated(
                    model_dir=model_dir,
                    checkpoint_version=checkpoint_version
                )
                if predictor.load_models():
                    self.predictors[target] = predictor
                    self.loaded_targets.add(target)
                    print(f"    ✓ Loaded HybridV5 Integrated model")
                    return True
                else:
                    print(f"    ✗ Failed to load HybridV5 Integrated model")
                    return False
            
            print(f"    ⚠ Unknown training style: {training_style}")
            return False
            
        except Exception as e:
            print(f"    ⚠ Error loading predictor for {target}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_multitask_predictor(self, target: str, config: dict, model_dir: str) -> bool:
        """Load a multitask predictor."""
        group_name = config.get('multitask_group')
        
        # Check if already loaded
        if group_name in self._multitask_predictors:
            self.loaded_targets.add(target)
            print(f"    ✓ Using cached multitask predictor for {group_name}")
            return True
        
        mt_config = MULTITASK_CONFIG.get(group_name)
        if not mt_config:
            print(f"    ⚠ No multitask config for {group_name}")
            return False
        
        target_list = mt_config['targets']
        training_style = mt_config.get('training_style', 'unified')
        use_legacy = mt_config.get('use_legacy_descriptor_calc', False)
        checkpoint_version = CHECKPOINT_VERSION_CONFIG.get(group_name, 'latest')
        
        if mt_config.get('use_first_checkpoint'):
            checkpoint_version = 0
        
        print(f"    Loading multitask model for {group_name}")
        print(f"    Model dir: {model_dir}")
        print(f"    Targets: {target_list}")
        
        try:
            if use_legacy:
                from models.multitask_predictors import LegacyMultitaskMPNNPredictor
                predictor = LegacyMultitaskMPNNPredictor(
                    group_name=group_name,
                    target_list=target_list,
                    model_dir=model_dir,
                    checkpoint_version=checkpoint_version
                )
            elif training_style == 'unified':
                from models.multitask_predictors import MultitaskMPNNPredictorUnified
                predictor = MultitaskMPNNPredictorUnified(
                    group_name=group_name,
                    target_list=target_list,
                    model_dir=model_dir,
                    checkpoint_version=checkpoint_version
                )
            else:
                from models.multitask_predictors import MultitaskMPNNPredictorScript1
                predictor = MultitaskMPNNPredictorScript1(
                    group_name=group_name,
                    target_list=target_list,
                    model_dir=model_dir,
                    checkpoint_version=checkpoint_version
                )
            
            if predictor.load_models():
                self._multitask_predictors[group_name] = predictor
                for t in target_list:
                    self.loaded_targets.add(t)
                print(f"    ✓ Loaded multitask model for {group_name}")
                return True
            else:
                print(f"    ✗ Failed to load multitask model for {group_name}")
                return False
                
        except Exception as e:
            print(f"    ⚠ Error loading multitask predictor: {e}")
            import traceback
            traceback.print_exc()
        
        return False
    
    def predict(self, target: str, smiles_list: List[str]) -> np.ndarray:
        """Make predictions for a target."""
        config = TARGET_CONFIG.get(target, {})
        
        # Handle multitask targets
        if config.get('is_multitask'):
            group_name = config.get('multitask_group')
            if group_name in self._multitask_predictors:
                predictions = self._multitask_predictors[group_name].predict(smiles_list)
                if target in predictions:
                    return self._apply_constraints(target, predictions[target])
            return np.zeros(len(smiles_list))
        
        # Handle single-task targets
        if target in self.predictors:
            preds = self.predictors[target].predict(smiles_list)
            return self._apply_constraints(target, preds)
        
        return np.zeros(len(smiles_list))
    
    def _apply_constraints(self, target: str, predictions: np.ndarray) -> np.ndarray:
        """Apply prediction constraints (clipping)."""
        # Ensure 1D array
        predictions = np.atleast_1d(predictions).flatten()
        
        if target in PREDICTION_CONSTRAINTS:
            constraints = PREDICTION_CONSTRAINTS[target]
            min_val = constraints.get('min')
            max_val = constraints.get('max')
            
            if min_val is not None:
                predictions = np.maximum(predictions, min_val)
            if max_val is not None:
                predictions = np.minimum(predictions, max_val)
        
        return predictions
    
    def predict_batch(self, targets: List[str], smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """Predict multiple targets at once."""
        results = {}
        
        for target in targets:
            if target not in self.loaded_targets:
                if not self.load_predictor(target):
                    results[target] = np.zeros(len(smiles_list))
                    continue
            
            results[target] = self.predict(target, smiles_list)
        
        return results
    
    def get_available_targets(self) -> List[str]:
        """Get list of targets that have models available."""
        available = []
        for target in TARGET_CONFIG.keys():
            if self.check_model_exists(target):
                available.append(target)
        return available