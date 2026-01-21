# utils/helpers.py
"""Helper functions."""

import os
import re
import glob
import numpy as np
from typing import List, Dict

from config.settings import ADAPTIVE_CONFIG

# Add to utils/helpers.py

import torch

def get_lightning_trainer():
    """Get a properly configured Lightning trainer for inference."""
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl
    
    from config.settings import ACCELERATOR, DEVICES
    
    # Explicitly configure for GPU if available
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            inference_mode=True,
        )
        print(f"Trainer using GPU: {torch.cuda.get_device_name(0)}")
    else:
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            inference_mode=True,
        )
        print("Trainer using CPU")
    
    return trainer
def get_version_checkpoints(model_dir: str, version: str | int) -> List[str]:
    """Get checkpoint files for a specific version."""
    all_ckpts = glob.glob(os.path.join(model_dir, 'fold_*.ckpt'))
    if not all_ckpts:
        return []
    
    fold_versions = {}
    for ckpt_path in all_ckpts:
        filename = os.path.basename(ckpt_path)
        match = re.match(r'fold_(\d+)(?:-v(\d+))?\.ckpt', filename)
        if match:
            fold_num = int(match.group(1))
            ver = int(match.group(2)) if match.group(2) else 0
            if fold_num not in fold_versions:
                fold_versions[fold_num] = {}
            fold_versions[fold_num][ver] = ckpt_path
    
    selected_ckpts = []
    if version == 'first':
        version = 0
    
    for fold_num in sorted(fold_versions.keys()):
        versions = fold_versions[fold_num]
        if version == 'latest':
            selected_ver = max(versions.keys())
        elif isinstance(version, int):
            if version in versions:
                selected_ver = version
            else:
                available = sorted(versions.keys())
                if version < min(available):
                    selected_ver = min(available)
                else:
                    selected_ver = max(v for v in available if v <= version)
        else:
            selected_ver = max(versions.keys())
        
        selected_ckpts.append(versions[selected_ver])
    
    return selected_ckpts


def remove_outliers(y: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Remove outliers using modified Z-score."""
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    if mad == 0:
        return np.ones(len(y), dtype=bool)
    modified_z = 0.6745 * (y - median) / mad
    return np.abs(modified_z) < threshold


def get_adaptive_config(n_samples: int) -> Dict:
    """Get adaptive configuration based on dataset size."""
    if n_samples < ADAPTIVE_CONFIG['small']['threshold']:
        config = ADAPTIVE_CONFIG['small'].copy()
    elif n_samples < ADAPTIVE_CONFIG['medium']['threshold']:
        config = ADAPTIVE_CONFIG['medium'].copy()
    elif n_samples < ADAPTIVE_CONFIG['large']['threshold']:
        config = ADAPTIVE_CONFIG['large'].copy()
    else:
        config = ADAPTIVE_CONFIG['very_large'].copy()
    config['n_samples'] = n_samples
    return config