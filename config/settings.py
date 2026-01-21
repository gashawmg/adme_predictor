# config/settings.py
"""Global settings and constants."""

import os
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import torch

# =============================================================================
# DEVICE CONFIGURATION - Matching original training script
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Print GPU status prominently
print("=" * 60)
if DEVICE == "cuda":
    print(f"🚀 GPU ENABLED: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Match original script settings
    torch.set_float32_matmul_precision('medium')
else:
    print("⚠️ CPU MODE - No GPU detected")
print("=" * 60)

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_MODELS_DIR = os.path.join(APP_DIR, "trained_models")
MODEL_SAVE_DIR = TRAINED_MODELS_DIR

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    'n_folds': 5,
    'random_seed': GLOBAL_SEED,
    'outlier_threshold': 3.5,
    'use_augmentation': True,
    'early_stopping_patience': 15,
    'gradient_clip_val': 1.0,
}

ADAPTIVE_CONFIG = {
    'small': {'threshold': 200, 'hidden_dim': 256, 'depth': 3, 'n_layers': 2, 'dropout': 0.25,
              'max_epochs': 100, 'patience': 10, 'batch_size': 32, 'augmentation_factor': 5, 'use_augmentation': True},
    'medium': {'threshold': 500, 'hidden_dim': 512, 'depth': 4, 'n_layers': 3, 'dropout': 0.20,
               'max_epochs': 150, 'patience': 15, 'batch_size': 64, 'augmentation_factor': 3, 'use_augmentation': True},
    'large': {'threshold': 1000, 'hidden_dim': 768, 'depth': 5, 'n_layers': 3, 'dropout': 0.15,
              'max_epochs': 200, 'patience': 20, 'batch_size': 64, 'augmentation_factor': 2, 'use_augmentation': True},
    'very_large': {'threshold': float('inf'), 'hidden_dim': 768, 'depth': 6, 'n_layers': 4, 'dropout': 0.10,
                   'max_epochs': 200, 'patience': 20, 'batch_size': 64, 'augmentation_factor': 1, 'use_augmentation': False},
}

PREDICTION_CONSTRAINTS = {
    'Log_Mouse_PPB': {'min': 0.0, 'max': 2.0},
    'Log_Mouse_BPB': {'min': 0.0, 'max': 2.0},
    'Log_Mouse_MPB': {'min': 0.0, 'max': 2.0},
    'Log_HLM_CLint': {'min': 0.0, 'max': None},
    'Log_MLM_CLint': {'min': 0.0, 'max': None},
}