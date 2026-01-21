# check_model_structure.py
"""Check model folder structure."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import TRAINED_MODELS_DIR
from config.model_config import TARGET_CONFIG, MULTITASK_CONFIG

print("=" * 70)
print("MODEL STRUCTURE CHECK")
print("=" * 70)
print(f"\nBase directory: {TRAINED_MODELS_DIR}")
print(f"Exists: {os.path.exists(TRAINED_MODELS_DIR)}\n")

# Expected files for each model type
EXPECTED_FILES = {
    'script1': ['scaler_y.pkl', 'desc_list_integrated.pkl', 'desc_scaler_integrated.pkl'],
    'script2': ['scaler_y.pkl', 'desc_list_integrated.pkl', 'desc_scaler_integrated.pkl'],
    'hybrid_v5': ['scaler_y.pkl', 'refinement_stack.pkl', 'desc_scaler_refinement.pkl'],
    'hybrid_v5_integrated': ['scaler_y.pkl', 'desc_list_integrated.pkl', 'desc_scaler_integrated.pkl'],
    'optimized_caco_er': ['models.pkl'],
    'multitask': ['scalers_y.pkl', 'desc_list.pkl', 'desc_scaler.pkl'],
}

for target, config in TARGET_CONFIG.items():
    is_multitask = config.get('is_multitask', False)
    training_style = config.get('training_style', 'unknown')
    
    if is_multitask:
        group = config.get('multitask_group')
        mt_config = MULTITASK_CONFIG.get(group, {})
        folder = mt_config.get('model_folder', target)
        expected = EXPECTED_FILES.get('multitask', [])
    else:
        folder = target
        expected = EXPECTED_FILES.get(training_style, [])
    
    model_dir = os.path.join(TRAINED_MODELS_DIR, folder)
    
    print(f"\n{target}:")
    print(f"  Folder: {folder}/")
    print(f"  Style: {training_style}")
    print(f"  Multitask: {is_multitask}")
    
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        ckpts = [f for f in files if f.endswith('.ckpt')]
        pkls = [f for f in files if f.endswith('.pkl')]
        
        print(f"  ✓ Directory exists")
        print(f"  Checkpoints: {len(ckpts)}")
        print(f"  PKL files: {pkls}")
        
        # Check expected files
        missing = []
        for exp_file in expected:
            if exp_file not in files:
                missing.append(exp_file)
        
        if missing:
            print(f"  ⚠ Missing: {missing}")
        else:
            print(f"  ✓ All expected files present")
    else:
        print(f"  ✗ Directory NOT FOUND")

print("\n" + "=" * 70)