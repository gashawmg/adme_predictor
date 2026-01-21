# check_models.py
"""Check model directory structure."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import TRAINED_MODELS_DIR
from config.model_config import TARGET_CONFIG, MULTITASK_CONFIG

print("=" * 70)
print("MODEL DIRECTORY CHECK")
print("=" * 70)
print(f"\nTRAINED_MODELS_DIR: {TRAINED_MODELS_DIR}")
print(f"Exists: {os.path.exists(TRAINED_MODELS_DIR)}")

if os.path.exists(TRAINED_MODELS_DIR):
    print(f"\nContents:")
    for item in sorted(os.listdir(TRAINED_MODELS_DIR)):
        item_path = os.path.join(TRAINED_MODELS_DIR, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            ckpt = len([f for f in files if f.endswith('.ckpt')])
            pkl = len([f for f in files if f.endswith('.pkl')])
            print(f"  📁 {item}/  ({ckpt} ckpt, {pkl} pkl)")
            # Show pkl file names
            pkl_files = [f for f in files if f.endswith('.pkl')]
            for pf in pkl_files:
                print(f"      - {pf}")
        else:
            print(f"  📄 {item}")

print("\n" + "-" * 70)
print("EXPECTED STRUCTURE FOR EACH TARGET")
print("-" * 70)

for target, config in TARGET_CONFIG.items():
    is_multitask = config.get('is_multitask', False)
    
    if is_multitask:
        group = config.get('multitask_group')
        mt_config = MULTITASK_CONFIG.get(group, {})
        folder = mt_config.get('model_folder', target)
    else:
        folder = target
    
    expected_path = os.path.join(TRAINED_MODELS_DIR, folder)
    exists = os.path.exists(expected_path)
    
    status = "✓" if exists else "✗"
    mt_note = f" (multitask: {config.get('multitask_group')})" if is_multitask else ""
    
    print(f"  {status} {target}: {folder}/{mt_note}")
    
    if exists:
        files = os.listdir(expected_path)
        ckpt_files = [f for f in files if f.endswith('.ckpt')]
        pkl_files = [f for f in files if f.endswith('.pkl')]
        print(f"      Checkpoints: {len(ckpt_files)}")
        print(f"      PKL files: {pkl_files}")
        
        # Check for required files based on model type
        training_style = config.get('training_style')
        
        if training_style == 'optimized_caco_er':
            required = ['models.pkl']
        elif is_multitask:
            # Multitask models use scalers_y.pkl (plural)
            required = ['scalers_y.pkl', 'desc_list.pkl']
        else:
            # Single-task models use scaler_y.pkl (singular)
            required = ['scaler_y.pkl']
        
        for req in required:
            if req in files:
                print(f"      ✓ {req} found")
            else:
                print(f"      ✗ {req} MISSING!")

print("\n" + "=" * 70)