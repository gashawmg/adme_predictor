# consolidate_models.py
"""Script to consolidate all model files into the trained_models folder."""

import os
import shutil
from pathlib import Path


def consolidate_models():
    """Consolidate all models into the trained_models directory."""
    
    # ==========================================================================
    # CONFIGURE THESE PATHS TO MATCH YOUR SETUP
    # ==========================================================================
    
    # Destination directory (inside your app folder)
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    DEST_DIR = os.path.join(APP_DIR, "trained_models")
    
    # Source paths - UPDATE THESE TO YOUR ACTUAL MODEL LOCATIONS
    # Format: 'target_name': 'source_path'
    SOURCE_PATHS = {
        # Single-task models
        'LogD': r'C:\Users\gasha\ADMET_competion\best_models_12_30\LogD',
        'Log_HLM_CLint': r'C:\Users\gasha\ADMET_competion\trained_models_fixed\Log_HLM_CLint',
        'Log_MLM_CLint': r'C:\Users\gasha\ADMET_competion\trained_models_fixed\Log_MLM_CLint',
        'Log_Caco_Papp_AB': r'C:\Users\gasha\ADMET_competion\best_models_12_30\Log_Caco_Papp_AB',
        'Log_Caco_ER': r'C:\Users\gasha\ADMET_competion\best_models_12_30\caco_optimized',
        'Log_Mouse_PPB': r'C:\Users\gasha\ADMET_competion\best_models_12_30\Log_Mouse_PPB',
        'Log_Mouse_BPB': r'C:\Users\gasha\ADMET_competion\clean_folder\Log_Mouse_BPB',
        
        # Multitask models (stored under output target name)
        'LogS': r'C:\Users\gasha\ADMET_competion\best_models_01_02\MULTITASK_LOGD_LOGS',
        'Log_Mouse_MPB': r'C:\Users\gasha\ADMET_competion\best_models_01_02\MULTITASK_MOUSE_BINDING',
    }
    
    # ==========================================================================
    # RUN CONSOLIDATION
    # ==========================================================================
    
    print("=" * 70)
    print("MODEL CONSOLIDATION SCRIPT")
    print("=" * 70)
    print(f"\nDestination: {DEST_DIR}")
    
    # Create destination directory
    os.makedirs(DEST_DIR, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for target_name, source_path in SOURCE_PATHS.items():
        dest_path = os.path.join(DEST_DIR, target_name)
        
        print(f"\n{target_name}:")
        print(f"  Source: {source_path}")
        print(f"  Dest:   {dest_path}")
        
        if not os.path.exists(source_path):
            print(f"  ✗ Source not found!")
            fail_count += 1
            continue
        
        try:
            # Create destination folder
            os.makedirs(dest_path, exist_ok=True)
            
            # Copy all files
            files_copied = 0
            for item in os.listdir(source_path):
                src_item = os.path.join(source_path, item)
                dst_item = os.path.join(dest_path, item)
                
                if os.path.isfile(src_item):
                    shutil.copy2(src_item, dst_item)
                    files_copied += 1
                elif os.path.isdir(src_item):
                    if os.path.exists(dst_item):
                        shutil.rmtree(dst_item)
                    shutil.copytree(src_item, dst_item)
                    files_copied += 1
            
            print(f"  ✓ Copied {files_copied} items")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {fail_count}")
    
    # Verify final structure
    print("\n" + "-" * 70)
    print("FINAL STRUCTURE")
    print("-" * 70)
    
    if os.path.exists(DEST_DIR):
        for folder in sorted(os.listdir(DEST_DIR)):
            folder_path = os.path.join(DEST_DIR, folder)
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                ckpt_count = len([f for f in files if f.endswith('.ckpt')])
                pkl_count = len([f for f in files if f.endswith('.pkl')])
                print(f"  {folder}: {ckpt_count} ckpt, {pkl_count} pkl")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
    if fail_count == 0:
        print("\n✅ All models consolidated successfully!")
        print("   You can now run: streamlit run app.py")
    else:
        print(f"\n⚠️ {fail_count} models failed. Please check the paths above.")


if __name__ == "__main__":
    consolidate_models()