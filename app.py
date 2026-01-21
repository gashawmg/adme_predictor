# app.py
"""ADMET Predictor Streamlit Application."""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import sys
import torch

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="ADMET Predictor",
    page_icon="🧪",
    layout="wide"
)

# =============================================================================
# Register classes for pickle compatibility BEFORE any model loading
# =============================================================================
def register_pickle_classes():
    """Register classes in __main__ for pickle compatibility."""
    import __main__
    
    try:
        from models.refinement import RefinementStack
        __main__.RefinementStack = RefinementStack
    except Exception as e:
        print(f"Warning: Could not register RefinementStack: {e}")
    
    try:
        from core.descriptors import (
            MPNNDescriptorCalculator,
            LegacyMPNNDescriptorCalculator,
            MPNNRefinementDescriptorCalculator,
        )
        __main__.MPNNDescriptorCalculator = MPNNDescriptorCalculator
        __main__.LegacyMPNNDescriptorCalculator = LegacyMPNNDescriptorCalculator
        __main__.MPNNRefinementDescriptorCalculator = MPNNRefinementDescriptorCalculator
    except Exception as e:
        print(f"Warning: Could not register descriptor classes: {e}")
    
    try:
        from sklearn.preprocessing import RobustScaler, StandardScaler
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.linear_model import Ridge, HuberRegressor
        from sklearn.ensemble import (
            RandomForestRegressor, 
            ExtraTreesRegressor, 
            HistGradientBoostingRegressor,
            GradientBoostingRegressor
        )
        import xgboost as xgb
        import lightgbm as lgb
        
        __main__.RobustScaler = RobustScaler
        __main__.StandardScaler = StandardScaler
        __main__.VarianceThreshold = VarianceThreshold
        __main__.Ridge = Ridge
        __main__.HuberRegressor = HuberRegressor
        __main__.RandomForestRegressor = RandomForestRegressor
        __main__.ExtraTreesRegressor = ExtraTreesRegressor
        __main__.HistGradientBoostingRegressor = HistGradientBoostingRegressor
        __main__.GradientBoostingRegressor = GradientBoostingRegressor
        __main__.XGBRegressor = xgb.XGBRegressor
        __main__.LGBMRegressor = lgb.LGBMRegressor
        
        try:
            from catboost import CatBoostRegressor
            __main__.CatBoostRegressor = CatBoostRegressor
        except ImportError:
            pass
            
    except Exception as e:
        print(f"Warning: Could not register ML classes: {e}")

# Call this BEFORE importing anything that loads models
register_pickle_classes()

# Now import our modules
from config.settings import DEVICE, TRAINED_MODELS_DIR, set_global_seed
from config.model_config import TARGET_CONFIG, TARGET_DISPLAY_NAMES, ALL_TARGETS
from config.conversion_config import (
    CONVERSION_CONFIG,
    convert_log_to_actual,
    get_display_name,
    get_unit,
)
from core.standardizer import MoleculeStandardizer
from core.engine import PredictionEngine
from ui.components import create_molecule_input
from ui.results import (
    display_results_table,
    create_download_button,
)
from utils.conversion import create_results_dataframe

# Set seed for reproducibility
set_global_seed()


def main():
    """Main application."""
    st.title("🧪 ADMET Property Predictor")
    st.markdown("""
    Predict ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) 
    properties for drug-like molecules using state-of-the-art machine learning models.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    # Show device info
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        st.sidebar.success(f"🚀 GPU: {gpu_name}")
    else:
        st.sidebar.warning("⚠️ Running on CPU")
    
    # Initialize engine
    engine = PredictionEngine()
    available_targets = engine.get_available_targets()
    
    if not available_targets:
        st.error(f"""
        ⚠️ No trained models found!
        
        Please ensure model files are in: `{TRAINED_MODELS_DIR}`
        """)
        return
    
    st.sidebar.success(f"✅ Found {len(available_targets)} available models")
    st.sidebar.caption(f"Device: {DEVICE}")
    
    # Target selection with units
    st.sidebar.subheader("Select Properties to Predict")
    
    target_options = {}
    for target in available_targets:
        config = CONVERSION_CONFIG.get(target, {})
        display_name = config.get('display_name', target)
        unit = config.get('unit', '')
        if unit:
            target_options[f"{display_name} ({unit})"] = target
        else:
            target_options[display_name] = target
    
    selected_display = st.sidebar.multiselect(
        "Properties",
        list(target_options.keys()),
        default=list(target_options.keys())[:3] if len(target_options) >= 3 else list(target_options.keys()),
    )
    
    selected_targets = [target_options[d] for d in selected_display]
    
    # Output options
    st.sidebar.subheader("📤 Output Options")
    include_log_values = st.sidebar.checkbox(
        "Include log-scale values",
        value=False,
        help="Include the original log-scale predictions in the output"
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        smiles_list = create_molecule_input()
    
    with col2:
        st.subheader("📋 Selected Properties")
        if selected_targets:
            for target in selected_targets:
                config = CONVERSION_CONFIG.get(target, {})
                display_name = config.get('display_name', target)
                unit = config.get('unit', '')
                if unit:
                    st.markdown(f"- **{display_name}** ({unit})")
                else:
                    st.markdown(f"- **{display_name}**")
        else:
            st.warning("Please select at least one property")
    
    # Predict button
    predict_disabled = not (smiles_list and selected_targets)
    
    if st.button("🚀 Predict", type="primary", disabled=predict_disabled):
        run_predictions(engine, smiles_list, selected_targets, include_log_values)
    
    # Display results if available
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']
        
        st.divider()
        display_results_table(results_df)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            create_download_button(results_df, "admet_predictions.csv")
        with col2:
            if st.button("🗑️ Clear Results"):
                del st.session_state['results_df']
                st.rerun()


def run_predictions(engine, smiles_list, selected_targets, include_log_values):
    """Run predictions and store results."""
    
    if not smiles_list:
        st.error("Please enter at least one SMILES")
        return
    
    if not selected_targets:
        st.error("Please select at least one property")
        return
    
    # Standardize SMILES
    standardizer = MoleculeStandardizer()
    
    with st.spinner("Standardizing molecules..."):
        clean_smiles = []
        valid_indices = []
        invalid_smiles = []
        
        for i, smi in enumerate(smiles_list):
            clean = standardizer.standardize_smiles(smi)
            if clean is not None:
                clean_smiles.append(clean)
                valid_indices.append(i)
            else:
                invalid_smiles.append((i, smi))
        
        if invalid_smiles:
            st.warning(f"⚠️ {len(invalid_smiles)} molecules could not be parsed")
    
    if not clean_smiles:
        st.error("No valid molecules to predict")
        return
    
    st.info(f"Processing {len(clean_smiles)} valid molecules...")
    
    # Storage for predictions
    log_predictions = {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load and predict for each target
    for t_idx, target in enumerate(selected_targets):
        config = CONVERSION_CONFIG.get(target, {})
        display_name = config.get('display_name', target)
        
        status_text.text(f"Loading model for {display_name}...")
        
        if engine.load_predictor(target):
            status_text.text(f"Predicting {display_name}...")
            
            try:
                predictions = engine.predict(target, clean_smiles)
                log_predictions[target] = predictions
            except Exception as e:
                st.error(f"Error predicting {display_name}: {e}")
                log_predictions[target] = np.zeros(len(clean_smiles))
        else:
            st.warning(f"Could not load model for {display_name}")
            log_predictions[target] = np.zeros(len(clean_smiles))
        
        progress_bar.progress((t_idx + 1) / len(selected_targets))
    
    status_text.text("Converting to actual values...")
    
    # Build results DataFrame
    results_data = {'SMILES': smiles_list}
    
    for target in selected_targets:
        config = CONVERSION_CONFIG.get(target, {})
        display_name = config.get('display_name', target)
        unit = config.get('unit', '')
        log_scale = config.get('log_scale', False)
        multiplier = config.get('multiplier', 1)
        
        col_name = f"{display_name} ({unit})" if unit else display_name
        
        # Initialize with NaN
        results_data[col_name] = [np.nan] * len(smiles_list)
        if include_log_values and log_scale:
            results_data[f"{display_name} (log)"] = [np.nan] * len(smiles_list)
        
        # Fill in predictions
        log_values = log_predictions[target]
        
        for i, idx in enumerate(valid_indices):
            if i < len(log_values):
                log_val = log_values[i]
                
                # Convert to actual value
                if log_scale:
                    actual_val = (10 ** log_val) / multiplier
                else:
                    actual_val = log_val
                
                results_data[col_name][idx] = actual_val
                
                if include_log_values and log_scale:
                    results_data[f"{display_name} (log)"][idx] = log_val
    
    results_df = pd.DataFrame(results_data)
    st.session_state['results_df'] = results_df
    
    status_text.empty()
    progress_bar.empty()
    
    st.success(f"✅ Predictions complete for {len(clean_smiles)} molecules!")


if __name__ == "__main__":
    main()