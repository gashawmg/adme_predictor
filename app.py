# app.py
"""ADME Predictor — Streamlit Application."""

import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import torch

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Page config — must be first Streamlit command
st.set_page_config(
    page_title="ADME Predictor | OpenADMET Challenge",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/gashawmg/adme_predictor/issues",
        "About": (
            "ADME Predictor — 14th place out of 370+ participants in the "
            "ExpansionRx-OpenADMET Blind Challenge."
        ),
    },
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
            LegacyMPNNDescriptorCalculator,
            MPNNDescriptorCalculator,
            MPNNRefinementDescriptorCalculator,
        )

        __main__.MPNNDescriptorCalculator = MPNNDescriptorCalculator
        __main__.LegacyMPNNDescriptorCalculator = LegacyMPNNDescriptorCalculator
        __main__.MPNNRefinementDescriptorCalculator = MPNNRefinementDescriptorCalculator
    except Exception as e:
        print(f"Warning: Could not register descriptor classes: {e}")

    try:
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            HistGradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.linear_model import HuberRegressor, Ridge
        from sklearn.preprocessing import RobustScaler, StandardScaler

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


# Call BEFORE importing anything that loads models
register_pickle_classes()

# Now import our modules
from config.conversion_config import (  # noqa: E402
    CONVERSION_CONFIG,
    PROPERTY_GROUPS,
)
from config.settings import DEVICE, TRAINED_MODELS_DIR, set_global_seed  # noqa: E402
from core.engine import PredictionEngine  # noqa: E402
from core.standardizer import MoleculeStandardizer  # noqa: E402
from ui.components import create_molecule_input  # noqa: E402
from ui.results import (  # noqa: E402
    create_download_button,
    display_property_info,
    display_results_table,
)

set_global_seed()


# =============================================================================
# Cached resource — loaded once per server process, not on every widget click
# =============================================================================
@st.cache_resource(show_spinner="Loading models...")
def get_engine() -> PredictionEngine:
    return PredictionEngine()


# =============================================================================
# Hero header
# =============================================================================
def render_header():
    col_title, col_badges = st.columns([3, 2])
    with col_title:
        st.title("ADME Property Predictor")
        st.markdown(
            "Predict 9 ADME properties for drug-like molecules using ensemble "
            "MPNN models trained on the **ExpansionRx-OpenADMET Blind Challenge** dataset."
        )
    with col_badges:
        st.markdown(
            """
            <div style='padding-top:1.2rem'>
            <span style='background:#1f77b4;color:white;padding:4px 10px;
                border-radius:12px;font-size:0.8em;margin-right:6px'>
                OpenADMET Challenge</span>
            <a href='https://openadmet.ghost.io/the-openadmet-expansionrx-blind-challenge-has-come-to-an-end/'
               target='_blank' rel='noopener noreferrer'
               style='background:#2ca02c;color:white;padding:4px 10px;
                border-radius:12px;font-size:0.8em;margin-right:6px;
                text-decoration:none'>
                14th / 370+ Teams</a>
            <span style='background:#9467bd;color:white;padding:4px 10px;
                border-radius:12px;font-size:0.8em'>
                9 ADME Properties</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("About this tool", expanded=False):
        m1, m2, m3 = st.columns(3)
        m1.metric("Challenge Rank", "14th")
        m2.metric("Participants", "370+")
        m3.metric("Properties Predicted", "9")

        st.markdown(
            """
**What are ADME properties?**
ADME stands for Absorption, Distribution, Metabolism, and Excretion —
the key pharmacokinetic properties that determine whether a drug candidate is
likely to be safe and effective in the body. Early ADME prediction helps
prioritise compounds during drug discovery and reduce costly late-stage attrition.

**How were the models built?**
All models were trained exclusively on the official ExpansionRx-OpenADMET
challenge training data — no external datasets were used. The architecture
combines **Message Passing Neural Networks (MPNN)** via ChemProp 2.2.1 with 40+
hand-crafted RDKit molecular descriptors in a hybrid ensemble.
Five-fold cross-validation ensembles are used for all targets to reduce
prediction variance. Caco-2 Efflux Ratio uses a gradient-boosting ensemble
(RF + XGBoost + LightGBM + CatBoost).

**Limitations:** Predictions are most reliable for drug-like small molecules
similar to the training set. Results are for research use only and do not
constitute medical or regulatory advice.
            """
        )
        st.divider()
        st.markdown("**Predicted properties and reference ranges:**")
        display_property_info()


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar(available_targets: list[str]) -> tuple[list[str], bool]:
    st.sidebar.header("Configuration")

    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        st.sidebar.success(f"GPU: {gpu_name}")
    else:
        st.sidebar.warning("Running on CPU")
    st.sidebar.caption(f"Device: {DEVICE} | Models: {len(available_targets)}")

    st.sidebar.divider()
    st.sidebar.subheader("Select Properties")

    if "selected_targets" not in st.session_state:
        st.session_state["selected_targets"] = available_targets[:3]

    # Build label -> log_name map
    target_options = {}
    for log_name in available_targets:
        cfg = CONVERSION_CONFIG.get(log_name, {})
        display = cfg.get("display_name", log_name)
        unit = cfg.get("unit", "")
        label = f"{display} ({unit})" if unit else display
        target_options[label] = log_name

    # Grouped multiselects per ADME category
    selected_display = []
    for group, members in PROPERTY_GROUPS.items():
        group_labels = [
            (
                f"{CONVERSION_CONFIG[t]['display_name']} ({CONVERSION_CONFIG[t]['unit']})"
                if CONVERSION_CONFIG[t]["unit"]
                else CONVERSION_CONFIG[t]["display_name"]
            )
            for t in members
            if t in available_targets
        ]
        if not group_labels:
            continue
        default_group = [
            lbl
            for lbl in group_labels
            if target_options.get(lbl) in st.session_state["selected_targets"]
        ]
        chosen = st.sidebar.multiselect(
            group, group_labels, default=default_group, key=f"group_{group}"
        )
        selected_display.extend(chosen)

    selected_targets = [target_options[d] for d in selected_display if d in target_options]
    st.session_state["selected_targets"] = selected_targets

    st.sidebar.divider()
    st.sidebar.subheader("Output Options")
    include_log_values = st.sidebar.checkbox(
        "Include log-scale values",
        value=False,
        help="Add raw log-scale predictions alongside converted values in the CSV/Excel output",
    )

    return selected_targets, include_log_values


# =============================================================================
# Prediction runner
# =============================================================================
def run_predictions(engine, smiles_list, selected_targets, include_log_values):
    if not smiles_list:
        st.error("Please enter at least one SMILES string.")
        return
    if not selected_targets:
        st.error("Please select at least one property from the sidebar.")
        return

    standardizer = MoleculeStandardizer()
    clean_smiles, valid_indices, invalid_smiles = [], [], []

    for i, smi in enumerate(smiles_list):
        clean = standardizer.standardize_smiles(smi)
        if clean is not None:
            clean_smiles.append(clean)
            valid_indices.append(i)
        else:
            invalid_smiles.append((i + 1, smi))

    if invalid_smiles:
        with st.expander(f"Show {len(invalid_smiles)} invalid SMILES", expanded=False):
            for row, smi in invalid_smiles:
                st.markdown(f"- Row {row}: `{smi}`")
        st.warning(f"{len(invalid_smiles)} molecule(s) could not be parsed and were skipped.")

    if not clean_smiles:
        st.error("No valid molecules to predict.")
        return

    log_predictions = {}

    n = len(selected_targets)
    label = (
        f"Predicting {n} propert{'y' if n == 1 else 'ies'} for {len(clean_smiles)} molecule(s)..."
    )
    with st.status(label, expanded=True) as status:
        for target in selected_targets:
            cfg = CONVERSION_CONFIG.get(target, {})
            display_name = cfg.get("display_name", target)

            st.write(f"Loading model: **{display_name}**")
            if engine.load_predictor(target):
                st.write(f"Predicting: **{display_name}**")
                try:
                    predictions = engine.predict(target, clean_smiles)
                    log_predictions[target] = predictions
                except Exception as e:
                    st.error(f"Error predicting {display_name}: {e}")
                    log_predictions[target] = np.zeros(len(clean_smiles))
            else:
                st.warning(f"Could not load model for {display_name}.")
                log_predictions[target] = np.zeros(len(clean_smiles))

        status.update(
            label=f"Complete — {len(clean_smiles)} molecule(s) predicted.",
            state="complete",
            expanded=False,
        )

    # Build results DataFrame
    results_data = {"SMILES": smiles_list}
    for target in selected_targets:
        cfg = CONVERSION_CONFIG.get(target, {})
        display_name = cfg.get("display_name", target)
        unit = cfg.get("unit", "")
        log_scale = cfg.get("log_scale", False)
        multiplier = cfg.get("multiplier", 1)
        col_name = f"{display_name} ({unit})" if unit else display_name

        results_data[col_name] = [np.nan] * len(smiles_list)
        if include_log_values and log_scale:
            results_data[f"{display_name} (log)"] = [np.nan] * len(smiles_list)

        log_values = log_predictions[target]
        for i, idx in enumerate(valid_indices):
            if i < len(log_values):
                log_val = log_values[i]
                actual_val = (10**log_val) / multiplier if log_scale else log_val
                results_data[col_name][idx] = actual_val
                if include_log_values and log_scale:
                    results_data[f"{display_name} (log)"][idx] = log_val

    st.session_state["results_df"] = pd.DataFrame(results_data)
    st.success(f"Predictions complete for {len(clean_smiles)} molecule(s).")


# =============================================================================
# Main
# =============================================================================
def main():
    render_header()

    engine = get_engine()
    available_targets = engine.get_available_targets()

    if not available_targets:
        st.error(f"No trained models found in `{TRAINED_MODELS_DIR}`. Please check setup.")
        return

    selected_targets, include_log_values = render_sidebar(available_targets)

    st.divider()

    col_input, _ = st.columns([3, 1])
    with col_input:
        smiles_list = create_molecule_input()

    if smiles_list:
        st.caption(f"{len(smiles_list)} molecule(s) ready for prediction.")

    if st.button(
        "Predict ADME Properties",
        type="primary",
        disabled=not (smiles_list and selected_targets),
    ):
        run_predictions(engine, smiles_list, selected_targets, include_log_values)

    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        st.divider()
        display_results_table(results_df)

        col_dl, col_clear, _ = st.columns([2, 1, 3])
        with col_dl:
            create_download_button(results_df, "admet_predictions.csv")
        with col_clear:
            if st.button("Clear Results", use_container_width=True):
                del st.session_state["results_df"]
                st.rerun()

    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:gray;font-size:0.78em'>"
        "Built for the <b>ExpansionRx-OpenADMET Blind Challenge</b> &nbsp;|&nbsp; "
        "<a href='https://github.com/gashawmg/adme_predictor'>GitHub</a> &nbsp;|&nbsp; "
        "<a href='https://admepredictor.streamlit.app/'>Live Demo</a> &nbsp;|&nbsp; "
        "MIT License &nbsp;|&nbsp; "
        "Predictions are for <b>research use only</b> and do not constitute "
        "medical or regulatory advice."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
