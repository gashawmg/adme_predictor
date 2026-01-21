# ui/components.py
"""Reusable UI components."""

import streamlit as st
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import Draw
import io
import base64


def create_molecule_input() -> List[str]:
    """Create molecule input section and return list of SMILES."""
    st.subheader("📥 Input Molecules")
    
    input_method = st.radio(
        "Input Method",
        ["Enter SMILES", "Upload CSV"],
        horizontal=True
    )
    
    smiles_list = []
    
    if input_method == "Enter SMILES":
        smiles_text = st.text_area(
            "Enter SMILES (one per line)",
            height=150,
            placeholder="CCO\nCC(=O)O\nc1ccccc1"
        )
        if smiles_text.strip():
            smiles_list = [s.strip() for s in smiles_text.strip().split('\n') if s.strip()]
    
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            
            # Find SMILES column
            smiles_col = None
            for col in ['SMILES', 'smiles', 'Smiles', 'canonical_smiles']:
                if col in df.columns:
                    smiles_col = col
                    break
            
            if smiles_col is None:
                smiles_col = st.selectbox("Select SMILES column", df.columns)
            
            if smiles_col:
                smiles_list = df[smiles_col].dropna().tolist()
                st.success(f"Loaded {len(smiles_list)} molecules")
    
    return smiles_list


def smiles_to_image(smiles: str, size: tuple = (200, 200)) -> Optional[str]:
    """Convert SMILES to base64 encoded image."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception:
        return None


def display_molecule_card(smiles: str, predictions: dict, index: int):
    """Display a molecule card with structure and predictions."""
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            img_data = smiles_to_image(smiles)
            if img_data:
                st.image(img_data, caption=f"Molecule {index + 1}")
            else:
                st.warning("Could not render structure")
            st.caption(f"`{smiles[:50]}...`" if len(smiles) > 50 else f"`{smiles}`")
        
        with col2:
            st.markdown("**Predictions:**")
            for target, value in predictions.items():
                st.metric(target, f"{value:.3f}")
        
        st.divider()