# ui/components.py
"""Reusable UI components."""

import base64
import io

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

# ---------------------------------------------------------------------------
# Example molecules — well-known drugs covering a range of ADME profiles
# ---------------------------------------------------------------------------
EXAMPLE_SMILES = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("Caffeine", "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
    ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O"),
    ("Metformin", "CN(C)C(=N)NC(=N)N"),
]


def create_molecule_input() -> list[str]:
    """Create molecule input section and return list of SMILES."""
    st.subheader("Input Molecules")

    # Example molecules shortcut
    if st.button("Load Example Molecules", help="Load 5 well-known drugs as example input"):
        st.session_state["example_smiles_text"] = "\n".join(
            f"{smi}  # {name}" for name, smi in EXAMPLE_SMILES
        )

    input_method = st.radio("Input method", ["Enter SMILES", "Upload CSV"], horizontal=True)

    smiles_list = []

    if input_method == "Enter SMILES":
        # Pre-populate if example was loaded
        default_text = st.session_state.get("example_smiles_text", "")
        placeholder = (
            "CC(=O)Oc1ccccc1C(=O)O  # Aspirin\n"
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O  # Ibuprofen\n"
            "Cn1cnc2c1c(=O)n(c(=O)n2C)C  # Caffeine"
        )
        smiles_text = st.text_area(
            "Enter SMILES (one per line; text after # is ignored)",
            value=default_text,
            height=160,
            placeholder=placeholder,
        )
        if smiles_text.strip():
            raw_lines = [line.strip() for line in smiles_text.strip().splitlines() if line.strip()]
            # Strip inline comments
            smiles_list = [
                line.split("#")[0].strip() for line in raw_lines if line.split("#")[0].strip()
            ]

    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            import pandas as pd

            df = pd.read_csv(uploaded_file)

            # Auto-detect SMILES column
            smiles_col = None
            for col in ["SMILES", "smiles", "Smiles", "canonical_smiles"]:
                if col in df.columns:
                    smiles_col = col
                    break

            if smiles_col is None:
                st.warning(
                    f"No SMILES column auto-detected. "
                    f"Available columns: {', '.join(df.columns.tolist())}. "
                    "Please select the correct one below."
                )
                smiles_col = st.selectbox("Select SMILES column", df.columns)

            if smiles_col:
                smiles_list = df[smiles_col].dropna().astype(str).tolist()
                st.success(f"Loaded **{len(smiles_list)}** molecules from column `{smiles_col}`")
                st.dataframe(df[[smiles_col]].head(3), use_container_width=True, hide_index=True)

    # Batch size warning
    if len(smiles_list) > 50:
        st.info(
            f"Large batch detected ({len(smiles_list)} molecules). "
            "Prediction may take several minutes on CPU."
        )

    # Preview first valid molecule
    if smiles_list:
        for smi in smiles_list[:5]:
            img_data = smiles_to_image(smi)
            if img_data:
                st.caption(f"Preview: `{smi[:60]}{'...' if len(smi) > 60 else ''}`")
                st.image(img_data, width=160)
                break

    return smiles_list


def smiles_to_image(smiles: str, size: tuple = (200, 200)) -> str | None:
    """Convert SMILES to base64-encoded PNG."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer_opts = Draw.MolDrawOptions()
        drawer_opts.atomLabelFontSize = 0.4
        img = Draw.MolToImage(mol, size=size, options=drawer_opts)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
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
