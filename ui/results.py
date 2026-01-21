# ui/results.py
"""Results display components."""

import streamlit as st
import pandas as pd
from typing import Dict, List
import io

from config.conversion_config import CONVERSION_CONFIG, get_display_name, get_unit


def display_results_table(results_df: pd.DataFrame, show_stats: bool = True):
    """Display results in an interactive table."""
    st.subheader("📊 Prediction Results")
    
    if show_stats:
        # Summary statistics for numeric columns
        st.markdown("**Summary Statistics:**")
        numeric_cols = results_df.select_dtypes(include=['float64', 'float32', 'int64']).columns
        
        if len(numeric_cols) > 0:
            summary_data = []
            for col in numeric_cols:
                values = results_df[col].dropna()
                if len(values) > 0:
                    summary_data.append({
                        'Property': col,
                        'Mean': values.mean(),
                        'Std': values.std(),
                        'Min': values.min(),
                        'Max': values.max(),
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df = summary_df.set_index('Property')
                
                # Format numbers
                st.dataframe(
                    summary_df.style.format("{:.4f}"),
                    use_container_width=True
                )
    
    st.markdown("**Full Results:**")
    st.dataframe(results_df, use_container_width=True, height=400)


def display_property_info():
    """Display information about all properties and their units."""
    st.subheader("📋 Property Information")
    
    rows = []
    for log_name, config in CONVERSION_CONFIG.items():
        rows.append({
            'Property': config['display_name'],
            'Unit': config['unit'] if config['unit'] else '-',
            'Scale': 'Log' if config['log_scale'] else 'Linear',
            'Description': config['description'],
        })
    
    info_df = pd.DataFrame(rows)
    st.dataframe(info_df, use_container_width=True, hide_index=True)


def create_download_button(results_df: pd.DataFrame, filename: str = "predictions.csv"):
    """Create a download button for results."""
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv_data,
        file_name=filename,
        mime="text/csv"
    )


def display_molecule_results(
    smiles: str,
    predictions: Dict[str, float],
    index: int,
    show_log: bool = False
):
    """Display predictions for a single molecule."""
    from config.conversion_config import convert_log_to_actual, format_value_with_unit
    
    st.markdown(f"**Molecule {index + 1}**")
    st.code(smiles, language=None)
    
    cols = st.columns(3)
    col_idx = 0
    
    for log_name, log_value in predictions.items():
        config = CONVERSION_CONFIG.get(log_name, {})
        display_name = config.get('display_name', log_name)
        unit = config.get('unit', '')
        
        with cols[col_idx % 3]:
            if config.get('log_scale', False):
                actual_value = convert_log_to_actual(log_value, log_name)
                
                if unit:
                    st.metric(
                        label=display_name,
                        value=f"{actual_value:.4f}",
                        help=f"Unit: {unit}" + (f" | Log value: {log_value:.4f}" if show_log else "")
                    )
                else:
                    st.metric(
                        label=display_name,
                        value=f"{actual_value:.4f}",
                        help=f"Log value: {log_value:.4f}" if show_log else None
                    )
            else:
                st.metric(label=display_name, value=f"{log_value:.4f}")
        
        col_idx += 1