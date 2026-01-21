# utils/conversion.py
"""Conversion utilities for log to actual values."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from config.conversion_config import (
    CONVERSION_CONFIG,
    convert_log_to_actual,
    get_display_name,
    get_unit,
)


def convert_predictions_to_actual(
    predictions: Dict[str, np.ndarray],
    include_log: bool = False
) -> Dict[str, np.ndarray]:
    """
    Convert all predictions from log scale to actual values.
    
    Args:
        predictions: Dictionary of {log_name: log_values}
        include_log: If True, also include the original log values
    
    Returns:
        Dictionary with converted values (and optionally original log values)
    """
    result = {}
    
    for log_name, log_values in predictions.items():
        config = CONVERSION_CONFIG.get(log_name, {})
        display_name = config.get('display_name', log_name)
        
        if config.get('log_scale', False):
            # Convert to actual values
            multiplier = config.get('multiplier', 1)
            actual_values = (10 ** log_values) / multiplier
            result[display_name] = actual_values
            
            if include_log:
                result[f"{display_name} (log)"] = log_values
        else:
            # Not log scale, keep as is
            result[display_name] = log_values
    
    return result


def create_results_dataframe(
    smiles_list: List[str],
    predictions: Dict[str, np.ndarray],
    include_log: bool = False,
    molecule_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a results DataFrame with proper column names and units.
    
    Args:
        smiles_list: List of SMILES strings
        predictions: Dictionary of {log_name: log_values}
        include_log: If True, include log-scale values
        molecule_names: Optional list of molecule names
    
    Returns:
        DataFrame with results
    """
    # Start with SMILES
    data = {'SMILES': smiles_list}
    
    # Add molecule names if provided
    if molecule_names is not None:
        data['Molecule Name'] = molecule_names
    
    # Convert and add predictions
    for log_name, log_values in predictions.items():
        config = CONVERSION_CONFIG.get(log_name, {})
        display_name = config.get('display_name', log_name)
        unit = config.get('unit', '')
        
        # Create column name with unit
        if unit:
            col_name = f"{display_name} ({unit})"
        else:
            col_name = display_name
        
        if config.get('log_scale', False):
            # Convert to actual values
            multiplier = config.get('multiplier', 1)
            actual_values = (10 ** log_values) / multiplier
            data[col_name] = actual_values
            
            if include_log:
                data[f"{log_name} (log)"] = log_values
        else:
            # Not log scale, keep as is
            data[col_name] = log_values
    
    return pd.DataFrame(data)


def get_column_info() -> pd.DataFrame:
    """
    Get information about all columns for display.
    
    Returns:
        DataFrame with column information
    """
    rows = []
    for log_name, config in CONVERSION_CONFIG.items():
        rows.append({
            'Log Name': log_name,
            'Display Name': config['display_name'],
            'Unit': config['unit'] if config['unit'] else '-',
            'Log Scale': 'Yes' if config['log_scale'] else 'No',
            'Description': config['description'],
        })
    
    return pd.DataFrame(rows)