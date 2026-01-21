# config/conversion_config.py
"""Conversion configuration for log to actual values."""

# Conversion configuration
# Format: log_name -> (display_name, unit, log_scale, multiplier, description)
CONVERSION_CONFIG = {
    'LogD': {
        'display_name': 'LogD',
        'unit': '',
        'log_scale': False,
        'multiplier': 1,
        'description': 'Distribution Coefficient (LogD)',
    },
    'LogS': {
        'display_name': 'KSol',
        'unit': 'µM',
        'log_scale': True,
        'multiplier': 1e-6,
        'description': 'Kinetic Solubility',
    },
    'Log_HLM_CLint': {
        'display_name': 'HLM CLint',
        'unit': 'mL/min/kg',
        'log_scale': True,
        'multiplier': 1,
        'description': 'Human Liver Microsomal Clearance',
    },
    'Log_MLM_CLint': {
        'display_name': 'MLM CLint',
        'unit': 'mL/min/kg',
        'log_scale': True,
        'multiplier': 1,
        'description': 'Mouse Liver Microsomal Clearance',
    },
    'Log_Caco_Papp_AB': {
        'display_name': 'Caco-2 Papp A>B',
        'unit': '10⁻⁶ cm/s',
        'log_scale': True,
        'multiplier': 1e-6,
        'description': 'Caco-2 Permeability Papp A→B',
    },
    'Log_Caco_ER': {
        'display_name': 'Caco-2 Efflux',
        'unit': '',
        'log_scale': True,
        'multiplier': 1,
        'description': 'Caco-2 Permeability Efflux Ratio',
    },
    'Log_Mouse_PPB': {
        'display_name': 'MPPB',
        'unit': '% Unbound',
        'log_scale': True,
        'multiplier': 1,
        'description': 'Mouse Plasma Protein Binding',
    },
    'Log_Mouse_BPB': {
        'display_name': 'MBPB',
        'unit': '% Unbound',
        'log_scale': True,
        'multiplier': 1,
        'description': 'Mouse Brain Protein Binding',
    },
    'Log_Mouse_MPB': {
        'display_name': 'MGMB',
        'unit': '% Unbound',
        'log_scale': True,
        'multiplier': 1,
        'description': 'Mouse Gastrocnemius Muscle Binding',
    },
}

# Reverse mapping: display_name -> log_name
DISPLAY_TO_LOG_NAME = {v['display_name']: k for k, v in CONVERSION_CONFIG.items()}

# Log name to display name
LOG_TO_DISPLAY_NAME = {k: v['display_name'] for k, v in CONVERSION_CONFIG.items()}


def convert_log_to_actual(log_value: float, log_name: str) -> float:
    """
    Convert a log-scale value to actual value.
    
    Formula: actual = (10^log_value) / multiplier
    
    Args:
        log_value: The predicted log-scale value
        log_name: The name of the property (e.g., 'LogS', 'Log_HLM_CLint')
    
    Returns:
        The actual value in the appropriate units
    """
    if log_name not in CONVERSION_CONFIG:
        return log_value
    
    config = CONVERSION_CONFIG[log_name]
    
    if not config['log_scale']:
        # Not log scale, return as is
        return log_value
    
    # Convert: actual = 10^log_value / multiplier
    multiplier = config['multiplier']
    actual_value = (10 ** log_value) / multiplier
    
    return actual_value


def convert_actual_to_log(actual_value: float, log_name: str) -> float:
    """
    Convert an actual value back to log-scale.
    
    Formula: log_value = log10(actual * multiplier)
    
    Args:
        actual_value: The actual value in appropriate units
        log_name: The name of the property
    
    Returns:
        The log-scale value
    """
    import numpy as np
    
    if log_name not in CONVERSION_CONFIG:
        return actual_value
    
    config = CONVERSION_CONFIG[log_name]
    
    if not config['log_scale']:
        return actual_value
    
    multiplier = config['multiplier']
    log_value = np.log10(actual_value * multiplier)
    
    return log_value


def get_unit(log_name: str) -> str:
    """Get the unit for a property."""
    if log_name in CONVERSION_CONFIG:
        return CONVERSION_CONFIG[log_name]['unit']
    return ''


def get_display_name(log_name: str) -> str:
    """Get the display name for a property."""
    if log_name in CONVERSION_CONFIG:
        return CONVERSION_CONFIG[log_name]['display_name']
    return log_name


def format_value_with_unit(value: float, log_name: str, decimals: int = 3) -> str:
    """Format a value with its unit."""
    unit = get_unit(log_name)
    if unit:
        return f"{value:.{decimals}f} {unit}"
    return f"{value:.{decimals}f}"