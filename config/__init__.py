# config/__init__.py
"""Configuration package."""

from .settings import (
    APP_DIR,
    TRAINED_MODELS_DIR,
    MODEL_SAVE_DIR,
    DEVICE,
    GLOBAL_SEED,
    TRAINING_CONFIG,
    ADAPTIVE_CONFIG,
    PREDICTION_CONSTRAINTS,
    set_global_seed,
)

from .model_config import (
    TARGET_CONFIG,
    MULTITASK_CONFIG,
    CHECKPOINT_VERSION_CONFIG,
    MULTITASK_DESCRIPTOR_CONFIGS,
    ALL_TARGETS,
    TARGET_DISPLAY_NAMES,
    get_model_dir,
)

from .descriptor_config import (
    MPNN_INTEGRATED_CONFIGS,
    MPNN_REFINEMENT_DESCRIPTORS,
    EFFLUX_DESCRIPTOR_CONFIG,
)

from .conversion_config import (
    CONVERSION_CONFIG,
    DISPLAY_TO_LOG_NAME,
    LOG_TO_DISPLAY_NAME,
    convert_log_to_actual,
    convert_actual_to_log,
    get_unit,
    get_display_name,
    format_value_with_unit,
)

__all__ = [
    # Settings
    'APP_DIR',
    'TRAINED_MODELS_DIR',
    'MODEL_SAVE_DIR',
    'DEVICE',
    'GLOBAL_SEED',
    'TRAINING_CONFIG',
    'ADAPTIVE_CONFIG',
    'PREDICTION_CONSTRAINTS',
    'set_global_seed',
    # Model config
    'TARGET_CONFIG',
    'MULTITASK_CONFIG',
    'CHECKPOINT_VERSION_CONFIG',
    'MULTITASK_DESCRIPTOR_CONFIGS',
    'ALL_TARGETS',
    'TARGET_DISPLAY_NAMES',
    'get_model_dir',
    # Descriptor config
    'MPNN_INTEGRATED_CONFIGS',
    'MPNN_REFINEMENT_DESCRIPTORS',
    'EFFLUX_DESCRIPTOR_CONFIG',
    # Conversion config
    'CONVERSION_CONFIG',
    'DISPLAY_TO_LOG_NAME',
    'LOG_TO_DISPLAY_NAME',
    'convert_log_to_actual',
    'convert_actual_to_log',
    'get_unit',
    'get_display_name',
    'format_value_with_unit',
]