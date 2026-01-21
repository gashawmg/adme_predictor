# config/model_config.py
"""Model configuration settings."""

import os
from .settings import TRAINED_MODELS_DIR


def get_model_dir(target_name: str) -> str:
    """Get the model directory for a target."""
    return os.path.join(TRAINED_MODELS_DIR, target_name)


# =============================================================================
# TARGET CONFIGURATION
# =============================================================================

TARGET_CONFIG = {
    'LogD': {
        'strategy': 'MPNN_INTEGRATED',
        'training_style': 'script2',
        'scaler_type': 'standard',
        'use_refinement': False,
    },
    'LogS': {
        'strategy': 'MULTITASK',
        'training_style': 'unified',
        'scaler_type': 'robust',
        'is_multitask': True,
        'multitask_group': 'MULTITASK_LOGD_LOGS',
        'use_legacy_descriptor_calc': True,
    },
    'Log_HLM_CLint': {
        'strategy': 'MPNN_INTEGRATED',
        'training_style': 'script2',
        'scaler_type': 'standard',
        'use_refinement': False,
    },
    'Log_MLM_CLint': {
        'strategy': 'MPNN_INTEGRATED',
        'training_style': 'hybrid_v5_integrated',
        'scaler_type': 'standard',
        'use_refinement': False,
    },
    'Log_Caco_Papp_AB': {
        'strategy': 'MPNN_INTEGRATED',
        'training_style': 'script2',
        'scaler_type': 'standard',
        'use_refinement': False,
    },
    'Log_Caco_ER': {
        'strategy': 'OPTIMIZED_CACO_ER',
        'training_style': 'optimized_caco_er',
    },
    # Single-task models for PPB and BPB
    'Log_Mouse_PPB': {
        'strategy': 'MPNN_REFINEMENT',
        'training_style': 'hybrid_v5',
        'scaler_type': 'standard',
        'use_refinement': True,
    },
    'Log_Mouse_BPB': {
        'strategy': 'MPNN_REFINEMENT',
        'training_style': 'script1',
        'scaler_type': 'robust',
        'use_refinement': True,
    },
    # Multitask model for MPB
    'Log_Mouse_MPB': {
        'strategy': 'MULTITASK',
        'training_style': 'script1',
        'scaler_type': 'robust',
        'is_multitask': True,
        'multitask_group': 'MULTITASK_MOUSE_BINDING',
        'use_legacy_descriptor_calc': True,
    },
}

# =============================================================================
# MULTITASK CONFIGURATION
# =============================================================================

MULTITASK_CONFIG = {
    'MULTITASK_LOGD_LOGS': {
        'targets': ['LogD', 'LogS'],
        'output_target': 'LogS',
        'model_folder': 'LogS',
        'training_style': 'unified',
        'scaler_type': 'robust',
        'use_legacy_descriptor_calc': True,
    },
    'MULTITASK_MOUSE_BINDING': {
        'targets': ['Log_Mouse_PPB', 'Log_Mouse_BPB', 'Log_Mouse_MPB'],
        'output_target': 'Log_Mouse_MPB',
        'model_folder': 'Log_Mouse_MPB',
        'training_style': 'script1',
        'scaler_type': 'robust',
        'use_first_checkpoint': True,
        'use_legacy_descriptor_calc': True,
    },
}

# =============================================================================
# CHECKPOINT VERSION CONFIG
# =============================================================================

CHECKPOINT_VERSION_CONFIG = {
    'LogD': 'latest',
    'LogS': 'latest',
    'Log_HLM_CLint': 'latest',
    'Log_MLM_CLint': 'latest',
    'Log_Caco_Papp_AB': 'latest',
    'Log_Caco_ER': 'latest',
    'Log_Mouse_PPB': 'latest',
    'Log_Mouse_BPB': 'latest',
    'Log_Mouse_MPB': 0,
    'MULTITASK_LOGD_LOGS': 'latest',
    'MULTITASK_MOUSE_BINDING': 0,
}

# =============================================================================
# ALL TARGETS & DISPLAY NAMES
# =============================================================================

ALL_TARGETS = [
    'LogD', 'LogS', 'Log_HLM_CLint', 'Log_MLM_CLint',
    'Log_Caco_Papp_AB', 'Log_Caco_ER',
    'Log_Mouse_PPB', 'Log_Mouse_BPB', 'Log_Mouse_MPB'
]

TARGET_DISPLAY_NAMES = {
    'LogD': 'LogD (Distribution Coefficient)',
    'LogS': 'LogS (Kinetic Solubility)',
    'Log_HLM_CLint': 'HLM CLint (Human Liver Microsomal)',
    'Log_MLM_CLint': 'MLM CLint (Mouse Liver Microsomal)',
    'Log_Caco_Papp_AB': 'Caco-2 Papp A→B (Permeability)',
    'Log_Caco_ER': 'Caco-2 ER (Efflux Ratio)',
    'Log_Mouse_PPB': 'Mouse PPB (Plasma Protein Binding)',
    'Log_Mouse_BPB': 'Mouse BPB (Brain Protein Binding)',
    'Log_Mouse_MPB': 'Mouse MPB (Muscle Protein Binding)',
}

# =============================================================================
# MULTITASK DESCRIPTOR CONFIGURATIONS
# =============================================================================

MULTITASK_DESCRIPTOR_CONFIGS = {
    'MULTITASK_LOGD_LOGS': [
        'logp', 'logd_proxies', 'tpsa', 'mw', 'HBD', 'HBA', 'rings',
        'aromatic_ring_count', 'rotatable', 'rotatable_fraction', 'charge',
        'ionization_proxies', 'fractionCSP3', 'MolMR', 'heavy_atom_count',
        'num_heteroatoms', 'peoe_vsa', 'slogp_vsa', 'labute_asa',
        'num_hetero_rings', 'five_membered_rings', 'six_membered_rings',
        'aromatic_proportion', 'num_aromatic_carbocycles', 'num_saturated_rings',
        'sp3_carbon_count', 'num_F', 'num_Cl', 'num_Br', 'halogen_fraction',
        'num_acidic_groups', 'num_basic_groups', 'polar_surface_ratio',
        'esol_logS', 'num_fused_aromatic_rings', 'balaban_j',
        'sp2_fraction', 'aromatic_density', 'bertz_ct',
    ],
    'MULTITASK_MOUSE_BINDING': [
        'logp', 'logd_proxies', 'tpsa', 'mw', 'HBD', 'HBA', 'rings',
        'aromatic_ring_count', 'rotatable', 'rotatable_fraction', 'charge',
        'ionization_proxies', 'fractionCSP3', 'MolMR', 'heavy_atom_count',
        'num_heteroatoms', 'peoe_vsa', 'slogp_vsa', 'labute_asa',
        'num_hetero_rings', 'five_membered_rings', 'six_membered_rings',
        'num_F', 'num_Cl', 'num_Br', 'num_S', 'halogen_fraction',
        'aromatic_proportion', 'num_acidic_groups', 'num_basic_groups',
        'num_aromatic_carbocycles', 'num_saturated_rings', 'sp3_carbon_count',
    ],
}