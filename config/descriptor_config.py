# config/descriptor_config.py
"""Descriptor configuration settings."""

# =============================================================================
# MPNN INTEGRATED DESCRIPTOR CONFIGS (per target)
# =============================================================================

MPNN_INTEGRATED_CONFIGS = {
    'LogD': [
        'logp', 'logd_proxies', 'tpsa', 'mw', 'HBD', 'HBA', 'rings',
        'aromatic_ring_count', 'rotatable', 'rotatable_fraction', 'charge',
        'ionization_proxies', 'fractionCSP3', 'MolMR', 'heavy_atom_count',
        'num_heteroatoms', 'peoe_vsa', 'slogp_vsa', 'labute_asa',
        'num_hetero_rings', 'five_membered_rings', 'six_membered_rings',
        'num_F', 'num_Cl', 'num_Br', 'num_S', 'halogen_fraction',
        'aromatic_proportion', 'num_acidic_groups', 'num_basic_groups',
    ],
    'LogS': [
        'logp', 'tpsa', 'mw', 'HBD', 'HBA', 'rings',
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
    'Log_HLM_CLint': [
        'logp', 'tpsa', 'mw', 'HBD', 'HBA', 'rings',
        'aromatic_ring_count', 'rotatable', 'rotatable_fraction', 'charge',
        'ionization_proxies', 'fractionCSP3', 'MolMR', 'heavy_atom_count',
        'num_heteroatoms', 'peoe_vsa', 'slogp_vsa', 'labute_asa',
        'num_allylic_sites', 'num_benzylic_sites', 'num_ester_groups',
        'num_amide_groups', 'cyp_substrate_alerts', 'aromatic_proportion',
        'num_F', 'num_Cl', 'halogen_fraction', 'num_tertiary_amine',
        'num_secondary_amine', 'oxidation_susceptibility',
    ],
    'Log_MLM_CLint': [
        'logp', 'tpsa', 'mw', 'HBD', 'HBA', 'rings',
        'aromatic_ring_count', 'rotatable', 'rotatable_fraction', 'charge',
        'ionization_proxies', 'fractionCSP3', 'MolMR', 'heavy_atom_count',
        'num_heteroatoms', 'peoe_vsa', 'slogp_vsa', 'labute_asa',
        'num_allylic_sites', 'num_benzylic_sites', 'num_ester_groups',
        'num_amide_groups', 'cyp_substrate_alerts', 'aromatic_proportion',
        'num_F', 'num_Cl', 'halogen_fraction', 'num_tertiary_amine',
        'num_secondary_amine', 'oxidation_susceptibility',
    ],
    'Log_Caco_Papp_AB': [
        'logp', 'logd_proxies', 'tpsa', 'mw', 'HBD', 'HBA', 'rings',
        'aromatic_ring_count', 'rotatable', 'rotatable_fraction', 'charge',
        'ionization_proxies', 'fractionCSP3', 'MolMR', 'heavy_atom_count',
        'num_heteroatoms', 'peoe_vsa', 'slogp_vsa', 'labute_asa',
        'num_hetero_rings', 'five_membered_rings', 'six_membered_rings',
        'num_F', 'num_Cl', 'num_Br', 'num_S', 'halogen_fraction',
        'aromatic_proportion', 'num_acidic_groups', 'num_basic_groups',
    ],
    'Log_Mouse_PPB': [
        'logp', 'tpsa', 'mw', 'HBD', 'HBA', 'rings',
        'aromatic_ring_count', 'rotatable', 'fractionCSP3', 'MolMR',
        'heavy_atom_count', 'num_heteroatoms', 'charge',
        'num_acidic_groups', 'num_basic_groups', 'num_tertiary_amine',
        'hydrophobic_SA', 'hydrophilic_SA',
        'aromatic_carbon_count', 'aromatic_proportion',
        'albumin_binding_score', 'agp_binding_score', 'cad_score',
        'protein_binding_proxy', 'longest_alkyl_chain',
        'aromatic_halogen_count', 'num_quaternary_N',
        'Chi0v', 'Chi1v', 'Chi2v',
        'Kappa1', 'Kappa2', 'Kappa3', 'HallKierAlpha',
        'NPR1', 'NPR2', 'Asphericity',
        'MaxPartialCharge', 'MinPartialCharge',
        'fused_ring_count', 'LE_proxy',
    ],
    'Log_Mouse_BPB': [
        'logp', 'tpsa', 'mw', 'HBD', 'HBA', 'rings',
        'aromatic_ring_count', 'rotatable', 'fractionCSP3', 'MolMR',
        'heavy_atom_count', 'num_heteroatoms', 'charge',
        'num_acidic_groups', 'num_basic_groups', 'num_tertiary_amine',
        'hydrophobic_SA', 'hydrophilic_SA',
        'aromatic_carbon_count', 'aromatic_proportion',
        'albumin_binding_score', 'agp_binding_score', 'cad_score',
        'protein_binding_proxy', 'longest_alkyl_chain',
        'aromatic_halogen_count', 'num_quaternary_N',
        'Chi0v', 'Chi1v', 'Chi2v',
        'Kappa1', 'Kappa2', 'Kappa3', 'HallKierAlpha',
        'NPR1', 'NPR2', 'Asphericity',
        'MaxPartialCharge', 'MinPartialCharge',
        'fused_ring_count', 'LE_proxy',
    ],
}

# =============================================================================
# MPNN REFINEMENT DESCRIPTORS
# =============================================================================

MPNN_REFINEMENT_DESCRIPTORS = [
    'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'LabuteASA',
    'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
    'NumHeteroatoms', 'NumAromaticRings', 'FractionCSP3',
    'RingCount', 'HeavyAtomCount',
    'NumAliphaticRings', 'NumSaturatedRings',
    'NumAromaticHeterocycles', 'NumAliphaticHeterocycles',
    'NHOHCount', 'NOCount',
    'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v',
    'Kappa1', 'Kappa2', 'Kappa3', 'HallKierAlpha',
    'BertzCT',
]

# =============================================================================
# EFFLUX DESCRIPTOR CONFIG (for Caco-ER)
# =============================================================================

EFFLUX_DESCRIPTOR_CONFIG = {
    'fps': ['atompair', 'fcfp4', 'ecfp4', 'maccs'],
    'desc': [
        'tpsa', 'mw', 'logp', 'HBD', 'HBA', 'rings', 'charge',
        'ionization_proxies', 'pgp_alerts', 'tertiary_amine',
        'quaternary_ammonium', 'polar_surface_components',
        'efflux_features', 'aromatic_rings', 'rotatable_bonds',
        'molecular_flexibility', 'amphiphilic_moment',
    ]
}