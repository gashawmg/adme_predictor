"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def standardizer():
    from core.standardizer import MoleculeStandardizer
    return MoleculeStandardizer()


@pytest.fixture
def sample_smiles():
    """Small set of known-good SMILES for smoke tests."""
    return [
        "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # caffeine
        "CCO",                           # ethanol
    ]
