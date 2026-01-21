# core/standardizer.py
"""Molecule standardization utilities."""

from rdkit import Chem
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser, Uncharger


class MoleculeStandardizer:
    """Standardize molecules for consistent processing."""
    
    def __init__(self):
        self.lfc = LargestFragmentChooser()
        self.uc = Uncharger()

    def sanitize(self, smiles: str):
        """Sanitize a SMILES string and return standardized molecule."""
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return None
            mol = self.lfc.choose(mol)
            mol = self.uc.uncharge(mol)
            return mol
        except Exception:
            return None
    
    def standardize_smiles(self, smiles: str) -> str:
        """Return standardized canonical SMILES."""
        mol = self.sanitize(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)