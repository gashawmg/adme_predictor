# core/descriptors.py
"""Molecular descriptor calculators."""

import numpy as np
from typing import List, Dict
from sklearn.preprocessing import RobustScaler, StandardScaler

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors as rdMD, Lipinski, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator

from config.descriptor_config import MPNN_REFINEMENT_DESCRIPTORS


class MPNNDescriptorCalculator:
    """Calculate molecular descriptors for MPNN integration."""
    
    def __init__(self, descriptor_list: List[str], scaler_type: str = 'standard'):
        self.descriptor_list = list(dict.fromkeys(descriptor_list))
        self.scaler_type = scaler_type
        if scaler_type == 'robust':
            self.scaler = RobustScaler(quantile_range=(5.0, 95.0))
        else:
            self.scaler = StandardScaler()

    def _count_pattern(self, mol, smarts: str) -> int:
        """Count SMARTS pattern matches."""
        try:
            pattern = Chem.MolFromSmarts(smarts)
            return len(mol.GetSubstructMatches(pattern)) if pattern else 0
        except Exception:
            return 0

    def _get_longest_alkyl_chain(self, mol) -> int:
        """Get the longest alkyl chain length."""
        try:
            max_chain = 0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6 and atom.GetHybridization().name == 'SP3':
                    chain_length = self._trace_alkyl_chain(mol, atom.GetIdx(), set())
                    max_chain = max(max_chain, chain_length)
            return max_chain
        except Exception:
            return 0

    def _trace_alkyl_chain(self, mol, atom_idx, visited) -> int:
        """Trace an alkyl chain from a starting atom."""
        if atom_idx in visited:
            return 0
        visited.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() != 6 or atom.GetHybridization().name != 'SP3':
            return 0
        max_length = 1
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 6 and neighbor.GetHybridization().name == 'SP3':
                length = 1 + self._trace_alkyl_chain(mol, neighbor.GetIdx(), visited.copy())
                max_length = max(max_length, length)
        return max_length

    def _count_fused_rings(self, mol) -> int:
        """Count fused ring systems."""
        try:
            ring_info = mol.GetRingInfo()
            rings = ring_info.AtomRings()
            if len(rings) < 2:
                return 0
            fused_count = 0
            for i, ring1 in enumerate(rings):
                for ring2 in rings[i+1:]:
                    if len(set(ring1) & set(ring2)) >= 2:
                        fused_count += 1
            return fused_count
        except Exception:
            return 0

    def _calculate_single(self, mol) -> Dict[str, float]:
        """Calculate all descriptors for a single molecule."""
        if mol is None:
            return {}
        
        desc = {}
        try:
            # Core descriptors
            desc['mw'] = Descriptors.MolWt(mol)
            desc['logp'] = Descriptors.MolLogP(mol)
            desc['tpsa'] = Descriptors.TPSA(mol)
            desc['HBD'] = Descriptors.NumHDonors(mol)
            desc['HBA'] = Descriptors.NumHAcceptors(mol)
            desc['rotatable'] = Descriptors.NumRotatableBonds(mol)
            desc['rings'] = Descriptors.RingCount(mol)
            desc['aromatic_ring_count'] = rdMD.CalcNumAromaticRings(mol)
            desc['fractionCSP3'] = Descriptors.FractionCSP3(mol)
            desc['MolMR'] = Descriptors.MolMR(mol)
            desc['heavy_atom_count'] = Descriptors.HeavyAtomCount(mol)
            desc['num_heteroatoms'] = Descriptors.NumHeteroatoms(mol)
            
            n_bonds = mol.GetNumBonds()
            desc['rotatable_fraction'] = desc['rotatable'] / n_bonds if n_bonds > 0 else 0
            desc['charge'] = Chem.GetFormalCharge(mol)
            
            # Ionization
            desc['num_basic_N'] = self._count_pattern(mol, '[N;H2,H1;!$(NC=O)]')
            desc['num_acidic_O'] = self._count_pattern(mol, '[OX2H][CX3](=[OX1])')
            desc['ionization_proxies'] = desc['num_basic_N'] + desc['num_acidic_O']
            desc['num_acidic_groups'] = self._count_pattern(mol, '[CX3](=O)[OX1H0-,OX2H1]')
            desc['num_basic_groups'] = self._count_pattern(mol, '[NX3;H2,H1;!$(NC=O)]')
            
            # Halogens
            desc['num_F'] = self._count_pattern(mol, '[F]')
            desc['num_Cl'] = self._count_pattern(mol, '[Cl]')
            desc['num_Br'] = self._count_pattern(mol, '[Br]')
            desc['num_S'] = self._count_pattern(mol, '[#16]')
            total_halogens = desc['num_F'] + desc['num_Cl'] + desc['num_Br']
            desc['halogen_fraction'] = total_halogens / desc['heavy_atom_count'] if desc['heavy_atom_count'] > 0 else 0
            
            # Rings
            ring_info = mol.GetRingInfo()
            ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
            desc['five_membered_rings'] = sum(1 for s in ring_sizes if s == 5)
            desc['six_membered_rings'] = sum(1 for s in ring_sizes if s == 6)
            desc['num_hetero_rings'] = rdMD.CalcNumHeterocycles(mol)
            desc['max_ring_size'] = max(ring_sizes) if ring_sizes else 0
            
            # Surface areas
            desc['peoe_vsa'] = sum(rdMD.PEOE_VSA_(mol))
            desc['slogp_vsa'] = sum(rdMD.SlogP_VSA_(mol))
            desc['labute_asa'] = rdMD.CalcLabuteASA(mol)
            
            # Derived
            desc['logd_proxies'] = desc['logp'] - 0.5 * desc['num_basic_N'] + 0.3 * desc['num_acidic_O']
            
            num_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            desc['aromatic_proportion'] = num_aromatic_atoms / desc['heavy_atom_count'] if desc['heavy_atom_count'] > 0 else 0
            desc['num_aromatic_carbocycles'] = rdMD.CalcNumAromaticCarbocycles(mol)
            desc['num_saturated_rings'] = rdMD.CalcNumSaturatedRings(mol)
            
            sp3_carbons = sum(1 for atom in mol.GetAtoms() 
                           if atom.GetAtomicNum() == 6 and atom.GetHybridization().name == 'SP3')
            desc['sp3_carbon_count'] = sp3_carbons
            desc['polar_surface_ratio'] = desc['tpsa'] / desc['labute_asa'] if desc['labute_asa'] > 0 else 0
            
            # Metabolism
            desc['num_allylic_sites'] = self._count_pattern(mol, '[CH2;!R][CH]=[CH]')
            desc['num_benzylic_sites'] = self._count_pattern(mol, '[cH1]~[CH2]')
            desc['num_ester_groups'] = self._count_pattern(mol, '[#6][CX3](=O)[OX2][#6]')
            desc['num_amide_groups'] = self._count_pattern(mol, '[NX3][CX3](=[OX1])[#6]')
            desc['num_tertiary_amine'] = self._count_pattern(mol, '[NX3;H0;!$(NC=O)]')
            desc['num_secondary_amine'] = self._count_pattern(mol, '[NX3;H1;!$(NC=O)]')
            
            cyp_alerts = 0
            if desc['mw'] > 500: cyp_alerts += 1
            if desc['logp'] > 4: cyp_alerts += 1
            if desc['aromatic_ring_count'] > 3: cyp_alerts += 1
            desc['cyp_substrate_alerts'] = cyp_alerts
            desc['oxidation_susceptibility'] = (desc['num_allylic_sites'] + 
                                                desc['num_benzylic_sites'] + 
                                                desc['num_tertiary_amine'] * 0.5)
            
            # Connectivity indices
            desc['Chi0v'] = Descriptors.Chi0v(mol)
            desc['Chi1v'] = Descriptors.Chi1v(mol)
            desc['Chi2v'] = Descriptors.Chi2v(mol)
            desc['Chi3v'] = Descriptors.Chi3v(mol)
            desc['Chi4v'] = Descriptors.Chi4v(mol)
            desc['Kappa1'] = Descriptors.Kappa1(mol)
            desc['Kappa2'] = Descriptors.Kappa2(mol)
            desc['Kappa3'] = Descriptors.Kappa3(mol)
            desc['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
            
            # 3D-like descriptors
            try:
                desc['NPR1'] = rdMD.CalcNPR1(mol)
                desc['NPR2'] = rdMD.CalcNPR2(mol)
                desc['Asphericity'] = rdMD.CalcAsphericity(mol)
            except Exception:
                desc['NPR1'] = desc['NPR2'] = desc['Asphericity'] = 0.0
            
            # Partial charges
            try:
                desc['MaxPartialCharge'] = Descriptors.MaxPartialCharge(mol)
                desc['MinPartialCharge'] = Descriptors.MinPartialCharge(mol)
            except Exception:
                desc['MaxPartialCharge'] = desc['MinPartialCharge'] = 0.0
            
            desc['num_quaternary_N'] = self._count_pattern(mol, '[NX4+]')
            
            # Surface area partitioning
            try:
                slogp_vsa_vals = rdMD.SlogP_VSA_(mol)
                desc['hydrophobic_SA'] = sum(slogp_vsa_vals[5:])
                desc['hydrophilic_SA'] = sum(slogp_vsa_vals[:5])
            except Exception:
                desc['hydrophobic_SA'] = desc['hydrophilic_SA'] = 0.0
            
            aromatic_carbons = sum(1 for atom in mol.GetAtoms() 
                                if atom.GetAtomicNum() == 6 and atom.GetIsAromatic())
            desc['aromatic_carbon_count'] = aromatic_carbons
            
            # Protein binding proxies
            desc['albumin_binding_score'] = desc['logp'] + desc['num_acidic_groups'] * 1.5
            desc['agp_binding_score'] = desc['logp'] + desc['num_basic_groups'] * 1.5 + desc['num_tertiary_amine']
            desc['cad_score'] = (desc['logp'] + (desc['num_basic_groups'] + desc['num_tertiary_amine']) * 2 
                               - desc['num_acidic_groups'])
            desc['protein_binding_proxy'] = max(desc['albumin_binding_score'], desc['agp_binding_score'])
            desc['longest_alkyl_chain'] = self._get_longest_alkyl_chain(mol)
            desc['aromatic_halogen_count'] = self._count_pattern(mol, '[F,Cl,Br,I;$([F,Cl,Br,I]c)]')
            desc['fused_ring_count'] = self._count_fused_rings(mol)
            desc['LE_proxy'] = desc['logp'] / desc['heavy_atom_count'] if desc['heavy_atom_count'] > 0 else 0
            
            # Solubility
            desc['esol_logS'] = (0.16 - 0.63 * desc['logp'] - 0.0062 * desc['mw'] + 
                               0.066 * desc['rotatable'] - 0.74 * desc['aromatic_proportion'])
            
            # Fused aromatic rings
            aromatic_rings = []
            for ring in ring_info.AtomRings():
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    aromatic_rings.append(set(ring))
            fused_aromatic = 0
            for i, r1 in enumerate(aromatic_rings):
                for r2 in aromatic_rings[i+1:]:
                    if len(r1 & r2) >= 2:
                        fused_aromatic += 1
            desc['num_fused_aromatic_rings'] = fused_aromatic
            
            try:
                desc['balaban_j'] = Descriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0
            except Exception:
                desc['balaban_j'] = 0.0
            
            sp2_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization().name == 'SP2')
            desc['sp2_fraction'] = sp2_atoms / desc['heavy_atom_count'] if desc['heavy_atom_count'] > 0 else 0
            desc['aromatic_density'] = num_aromatic_atoms / desc['heavy_atom_count'] if desc['heavy_atom_count'] > 0 else 0
            desc['bertz_ct'] = Descriptors.BertzCT(mol)
            
        except Exception:
            pass
        
        return desc

    def fit_transform(self, smiles_list: List[str]) -> np.ndarray:
        """Fit scaler and transform SMILES to descriptors."""
        data = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            all_desc = self._calculate_single(mol)
            row = [all_desc.get(name, 0.0) for name in self.descriptor_list]
            data.append(row)
        data = np.nan_to_num(np.array(data, dtype=np.float32), nan=0.0)
        return self.scaler.fit_transform(data)

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        """Transform SMILES to descriptors using fitted scaler."""
        data = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            all_desc = self._calculate_single(mol)
            row = [all_desc.get(name, 0.0) for name in self.descriptor_list]
            data.append(row)
        data = np.nan_to_num(np.array(data, dtype=np.float32), nan=0.0)
        return self.scaler.transform(data)


class LegacyMPNNDescriptorCalculator:
    """Simpler descriptor calculator for backward compatibility."""
    
    def __init__(self, descriptor_list: List[str], scaler_type: str = 'standard'):
        self.descriptor_list = descriptor_list
        if scaler_type == 'robust':
            self.scaler = RobustScaler(quantile_range=(5.0, 95.0))
        else:
            self.scaler = StandardScaler()

    def _calc_batch(self, smiles_list: List[str]) -> np.ndarray:
        """Calculate descriptors for a batch of SMILES."""
        data = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            row = []
            if mol:
                for d in self.descriptor_list:
                    try:
                        if hasattr(Descriptors, d):
                            val = getattr(Descriptors, d)(mol)
                        elif hasattr(rdMD, d):
                            val = getattr(rdMD, d)(mol)
                        else:
                            val = 0.0
                    except Exception:
                        val = 0.0
                    row.append(float(val) if val is not None else 0.0)
            else:
                row = [0.0] * len(self.descriptor_list)
            data.append(row)
        return np.array(data, dtype=np.float32)

    def fit_transform(self, smiles_list: List[str]) -> np.ndarray:
        """Fit scaler and transform."""
        data = self._calc_batch(smiles_list)
        return self.scaler.fit_transform(np.nan_to_num(data, nan=0.0))

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        """Transform using fitted scaler."""
        data = self._calc_batch(smiles_list)
        return self.scaler.transform(np.nan_to_num(data, nan=0.0))


class MPNNRefinementDescriptorCalculator:
    """Descriptor calculator for refinement models."""
    
    def __init__(self, scaler_type: str = 'standard'):
        if scaler_type == 'robust':
            self.scaler = RobustScaler(quantile_range=(5.0, 95.0))
        else:
            self.scaler = StandardScaler()

    def calculate_batch(self, smiles_list: List[str]) -> np.ndarray:
        """Calculate refinement descriptors for a batch."""
        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            row = []
            if mol:
                for name in MPNN_REFINEMENT_DESCRIPTORS:
                    try:
                        if hasattr(Descriptors, name):
                            val = getattr(Descriptors, name)(mol)
                        elif hasattr(rdMD, f'Calc{name}'):
                            val = getattr(rdMD, f'Calc{name}')(mol)
                        elif hasattr(rdMD, name):
                            val = getattr(rdMD, name)(mol)
                        else:
                            val = 0.0
                        row.append(float(val) if val is not None else 0.0)
                    except Exception:
                        row.append(0.0)
            else:
                row = [0.0] * len(MPNN_REFINEMENT_DESCRIPTORS)
            results.append(row)
        return np.array(results, dtype=np.float32)

    def fit_transform(self, smiles_list: List[str]) -> np.ndarray:
        """Fit scaler and transform."""
        data = np.nan_to_num(self.calculate_batch(smiles_list), nan=0.0)
        return self.scaler.fit_transform(data)

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        """Transform using fitted scaler."""
        data = np.nan_to_num(self.calculate_batch(smiles_list), nan=0.0)
        return self.scaler.transform(data)


# Add/update this class in core/descriptors.py

class CacoERFeatureGenerator:
    """Feature generator for Caco-2 ER model."""
    
    def __init__(self):
        from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser, Uncharger
        self.lfc = LargestFragmentChooser()
        self.uc = Uncharger()
        
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.fcfp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        )
        
        self.smarts = {
            'basic_n_primary': Chem.MolFromSmarts('[NX3;H2;!$(NC=O)]'),
            'basic_n_secondary': Chem.MolFromSmarts('[NX3;H1;!$(NC=O)]'),
            'basic_n_tertiary': Chem.MolFromSmarts('[NX3;H0;!$(NC=O)]'),
            'carboxylic_acid': Chem.MolFromSmarts('[CX3](=O)[OX1H0-,OX2H1]'),
            'amide': Chem.MolFromSmarts('[NX3][CX3](=[OX1])[#6]'),
        }
    
    def standardize(self, smiles: str):
        """Standardize a molecule."""
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return None
            mol = self.lfc.choose(mol)
            mol = self.uc.uncharge(mol)
            return mol
        except Exception:
            return None
    
    def _get_descriptors(self, mol) -> List[float]:
        """Get descriptors for a molecule."""
        desc = []
        
        try:
            desc.append(Descriptors.MolWt(mol))
            desc.append(Descriptors.MolLogP(mol))
            desc.append(Descriptors.TPSA(mol))
            desc.append(Descriptors.MolMR(mol))
            desc.append(Descriptors.NumHDonors(mol))
            desc.append(Descriptors.NumHAcceptors(mol))
            desc.append(Descriptors.NumRotatableBonds(mol))
            desc.append(Lipinski.RingCount(mol))
            desc.append(rdMD.CalcNumAromaticRings(mol))
            desc.append(Descriptors.HeavyAtomCount(mol))
            desc.append(Descriptors.FractionCSP3(mol))
            desc.append(Chem.GetFormalCharge(mol))
            
            try:
                desc.append(Descriptors.MaxPartialCharge(mol))
                desc.append(Descriptors.MinPartialCharge(mol))
            except Exception:
                desc.extend([0.0, 0.0])
            
            for name, smarts in self.smarts.items():
                if smarts:
                    desc.append(float(len(mol.GetSubstructMatches(smarts))))
                else:
                    desc.append(0.0)
            
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            mw = Descriptors.MolWt(mol)
            
            basic_n = sum([len(mol.GetSubstructMatches(self.smarts[k])) 
                           for k in ['basic_n_primary', 'basic_n_secondary', 'basic_n_tertiary']
                           if self.smarts[k]])
            
            desc.append(float(basic_n))
            desc.append(logp * 0.3 + basic_n * 0.5)
            desc.append(tpsa / max(mw, 1) * 100)
            
        except Exception as e:
            # Return zeros if something fails
            desc = [0.0] * 20
        
        return desc
    
    def process_smiles(self, smiles_list: List[str], verbose: bool = False) -> tuple[np.ndarray, List[int]]:
        """Process SMILES and generate features."""
        try:
            from rdkit.Avalon import pyAvalonTools
        except ImportError:
            pyAvalonTools = None
        
        features = []
        valid_indices = []
        
        for idx, smi in enumerate(smiles_list):
            if verbose and idx % 500 == 0:
                print(f"      Processing {idx}/{len(smiles_list)}")
            
            mol = self.standardize(smi)
            if mol is None:
                continue
            
            try:
                mol_feats = []
                
                # Fingerprints
                mol_feats.append(np.array(self.morgan_gen.GetFingerprint(mol)))
                mol_feats.append(np.array(self.fcfp_gen.GetFingerprint(mol)))
                mol_feats.append(np.array(MACCSkeys.GenMACCSKeys(mol)))
                
                if pyAvalonTools is not None:
                    mol_feats.append(np.array(pyAvalonTools.GetAvalonFP(mol, nBits=1024)))
                else:
                    # Use another fingerprint if Avalon not available
                    mol_feats.append(np.zeros(1024))
                
                # Descriptors
                mol_feats.append(np.array(self._get_descriptors(mol)))
                
                features.append(np.concatenate(mol_feats))
                valid_indices.append(idx)
                
            except Exception as e:
                if verbose:
                    print(f"      Error processing molecule {idx}: {e}")
                continue
        
        if not features:
            return np.array([]).reshape(0, 0), []
        
        result = np.array(features, dtype=np.float32)
        
        # Ensure 2D
        if result.ndim == 1:
            result = result.reshape(1, -1)
        
        return result, valid_indices