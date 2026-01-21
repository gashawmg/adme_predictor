# models/caco_er_model.py
"""Optimized Caco-2 ER model."""

import os
import numpy as np
from typing import List, Tuple, Optional

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors as rdMD, Lipinski, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser, Uncharger

from utils.io_utils import load_pickle


class CacoERFeatureGenerator:
    """Feature generator specifically for Caco-2 ER model."""
    
    def __init__(self):
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
    
    def standardize(self, smiles: str) -> Optional[Chem.Mol]:
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
            desc.append(float(Descriptors.MolWt(mol)))
            desc.append(float(Descriptors.MolLogP(mol)))
            desc.append(float(Descriptors.TPSA(mol)))
            desc.append(float(Descriptors.MolMR(mol)))
            desc.append(float(Descriptors.NumHDonors(mol)))
            desc.append(float(Descriptors.NumHAcceptors(mol)))
            desc.append(float(Descriptors.NumRotatableBonds(mol)))
            desc.append(float(Lipinski.RingCount(mol)))
            desc.append(float(rdMD.CalcNumAromaticRings(mol)))
            desc.append(float(Descriptors.HeavyAtomCount(mol)))
            desc.append(float(Descriptors.FractionCSP3(mol)))
            desc.append(float(Chem.GetFormalCharge(mol)))
            
            try:
                desc.append(float(Descriptors.MaxPartialCharge(mol)))
                desc.append(float(Descriptors.MinPartialCharge(mol)))
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
            desc.append(float(logp * 0.3 + basic_n * 0.5))
            desc.append(float(tpsa / max(mw, 1) * 100))
            
        except Exception as e:
            print(f"      Error computing descriptors: {e}")
            desc = [0.0] * 22  # Expected number of descriptors
        
        return desc
    
    def process_smiles(self, smiles_list: List[str], verbose: bool = False) -> Tuple[np.ndarray, List[int]]:
        """Process SMILES and generate features."""
        # Try to import Avalon
        try:
            from rdkit.Avalon import pyAvalonTools
            use_avalon = True
        except ImportError:
            use_avalon = False
            print("      Warning: Avalon fingerprints not available")
        
        features = []
        valid_indices = []
        
        for idx, smi in enumerate(smiles_list):
            if verbose and idx % 100 == 0:
                print(f"      Processing {idx}/{len(smiles_list)}")
            
            mol = self.standardize(smi)
            if mol is None:
                if verbose:
                    print(f"      Could not parse: {smi}")
                continue
            
            try:
                mol_feats = []
                
                # Morgan fingerprint (2048 bits)
                morgan_fp = np.array(self.morgan_gen.GetFingerprint(mol), dtype=np.float32)
                mol_feats.append(morgan_fp)
                
                # FCFP fingerprint (2048 bits)
                fcfp_fp = np.array(self.fcfp_gen.GetFingerprint(mol), dtype=np.float32)
                mol_feats.append(fcfp_fp)
                
                # MACCS keys (167 bits)
                maccs_fp = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
                mol_feats.append(maccs_fp)
                
                # Avalon or placeholder (1024 bits)
                if use_avalon:
                    avalon_fp = np.array(pyAvalonTools.GetAvalonFP(mol, nBits=1024), dtype=np.float32)
                else:
                    avalon_fp = np.zeros(1024, dtype=np.float32)
                mol_feats.append(avalon_fp)
                
                # Descriptors
                desc = np.array(self._get_descriptors(mol), dtype=np.float32)
                mol_feats.append(desc)
                
                # Concatenate all features
                combined = np.concatenate(mol_feats)
                features.append(combined)
                valid_indices.append(idx)
                
            except Exception as e:
                if verbose:
                    print(f"      Error processing molecule {idx}: {e}")
                continue
        
        if not features:
            # Return empty 2D array
            return np.array([]).reshape(0, 0), []
        
        # Stack into 2D array
        result = np.vstack(features).astype(np.float32)
        
        if verbose:
            print(f"      Generated features shape: {result.shape}")
        
        return result, valid_indices


class OptimizedCacoERModel:
    """Optimized stacking ensemble for Log_Caco_ER prediction."""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.var_selector = None
        self.scaler_X = None
        self.scaler_y = None
        self.final_models = {}
        self.meta_model = None
        self.training_stats = {}
        self.feature_generator = CacoERFeatureGenerator()
        self._is_loaded = False
    
    def load_models(self) -> bool:
        """Load saved model components."""
        print(f"      Loading Caco-ER model from: {self.model_dir}")
        
        try:
            model_path = os.path.join(self.model_dir, 'models.pkl')
            if not os.path.exists(model_path):
                print(f"      ✗ models.pkl not found at: {model_path}")
                return False
            
            data = load_pickle(model_path)
            
            self.var_selector = data.get('var_selector')
            self.scaler_X = data.get('scaler_X')
            self.scaler_y = data.get('scaler_y')
            self.final_models = data.get('final_models', {})
            self.meta_model = data.get('meta_model')
            self.training_stats = data.get('training_stats', {})
            
            print(f"      ✓ Loaded {len(self.final_models)} base models")
            print(f"      ✓ Training stats: {self.training_stats}")
            
            self._is_loaded = True
            return True
            
        except Exception as e:
            print(f"      ✗ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Make predictions from SMILES."""
        print(f"      Caco-ER predicting for {len(smiles_list)} molecules")
        
        if not self._is_loaded:
            print(f"      ✗ Model not loaded")
            return np.zeros(len(smiles_list))
        
        # Default prediction value
        default_value = self.training_stats.get('y_mean', 0.5)
        
        # Generate features
        print(f"      Generating features...")
        X, valid_idx = self.feature_generator.process_smiles(smiles_list, verbose=False)
        
        print(f"      Features shape: {X.shape if len(X) > 0 else 'empty'}")
        print(f"      Valid indices: {valid_idx}")
        
        if len(X) == 0 or X.shape[0] == 0:
            print(f"      ✗ No valid features generated")
            return np.full(len(smiles_list), default_value)
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        try:
            # Apply variance selector
            if self.var_selector is not None:
                print(f"      Applying variance selector...")
                X_sel = self.var_selector.transform(X)
            else:
                X_sel = X
            
            print(f"      After variance selection: {X_sel.shape}")
            
            # Ensure still 2D
            if X_sel.ndim == 1:
                X_sel = X_sel.reshape(1, -1)
            
            # Apply scaler
            if self.scaler_X is not None:
                print(f"      Applying X scaler...")
                X_scaled = self.scaler_X.transform(X_sel)
            else:
                X_scaled = X_sel
            
            # Clip extreme values
            X_scaled = np.clip(X_scaled, -10, 10)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10, neginf=-10)
            
            print(f"      Scaled features shape: {X_scaled.shape}")
            
            # Get base model predictions
            base_preds = []
            for name, model in self.final_models.items():
                try:
                    pred = model.predict(X_scaled)
                    pred = np.atleast_1d(pred).flatten()
                    pred = np.clip(pred, -5, 5)
                    base_preds.append(pred)
                    print(f"      Base model {name}: {pred}")
                except Exception as e:
                    print(f"      ✗ Error in base model {name}: {e}")
                    base_preds.append(np.zeros(X_scaled.shape[0]))
            
            if not base_preds:
                print(f"      ✗ No base predictions")
                return np.full(len(smiles_list), default_value)
            
            # Stack predictions
            stack_input = np.column_stack(base_preds)
            print(f"      Stack input shape: {stack_input.shape}")
            
            # Meta prediction
            if self.meta_model is not None:
                meta_pred_scaled = self.meta_model.predict(stack_input)
            else:
                meta_pred_scaled = np.mean(stack_input, axis=1)
            
            meta_pred_scaled = np.atleast_1d(meta_pred_scaled).flatten()
            print(f"      Meta predictions (scaled): {meta_pred_scaled}")
            
            # Inverse transform
            if self.scaler_y is not None:
                final_pred = self.scaler_y.inverse_transform(
                    meta_pred_scaled.reshape(-1, 1)
                ).ravel()
            else:
                final_pred = meta_pred_scaled
            
            print(f"      Final predictions: {final_pred}")
            
            # Clip to training range
            y_min = self.training_stats.get('y_min', -2)
            y_max = self.training_stats.get('y_max', 3)
            final_pred = np.clip(final_pred, y_min - 0.1, y_max + 0.2)
            
            # Handle NaN
            final_pred = np.nan_to_num(final_pred, nan=default_value)
            
            # Map back to full array
            full_preds = np.full(len(smiles_list), default_value)
            for i, idx in enumerate(valid_idx):
                if i < len(final_pred):
                    full_preds[idx] = final_pred[i]
            
            print(f"      Output predictions: {full_preds}")
            return full_preds
            
        except Exception as e:
            print(f"      ✗ Error in prediction pipeline: {e}")
            import traceback
            traceback.print_exc()
            return np.full(len(smiles_list), default_value)