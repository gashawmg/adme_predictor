# utils/io_utils.py
"""File I/O utilities with pickle compatibility fixes."""

import os
import pickle
import sys
from typing import Any


def _register_classes_for_unpickling():
    """Register classes that might be needed for unpickling old models."""
    # Import the classes
    from models.refinement import RefinementStack
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    
    # Register them in __main__ so pickle can find them
    import __main__
    
    if not hasattr(__main__, 'RefinementStack'):
        __main__.RefinementStack = RefinementStack
    
    # Also try to handle any other classes that might be needed
    try:
        from sklearn.linear_model import Ridge, HuberRegressor
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        import xgboost as xgb
        import lightgbm as lgb
        
        __main__.Ridge = Ridge
        __main__.HuberRegressor = HuberRegressor
        __main__.RandomForestRegressor = RandomForestRegressor
        __main__.ExtraTreesRegressor = ExtraTreesRegressor
        __main__.XGBRegressor = xgb.XGBRegressor
        __main__.LGBMRegressor = lgb.LGBMRegressor
    except Exception:
        pass


def save_pickle(obj: Any, path: str) -> None:
    """Save an object to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load an object from a pickle file with compatibility handling."""
    # Register classes before unpickling
    _register_classes_for_unpickling()
    
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except AttributeError as e:
        # Try alternative loading methods
        print(f"      Warning: Standard pickle load failed: {e}")
        print(f"      Attempting alternative load...")
        
        try:
            # Try with custom unpickler
            with open(path, 'rb') as f:
                return CustomUnpickler(f).load()
        except Exception as e2:
            print(f"      Alternative load also failed: {e2}")
            raise


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing class references."""
    
    def find_class(self, module, name):
        """Override to redirect class lookups."""
        # Map old module paths to new ones
        class_mappings = {
            'RefinementStack': ('models.refinement', 'RefinementStack'),
            'MPNNDescriptorCalculator': ('core.descriptors', 'MPNNDescriptorCalculator'),
            'LegacyMPNNDescriptorCalculator': ('core.descriptors', 'LegacyMPNNDescriptorCalculator'),
            'MPNNRefinementDescriptorCalculator': ('core.descriptors', 'MPNNRefinementDescriptorCalculator'),
        }
        
        if name in class_mappings:
            new_module, new_name = class_mappings[name]
            try:
                mod = __import__(new_module, fromlist=[new_name])
                return getattr(mod, new_name)
            except Exception:
                pass
        
        # Handle __main__ references
        if module == '__main__':
            # Try to find the class in our modules
            try_modules = [
                'models.refinement',
                'core.descriptors',
                'sklearn.preprocessing',
                'sklearn.linear_model',
                'sklearn.ensemble',
            ]
            
            for try_module in try_modules:
                try:
                    mod = __import__(try_module, fromlist=[name])
                    if hasattr(mod, name):
                        return getattr(mod, name)
                except Exception:
                    continue
        
        # Fall back to default behavior
        return super().find_class(module, name)