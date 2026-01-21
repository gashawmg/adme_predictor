# models/refinement.py
"""Refinement stack for residual correction."""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb

from config.settings import GLOBAL_SEED


class RefinementStack:
    """Stack of models for residual refinement."""
    
    def __init__(self):
        self.models = {
            'lgb': lgb.LGBMRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.03,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, verbose=-1, random_state=GLOBAL_SEED
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                verbosity=0, random_state=GLOBAL_SEED
            ),
            'rf': RandomForestRegressor(
                n_estimators=300, max_depth=8, max_features='sqrt',
                n_jobs=-1, random_state=GLOBAL_SEED
            ),
        }
        self.meta = Ridge(alpha=1.0)
        self.trained_models = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the refinement stack."""
        oof_preds = np.zeros((len(y), len(self.models)))
        kf = KFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
        
        for tr_idx, val_idx in kf.split(X):
            for i, (name, model) in enumerate(self.models.items()):
                m = clone(model)
                m.fit(X[tr_idx], y[tr_idx])
                oof_preds[val_idx, i] = m.predict(X[val_idx])
        
        self.meta.fit(oof_preds, y)
        
        for name, model in self.models.items():
            m = clone(model)
            m.fit(X, y)
            self.trained_models[name] = m

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.trained_models:
            return np.zeros(X.shape[0])
        
        preds = np.column_stack([m.predict(X) for m in self.trained_models.values()])
        return self.meta.predict(preds)
    
    # Make the class picklable
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, state):
        self.__dict__.update(state)