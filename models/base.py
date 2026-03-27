# models/base.py
"""Base predictor class."""

from abc import ABC, abstractmethod

import numpy as np


class BasePredictor(ABC):
    """Abstract base class for all predictors."""

    @abstractmethod
    def load_models(self) -> bool:
        """Load model files. Returns True if successful."""
        pass

    @abstractmethod
    def predict(self, smiles_list: list[str]) -> np.ndarray:
        """Make predictions for a list of SMILES."""
        pass
