# models/__init__.py
"""Model classes package."""

# Import RefinementStack first so it's available for pickle
from .base import BasePredictor
from .caco_er_model import OptimizedCacoERModel
from .mpnn_predictors import (
    MPNNPredictorHybridV5Integrated,
    MPNNPredictorHybridV5Refinement,
    MPNNPredictorScript1,
    MPNNPredictorScript2,
)
from .multitask_predictors import (
    LegacyMultitaskMPNNPredictor,
    MultitaskMPNNPredictorScript1,
    MultitaskMPNNPredictorUnified,
)
from .refinement import RefinementStack

__all__ = [
    "RefinementStack",
    "BasePredictor",
    "OptimizedCacoERModel",
    "MPNNPredictorScript1",
    "MPNNPredictorScript2",
    "MPNNPredictorHybridV5Integrated",
    "MPNNPredictorHybridV5Refinement",
    "MultitaskMPNNPredictorUnified",
    "MultitaskMPNNPredictorScript1",
    "LegacyMultitaskMPNNPredictor",
]
