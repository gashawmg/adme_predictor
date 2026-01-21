# models/__init__.py
"""Model classes package."""

# Import RefinementStack first so it's available for pickle
from .refinement import RefinementStack

from .base import BasePredictor
from .caco_er_model import OptimizedCacoERModel
from .mpnn_predictors import (
    MPNNPredictorScript1,
    MPNNPredictorScript2,
    MPNNPredictorHybridV5Integrated,
    MPNNPredictorHybridV5Refinement,
)
from .multitask_predictors import (
    MultitaskMPNNPredictorUnified,
    MultitaskMPNNPredictorScript1,
    LegacyMultitaskMPNNPredictor,
)

__all__ = [
    'RefinementStack',
    'BasePredictor',
    'OptimizedCacoERModel',
    'MPNNPredictorScript1',
    'MPNNPredictorScript2',
    'MPNNPredictorHybridV5Integrated',
    'MPNNPredictorHybridV5Refinement',
    'MultitaskMPNNPredictorUnified',
    'MultitaskMPNNPredictorScript1',
    'LegacyMultitaskMPNNPredictor',
]