"""Integration smoke tests — require trained model files. Run with: pytest -m integration"""

from pathlib import Path
import numpy as np
import pytest

MODELS_DIR = Path(__file__).parent.parent.parent / "trained_models"
LOGD_DIR = MODELS_DIR / "LogD"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("torch")
    if not LOGD_DIR.exists():
        pytest.skip("trained_models/LogD not found — skipping integration tests")
    from core.engine import PredictionEngine
    return PredictionEngine()


SAMPLE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",     # aspirin
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # caffeine
    "CCO",                            # ethanol
]


def test_logd_predictions_shape(engine):
    if not engine.load_predictor("LogD"):
        pytest.skip("Could not load LogD model")
    preds = engine.predict("LogD", SAMPLE_SMILES)
    assert len(preds) == len(SAMPLE_SMILES)


def test_logd_predictions_finite(engine):
    if not engine.load_predictor("LogD"):
        pytest.skip("Could not load LogD model")
    preds = engine.predict("LogD", SAMPLE_SMILES)
    assert all(np.isfinite(p) for p in preds), "LogD predictions contain non-finite values"


def test_available_targets_non_empty(engine):
    targets = engine.get_available_targets()
    assert len(targets) > 0, "No available targets found"
