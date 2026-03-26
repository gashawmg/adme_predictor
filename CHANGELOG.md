# Changelog

All notable changes to ADME Predictor are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/):
- **Patch** (`0.1.x`): model weight updates, bug fixes — no API change
- **Minor** (`0.x.0`): new ADMET targets added
- **Major** (`x.0.0`): breaking change to SMILES input interface or prediction output schema

---

## [Unreleased]

### Added
- `pyproject.toml` with Ruff, mypy, and pytest configuration
- `.github/workflows/ci.yml` with lint, security audit, and unit test jobs
- `tests/unit/` with tests for standardizer, conversion config, and model config integrity
- `tests/integration/` smoke tests (require local model files; skipped in CI)
- `.streamlit/config.toml` — moves CORS/XSRF settings out of devcontainer CLI flags
- `.pre-commit-config.yaml` with ruff, large-file guard, and hardcoded-path detection
- `model_sources.example.json` — template for `consolidate_models.py` source paths

### Changed
- `consolidate_models.py`: replaced hardcoded `C:\Users\gasha\...` paths with
  `model_sources.local.json` (gitignored); users copy the example template and
  fill in their own paths
- `.devcontainer/devcontainer.json`: removed insecure `--enableCORS false
  --enableXsrfProtection false` CLI flags (now in `.streamlit/config.toml`)

### Fixed
- `.gitignore` added — `__pycache__/` and `*.pyc` files no longer tracked

---

## [0.1.0] - 2026-01-02

Initial release of ADME Predictor.

### Added
- Streamlit web application (`app.py`) for ADMET property prediction
- 9 predicted targets: LogD, KSol (LogS), HLM CLint, MLM CLint,
  Caco-2 Papp A>B, Caco-2 Efflux Ratio, Mouse PPB/BPB/MPB
- MPNN backbone via ChemProp 2.2.1 with 5-fold cross-validation ensembles
- Multitask models: LogD+LogS shared checkpoint; Mouse PPB/BPB/MPB shared checkpoint
- Caco-2 ER ensemble model (Random Forest + XGBoost + LightGBM + CatBoost)
- RDKit molecular descriptor calculator (40+ features)
- Molecule standardizer: canonical SMILES, largest-fragment selection, decharging
- GPU/CPU auto-detection with deterministic seed (42)
- CSV upload (column auto-detection: `SMILES`, `smiles`, `Smiles`, `canonical_smiles`)
- CSV download of predictions with optional log-scale columns
- Dev Container configuration for GitHub Codespaces / VS Code Remote

### Known Issues
- Pickle class registration workaround required in `app.py` for cross-script
  model deserialization compatibility
- `streamlit` pinned to `<1.33` due to `altair==4.2.2` compatibility constraint
- Trained model weights committed directly to git (should migrate to Git LFS
  or GitHub Releases for repositories with storage constraints)
