# ADME Predictor

A web application for predicting ADMET (Absorption, Distribution, Metabolism,
Excretion, Toxicity) properties of drug-like molecules. Built with
[ChemProp](https://github.com/chemprop/chemprop) MPNN models, ensemble ML, and
[Streamlit](https://streamlit.io).

[![CI](https://github.com/gashawmg/adme_predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/gashawmg/adme_predictor/actions/workflows/ci.yml)

---

## Predicted Properties

| Property | Symbol | Unit | Model |
|---|---|---|---|
| Distribution Coefficient | LogD | — | MPNN |
| Kinetic Solubility | KSol | µM | Multitask MPNN |
| Human Liver Microsomal Clearance | HLM CLint | mL/min/kg | MPNN |
| Mouse Liver Microsomal Clearance | MLM CLint | mL/min/kg | MPNN |
| Caco-2 Permeability A→B | Papp A>B | 10⁻⁶ cm/s | MPNN |
| Caco-2 Efflux Ratio | Efflux | — | RF/XGB/LGB ensemble |
| Mouse Plasma Protein Binding | MPPB | % Unbound | MPNN |
| Mouse Brain Protein Binding | MBPB | % Unbound | MPNN |
| Mouse Muscle Protein Binding | MGMB | % Unbound | Multitask MPNN |

---

## Requirements

- **Python 3.11**
- **CUDA 11.8+** (optional — CPU inference is supported but slower)
- **libxrender1** (Linux/headless only — required by RDKit for molecule rendering)

```bash
# Ubuntu/Debian
sudo apt install libxrender1
```

---

## Installation

### Option A — conda (recommended)

```bash
# 1. Create environment
conda create -n adme_predictor python=3.11 -y
conda activate adme_predictor

# 2. Install PyTorch with CUDA (adjust the index URL for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 3. Install ChemProp (must come after torch)
pip install chemprop==2.2.1

# 4. Install remaining dependencies
pip install -r requirements.txt
```

### Option B — pip only

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install chemprop==2.2.1
pip install -r requirements.txt
```

> **Note:** `altair==4.2.2` is pinned for compatibility with `streamlit<1.33`.
> Upgrade both together when moving to Streamlit ≥1.33.

---

## Model Files

The trained model weights are stored in `trained_models/` and are included in
this repository. Each subdirectory contains:

```
trained_models/
├── LogD/
│   ├── fold_0.ckpt … fold_4.ckpt   # PyTorch Lightning checkpoints
│   ├── scaler_y.pkl                 # Target variable scaler
│   ├── desc_list_integrated.pkl     # Descriptor feature names
│   └── desc_scaler_integrated.pkl   # Descriptor scaler
├── LogS/          # Multitask LogD+LogS checkpoint
├── Log_HLM_CLint/
├── Log_MLM_CLint/
├── Log_Caco_Papp_AB/
├── Log_Caco_ER/   # Ensemble model (RF/XGB/LGB/CatBoost)
├── Log_Mouse_PPB/
├── Log_Mouse_BPB/
└── Log_Mouse_MPB/ # Multitask Mouse PPB/BPB/MPB checkpoint
```

### Assembling models from scratch

If you have trained your own models and want to copy them in:

```bash
# Copy model_sources.example.json → model_sources.local.json
cp model_sources.example.json model_sources.local.json

# Edit model_sources.local.json — fill in each source path
# Then run:
python consolidate_models.py
```

`model_sources.local.json` is gitignored and never committed.

---

## Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

The app auto-detects GPU. If no CUDA device is found, it falls back to CPU.

---

## Usage

### Input

**Manual entry:** Paste one SMILES per line in the text area.

**CSV upload:** Upload a `.csv` file with a SMILES column. The column is
auto-detected by name (`SMILES`, `smiles`, `Smiles`, or `canonical_smiles`).

### Configuration (sidebar)

- **Select Properties:** Choose any subset of the 9 ADMET targets.
- **Include log-scale values:** Toggle to add the raw log-scale predictions
  alongside the converted values in the output CSV.

### Output

Results are displayed as a table and can be downloaded as CSV.
Invalid SMILES are silently skipped with a warning count shown.

---

## Development

### Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

### Running tests

```bash
# Unit tests (no model files required — runs in CI)
pytest tests/unit/

# Integration tests (requires trained_models/ to be populated)
pytest -m integration
```

### Code style

```bash
ruff check .          # lint
ruff format .         # format
```

---

## Architecture

```
app.py                  ← Streamlit entry point
config/
  settings.py           ← Device, paths, seeds, training hyperparams
  model_config.py       ← TARGET_CONFIG, MULTITASK_CONFIG, checkpoint versions
  conversion_config.py  ← Log→actual conversions, units, display names
  descriptor_config.py  ← Descriptor selection per target
core/
  engine.py             ← PredictionEngine — unified inference interface
  descriptors.py        ← RDKit molecular descriptor calculator (40+ features)
  standardizer.py       ← SMILES canonicalization and fragment selection
models/
  mpnn_predictors.py    ← 4 MPNN predictor variants (script1/2, hybrid_v5, etc.)
  multitask_predictors.py ← Shared-checkpoint multitask predictors
  caco_er_model.py      ← RF/XGB/LGB/CatBoost ensemble for Caco-2 ER
  refinement.py         ← Post-prediction refinement stacks (PPB, BPB)
ui/
  components.py         ← SMILES input, molecule rendering
  results.py            ← Results table, statistics, download button
utils/
  conversion.py         ← DataFrame construction, value formatting
  helpers.py            ← Checkpoint loading, training config utilities
  io_utils.py           ← Pickle / CSV I/O
trained_models/         ← Pre-trained model checkpoints and scalers
```

**Inference flow:**

1. SMILES → `MoleculeStandardizer` → canonical SMILES
2. `PredictionEngine.predict(target, smiles)` dispatches to the correct predictor
3. Predictor: calculate descriptors → ChemProp dataloader → MPNN forward pass → inverse-scale
4. Apply per-target constraints (e.g., PPB clipped to [0, 2.0])
5. Convert log-scale predictions → actual values with units

### Known limitations

- **Pickle compatibility:** `app.py` registers model classes in `__main__` before loading any
  checkpoint. This is required because models were serialized in separate training scripts.
  Renaming any model class will break deserialization of existing checkpoints.
- **Streamlit version constraint:** Pinned to `<1.33` because `altair==4.2.2` (required by
  that Streamlit generation) is incompatible with newer Streamlit releases.
- **No concurrent session isolation:** Multiple browser tabs share GPU memory. Very large
  batches on multi-user deployments may cause OOM errors.

---

## License

MIT
