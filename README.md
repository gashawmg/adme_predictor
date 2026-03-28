# ADME Predictor

A web application for predicting ADM(Absorption, Distribution, Metabolism) properties of drug-like molecules. Built with
[ChemProp](https://github.com/chemprop/chemprop) MPNN models, ensemble ML, and
[Streamlit](https://streamlit.io).

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://admepredictor.streamlit.app/)
[![CI](https://github.com/gashawmg/adme_predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/gashawmg/adme_predictor/actions/workflows/ci.yml)

---

## Background

These models were developed for the **[ExpansionRx-OpenADMET Blind Challenge](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge)**,
a competitive benchmarking challenge with **370+ participants** from academia and industry.
The submitted predictions on the blind test set achieved **14th place** overall.

All models were trained exclusively on the challenge's official training data — no external
datasets or pre-trained property models were used. This makes the results a direct reflection
of what can be achieved with modern MPNN architectures and carefully engineered molecular
descriptors on the provided data alone.

---

## Predicted Properties

| Property | Symbol | Unit | Model Architecture |
|---|---|---|---|
| Distribution Coefficient | LogD | — | MPNN (ChemProp) |
| Kinetic Solubility | KSol | µM | Multitask MPNN |
| Human Liver Microsomal Clearance | HLM CLint | mL/min/kg | MPNN (ChemProp) |
| Mouse Liver Microsomal Clearance | MLM CLint | mL/min/kg | MPNN + descriptors |
| Caco-2 Permeability A→B | Papp A>B | 10⁻⁶ cm/s | MPNN (ChemProp) |
| Caco-2 Efflux Ratio | Efflux | — | RF / XGB / LGB ensemble |
| Mouse Plasma Protein Binding | MPPB | % Unbound | MPNN + refinement |
| Mouse Brain Protein Binding | MBPB | % Unbound | MPNN + refinement |
| Mouse Muscle Protein Binding | MGMB | % Unbound | Multitask MPNN |

### Model design highlights

- **5-fold cross-validation ensembles** for all targets — each prediction is the mean
  across 5 independently trained folds, which reduces variance and improves generalization.
- **Hybrid MPNN + descriptor integration** — 40+ RDKit physicochemical descriptors
  (logP, TPSA, HBD/HBA, ring counts, halogen counts, ionization proxies, etc.) are
  concatenated with the MPNN molecular embedding before the output head, giving the
  model both learned graph features and domain-informed structural features.
- **Multitask learning** for related properties — LogD and LogS share a single model
  with two output heads; Mouse PPB, BPB, and MPB share a three-head model. This
  exploits inter-property correlations and regularizes training on smaller datasets.
- **Caco-2 Efflux Ratio** uses a stacked ensemble of Random Forest, XGBoost, LightGBM,
  and CatBoost trained on molecular descriptors and fingerprints — gradient boosting
  methods outperformed MPNN for this target on the challenge data.
- **Post-prediction refinement stacks** for protein binding targets — a secondary
  scikit-learn model corrects systematic biases in the MPNN output for PPB and BPB.
- **No external data** — all models trained solely on ExpansionRx-OpenADMET challenge
  training data, making this a clean benchmark of model architecture choices.

---

## Live Demo

**[https://admepredictor.streamlit.app/](https://admepredictor.streamlit.app/)**

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

## Running Locally

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
app.py                  <- Streamlit entry point
config/
  settings.py           <- Device, paths, seeds, training hyperparams
  model_config.py       <- TARGET_CONFIG, MULTITASK_CONFIG, checkpoint versions
  conversion_config.py  <- Log->actual conversions, units, display names
  descriptor_config.py  <- Descriptor selection per target
core/
  engine.py             <- PredictionEngine -- unified inference interface
  descriptors.py        <- RDKit molecular descriptor calculator (40+ features)
  standardizer.py       <- SMILES canonicalization and fragment selection
models/
  mpnn_predictors.py    <- 4 MPNN predictor variants (script1/2, hybrid_v5, etc.)
  multitask_predictors.py <- Shared-checkpoint multitask predictors
  caco_er_model.py      <- RF/XGB/LGB/CatBoost ensemble for Caco-2 ER
  refinement.py         <- Post-prediction refinement stacks (PPB, BPB)
ui/
  components.py         <- SMILES input, molecule rendering
  results.py            <- Results table, statistics, download button
utils/
  conversion.py         <- DataFrame construction, value formatting
  helpers.py            <- Checkpoint loading, training config utilities
  io_utils.py           <- Pickle / CSV I/O
trained_models/         <- Pre-trained model checkpoints and scalers
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

## Citation

If you use this tool in your research, please cite the ExpansionRx-OpenADMET challenge:

> Gashaw M. Goshu. *ADME Predictor: MPNN-based ADMET property prediction.*
> Developed for the ExpansionRx-OpenADMET Blind Challenge (14th place / 370+ participants).
> GitHub: https://github.com/gashawmg/adme_predictor

---

## License

MIT
