# ToxLens 🔬

AI-powered drug toxicity prediction across 12 biological assays with atomic-level explainability.

## Overview

ToxLens is a machine learning system that predicts the toxicity of chemical compounds using molecular graph representations and fingerprint-based features. Given a SMILES string, ToxLens predicts toxicity risk across 12 Tox21 assay targets and highlights which atoms in the molecule drive the prediction.

## Live Demo

[Launch ToxLens](https://huggingface.co/spaces/Swastikraj599/ToxLens) ← *deploy to HuggingFace Spaces after hackathon*

## Key Features

- **Multi-task prediction** across 12 Tox21 toxicity assays simultaneously
- **Atomic-level explainability** — highlights high-risk atoms directly on the molecular structure
- **Model comparison** — XGBoost (ECFP4) vs AttentiveFP GNN with documented results
- **SHAP analysis** — identifies which molecular descriptors drive toxicity
- **Interactive UI** — live SMILES input with instant predictions via Gradio

## Results

| Model | Mean AUC-ROC | Notes |
|-------|-------------|-------|
| XGBoost + ECFP4 | **0.8566** | Primary model, used in production |
| AttentiveFP GNN | 0.8278 | Graph-native, atomic explainability |

Best single assay: SR-MMP at 0.9254 (XGBoost)

## Datasets

| Dataset | Source | Usage |
|---------|--------|-------|
| Tox21 | [Kaggle](https://www.kaggle.com/datasets/epicskills/tox21-dataset) | Primary — 7823 compounds, 12 assays |
| ZINC250k | [Kaggle](https://www.kaggle.com/datasets/basu369victor/zinc250k) | Secondary — logP, QED feature enrichment |

## Tech Stack

| Layer | Tools |
|-------|-------|
| Molecular featurization | RDKit, ECFP4 Morgan fingerprints |
| Baseline model | XGBoost + Optuna hyperparameter tuning |
| GNN model | AttentiveFP via PyTorch Geometric |
| Explainability | SHAP, atom-level fingerprint attribution |
| Experiment tracking | MLflow |
| Interface | Gradio |

## Project Structure
```
ToxLens/
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_baseline_xgboost.ipynb
│   ├── 03_gnn_attentivefp.ipynb
│   └── 04_explainability.ipynb
├── app/
│   └── gradio_app.py
├── assets/
│   └── demo.gif
├── requirements.txt
└── README.md
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| 01 | EDA, SMILES validation, RDKit descriptors, ZINC250k feature merge, Morgan fingerprints |
| 02 | Multi-task XGBoost with Optuna tuning, SHAP global importance, MLflow tracking |
| 03 | AttentiveFP GNN, molecular graph construction, model comparison vs baseline |
| 04 | Atom-level attribution, highlighted molecular visualization, Gradio live app |

## Molecular Property Analysis

Top molecular descriptors by SHAP importance:
- **TPSA** (Topological Polar Surface Area)
- **MolWt** (Molecular Weight)
- **LogP** (Lipophilicity)

## Setup
```bash
pip install rdkit xgboost torch torch-geometric gradio shap optuna mlflow deepchem
```

All notebooks are self-contained with Kaggle API data loading built in.
Run each notebook top to bottom in order.

## Hackathon

Built for CodeCure AI Hackathon — Track A: Drug Toxicity Prediction.
```


cairosvg
