# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IDS-QT is a Python-based Network Intrusion Detection System using deep learning (PyTorch). It supports three model architectures for binary classification (normal/attack) on network traffic, with CLI, PyQt5 GUI, and Django Web interfaces.

## Common Commands

```bash
# Install dependencies (use virtual environment in .venv/)
pip install -r requirements.txt

# Data preprocessing
python main.py --task preprocess --dataset cicids2017

# Train models (cnn, lstm, cnn_lstm, or all)
python main.py --task train --dataset cicids2017 --model cnn_lstm --epochs 30

# Evaluate a trained model
python main.py --task evaluate --dataset cicids2017 --model cnn

# Full pipeline (preprocess + train + evaluate)
python main.py --task all --dataset cicids2017 --model all

# Real-time traffic capture (requires admin privileges)
python main.py --task capture --capture_time 60

# Detect attacks in captured traffic
python main.py --task detect --capture_file captured_data/file.csv --detect_model cnn_lstm

# Launch GUI (PyQt5 desktop)
python gui.py

# Launch Web UI (Django)
python manage.py runserver
```

Key CLI parameters: `--batch_size` (default 64), `--lr` (default 0.001), `--hidden_dim` (default 128), `--num_layers` (default 2), `--no_cuda`, `--data_dir`, `--save_dir`, `--results_dir`.

## Architecture

**Three-layer design:**
- **Data Layer** (`utils/data_utils.py`, `data_preprocessing.py`, `capture_to_csv.py`): Dataset loading/preprocessing for NSL-KDD and CICIDS2017, plus real-time packet capture via Scapy that extracts 28 statistical features from network flows.
- **Model Layer** (`models/`): Three interchangeable PyTorch models, all ending with FC layers + softmax:
  - `cnn_model.py` — IDSConvNet: 1D CNN with 3 conv blocks (BatchNorm + Pool + Dropout)
  - `lstm_model.py` — IDSLSTM: Bidirectional LSTM with self-attention
  - `cnn_lstm_model.py` — IDSCNNLSTM: CNN (2 conv blocks) → BiLSTM → attention (hybrid)
- **Application Layer**:
  - `main.py` — CLI entry point
  - `gui.py` — PyQt5 desktop GUI
  - `ids_web/` + `ids/` — Django Web UI (views, templates, static files)
    - `ids/tasks.py` — TaskManager: background task runner (replaces QThread Worker)
    - `ids/views.py` — API endpoints (preprocess/train/evaluate/capture/detect)
    - `ids/templates/ids/index.html` — main page template
    - `ids/static/ids/` — CSS & JS

**Training pipeline** (`utils/training.py`): Adam optimizer, cross-entropy loss, early stopping (patience=5), best-model checkpointing based on validation loss.

**Data pipeline** (`utils/data_utils.py`): StandardScaler for numeric features, OneHotEncoder for categorical, via sklearn ColumnTransformer.

## Key Paths

- Datasets: `data/nsl_kdd/`, `data/cicids2017/` (processed numpy files in `*_processed/` subdirs)
- Saved models: `saved_models/{model}_{dataset}_model.pth`
- Results: `results/{model}_{dataset}/` (confusion matrices, ROC curves, precision-recall curves)
- Captured traffic: `captured_data/`

## Important Notes

- CICIDS2017-trained models match real-time captured traffic features; NSL-KDD models may not align with real flow features.
- Real-time capture requires administrator/root privileges.
- All models perform binary classification (normal vs. attack).

## Behavior Guidelines

- Always respond in Chinese (Simplified Chinese / 简体中文).
