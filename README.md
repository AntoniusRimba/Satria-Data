# Smart Waste Classification

## SATRIA DATA 2026 - Big Data Challenge

Machine Learning & Computer Vision project for automatic waste classification.

---

# Project Overview

Smart Waste Classification is an image classification project developed for the SATRIA DATA 2026 Big Data Challenge.

The project focuses on building an Intelligent Decision Module capable of identifying waste materials from images and classifying them into predefined categories.

The resulting model is intended to support the development of an Automatic Waste Sorting System by providing accurate waste classification decisions that can later be utilized by external systems such as microcontrollers, PLCs, or robotic actuators.

---

# Problem Statement

How can a machine learning model automatically recognize waste material types from images?

The model receives an image as input and predicts one of the following classes:

- Recyclable
- Electronic
- Organic

This task belongs to:

- Supervised Learning
- Computer Vision
- Image Classification

---

# Competition Context

Competition:
SATRIA DATA 2026

Category:
Big Data Challenge (BDC)

Theme:
Automatic Waste Classification using Machine Learning and Computer Vision

Official Evaluation Metric:
Macro F1 Score

---

# Project Objectives

The objectives of this project are:

- Build a robust image classification model.
- Compare multiple Deep Learning architectures.
- Evaluate model performance using Macro F1 Score.
- Produce reproducible Machine Learning experiments.
- Support intelligent waste sorting systems.

---

# Modeling Strategy

The project follows a staged experimentation strategy.

## Baseline

Custom CNN

Purpose:

- Establish baseline performance.
- Understand dataset behavior.

---

## Transfer Learning

Main candidate architectures:

- ResNet50
- EfficientNet-B0

Optional candidate:

- ConvNeXt-Tiny

Purpose:

- Improve classification performance.
- Leverage pretrained visual representations.

---

# Machine Learning Workflow

Business Understanding

↓

Data Collection

↓

Exploratory Data Analysis

↓

Data Preprocessing

↓

CNN Baseline

↓

Transfer Learning

↓

Model Evaluation

↓

Experiment Comparison

↓

Final Submission

---

# Project Structure

```text
Smart Waste Classification/

├── .claude/
├── configs/
├── data/
├── experiments/
├── notebooks/
├── outputs/
├── reports/
├── src/
│
├── CLAUDE.md
├── train.py
├── evaluate.py
├── predict.py
└── README.md
```

---

# Main Directories

## data/

Dataset storage.

```text
data/
├── train/
├── validation/
├── test/
├── raw/
└── processed/
```

---

## notebooks/

Experiment notebooks.

```text
01_business_understanding.ipynb
02_data_collection.ipynb
03_eda.ipynb
04_preprocessing.ipynb
05_cnn_baseline.ipynb
06_evaluation.ipynb
07_resnet50.ipynb
08_efficientnet.ipynb
09_comparison.ipynb
10_final_submission.ipynb
```

---

## src/

Source code implementation.

Contains:

- datasets
- preprocessing
- models
- training
- evaluation
- inference
- utilities

---

## configs/

Experiment configurations.

Examples:

- baseline.yaml
- resnet50.yaml
- efficientnet.yaml

---

## experiments/

Experiment tracking.

Examples:

```text
01_baseline/
02_resnet50/
03_resnet50_finetune/
04_efficientnet/
```

---

## outputs/

Generated outputs.

Examples:

- checkpoints
- figures
- logs
- reports
- submissions

---

# Evaluation Metrics

Primary Metric

- Macro F1 Score

Secondary Metrics

- Accuracy
- Precision
- Recall

Additional Analysis

- Classification Report
- Confusion Matrix
- Learning Curves
- Error Analysis

---

# Reproducibility

The project follows reproducible Machine Learning practices.

Features:

- Centralized configuration files
- Fixed random seeds
- Modular code structure
- Experiment tracking
- Automatic artifact saving

---

# Technologies

Programming Language

- Python

Deep Learning

- PyTorch

Data Processing

- NumPy
- Pandas

Visualization

- Matplotlib
- Seaborn

Evaluation

- Scikit-Learn

---

# Team Workflow

This repository integrates AI-assisted development using Claude Code.

Project instructions:

```text
CLAUDE.md
```

Workflow commands:

```text
.claude/commands/
```

Specialized knowledge modules:

```text
.claude/skills/
```

This setup ensures consistent implementation, experimentation, evaluation, and reporting throughout the competition lifecycle.

---

# Expected Deliverables

- Trained Models
- Experiment Logs
- Evaluation Results
- Competition Submission Files
- Technical Report

---

# License

This repository is developed exclusively for educational and competition purposes under SATRIA DATA 2026.