# SKILL: Deep Learning Training

---

# PURPOSE

Provide professional guidance for developing, training, validating, and managing Deep Learning image classification models following modern Machine Learning best practices.

This skill is designed to support both

- Custom CNN (Baseline)
- Transfer Learning
- Future Deep Learning architectures

while ensuring reproducibility, modularity, and compliance with SATRIA DATA 2026.

---

# CAPABILITY

This skill enables Claude to

- Build Deep Learning models
- Configure training pipelines
- Train reproducibly
- Monitor learning progress
- Save checkpoints
- Compare experiments
- Produce clean and modular implementations

---

# WHEN TO USE

Use this skill

- after preprocessing
- before evaluation
- whenever building a new architecture
- whenever changing hyperparameters

Applicable to

- CNN
- ResNet
- EfficientNet
- ConvNeXt
- Vision Transformer
- Other PyTorch image classification models

---

# BDC CONTEXT

Competition

SATRIA DATA 2026

Category

Big Data Challenge

Task

Computer Vision

Image Classification

Target Classes

- Recyclable
- Electronic
- Organic

Goal

Develop an Intelligent Decision Module capable of classifying waste images accurately while maximizing Macro F1 Score.

---

# IMPLEMENTATION OBJECTIVES

The training stage should answer

1.

Can the model learn meaningful visual features?

2.

Does the model converge properly?

3.

Does the model generalize well?

4.

Does the model overfit?

5.

Is the experiment reproducible?

6.

Is the implementation modular?

---

# IMPLEMENTATION WORKFLOW

Follow this order.

---

## Step 1

Experiment Configuration

Prepare

- random seed
- configuration
- paths
- output directory

Store every configuration in one place.

Example

CFG

config.py

yaml

Never hardcode hyperparameters.

---

## Step 2

Load Dataset

Receive

Training Loader

Validation Loader

Test Loader

Verify

- tensor shape

- labels

- batch size

before training.

---

## Step 3

Model Initialization

Initialize the selected architecture.

Possible models

- Custom CNN

- ResNet50

- EfficientNet-B0

- ConvNeXt-Tiny

Future support

- Vision Transformer

Document

- architecture

- number of parameters

- trainable parameters

---

## Step 4

Loss Function

Choose an appropriate loss.

Examples

CrossEntropyLoss

Focal Loss

Weighted CrossEntropy

Explain

why the selected loss is appropriate.

---

## Step 5

Optimizer

Configure

examples

Adam

AdamW

SGD

Document

- learning rate

- weight decay

- momentum

if applicable.

---

## Step 6

Learning Rate Scheduler

When appropriate

use scheduler

Examples

ReduceLROnPlateau

CosineAnnealingLR

StepLR

Document

scheduler behavior.

---

## Step 7

Training Loop

Implement

Epoch Loop

↓

Batch Loop

↓

Forward

↓

Loss

↓

Backward

↓

Optimizer Step

↓

Metric Update

Training loop should be modular.

Never duplicate logic.

---

## Step 8

Validation Loop

Evaluate after every epoch.

Compute

- Loss

- Accuracy

- Precision

- Recall

- Macro F1

Do not update gradients.

---

## Step 9

Checkpoint

Automatically save

Best Model

Latest Model

Training History

Configuration

Never overwrite without versioning.

---

## Step 10

Training Monitoring

Track

Training Loss

Validation Loss

Training Accuracy

Validation Accuracy

Macro F1

Learning Rate

Save history.

---

## Step 11

Early Stopping

Monitor

Validation Loss

or

Macro F1

Stop training

when improvement stagnates.

Document patience.

---

## Step 12

Experiment Summary

Summarize

Training Time

Best Epoch

Best Metric

Final Metric

Model Size

Inference Readiness

---

# TRAINING STANDARD

Training should always follow

Initialize

↓

Forward

↓

Loss

↓

Backward

↓

Optimizer

↓

Validation

↓

Checkpoint

↓

Next Epoch

Maintain the same workflow across experiments.

---

# INTERPRETATION GUIDELINES

Always explain

Why was this model selected?

Why was this optimizer selected?

Why was this learning rate selected?

Why did training stop?

Never report numbers without interpretation.

---

# REPRODUCIBILITY STANDARD

Always

Set random seed.

Document configuration.

Save checkpoints.

Save training history.

Save model configuration.

Maintain deterministic behavior when possible.

---

# EXPERIMENT MANAGEMENT

Every experiment must have

Experiment ID

Model Name

Date

Hyperparameters

Random Seed

Dataset Version

Evaluation Metric

Training Time

Checkpoint Location

This enables experiment comparison.

---

# PROFESSIONAL CHECKLIST

Before training ensure

□ Dataset verified

□ Configuration saved

□ Model initialized

□ Optimizer configured

□ Scheduler configured

□ Loss configured

□ Random seed fixed

□ Checkpoint enabled

□ Logging enabled

□ Output directory prepared

---

# COMMON MISTAKES

Never

- hardcode hyperparameters

- mix preprocessing with training

- evaluate using training data

- overwrite checkpoints

- ignore random seed

- train without validation

- ignore Macro F1

- duplicate training loop

---

# EXPECTED DELIVERABLES

Save outputs inside

outputs/models/

Recommended files

best_model.pt

last_model.pt

training_history.csv

training_curve.png

config.json

experiment_summary.md

training_log.csv

---

# CODING STANDARDS

Always

- Modular functions

- Configurable architecture

- pathlib

- typing

- logging

- reusable training loop

- reusable validation loop

- reusable checkpoint manager

Never

- hardcode paths

- duplicate code

- mix evaluation into training

- write monolithic notebooks

---

# BDC COMPLIANCE

Always comply with SATRIA DATA 2026.

Allowed

- Custom CNN

- Transfer Learning

- Fine Tuning

- Hyperparameter Tuning

Never

- train using test data

- tune using test labels

- use external labeled datasets

Maintain fairness.

---

# SUCCESS CRITERIA

Deep Learning Training is complete only if

✓ Configuration documented

✓ Model trained successfully

✓ Validation completed

✓ Checkpoints saved

✓ Training history saved

✓ Macro F1 monitored

✓ Best model selected

✓ Experiment documented

Only then should the project proceed to Model Evaluation.