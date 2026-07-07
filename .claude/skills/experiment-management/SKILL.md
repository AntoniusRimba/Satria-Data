# SKILL: Experiment Management

---

# PURPOSE

Provide professional guidance for organizing, tracking, comparing, and documenting Machine Learning experiments throughout the entire project lifecycle.

This skill ensures that every experiment is reproducible, traceable, and scientifically comparable, enabling systematic model improvement and reliable competition submissions.

Experiment management is essential for AI research and competition environments such as SATRIA DATA 2026.

---

# CAPABILITY

This skill enables Claude to

- Organize experiments
- Track experiment history
- Compare multiple models
- Record configurations
- Maintain reproducibility
- Select the best candidate model
- Prepare experiments for submission

---

# WHEN TO USE

Use this skill

- before starting a new experiment
- after every completed training
- before changing hyperparameters
- before trying another architecture
- before submission

Every experiment should be recorded.

---

# BDC CONTEXT

Competition

SATRIA DATA 2026

Category

Big Data Challenge

Task

Image Classification

Goal

Develop the most robust image classification model while maintaining complete reproducibility and transparent experiment history.

---

# EXPERIMENT OBJECTIVES

Experiment management should answer

1.

What experiment was performed?

2.

What configuration was used?

3.

What changed from the previous experiment?

4.

Did performance improve?

5.

Why did performance improve?

6.

Which experiment should become the submission candidate?

---

# IMPLEMENTATION WORKFLOW

Follow this order.

---

## Step 1

Create Experiment ID

Every experiment must have

Experiment ID

Example

EXP001

EXP002

EXP003

Never overwrite previous experiments.

---

## Step 2

Record Experiment Metadata

Document

Experiment ID

Date

Research Objective

Research Notes

Researcher

Dataset Version

Framework Version

Git Commit (if available)

---

## Step 3

Record Model Configuration

Save

Architecture

Backbone

Input Size

Batch Size

Learning Rate

Optimizer

Scheduler

Epoch

Loss Function

Random Seed

Trainable Parameters

Frozen Layers (Transfer Learning)

---

## Step 4

Record Data Configuration

Document

Dataset Path

Train Split

Validation Split

Test Split

Augmentation Strategy

Normalization Strategy

Image Size

Class Distribution

---

## Step 5

Record Training Results

Save

Training Time

Best Epoch

Training Loss

Validation Loss

Accuracy

Precision

Recall

Macro F1

---

## Step 6

Record Evaluation Results

Store

Confusion Matrix

Classification Report

Learning Curve

Error Analysis

Model Size

Inference Time

---

## Step 7

Compare Experiments

Compare

CNN Baseline

↓

ResNet50

↓

EfficientNet-B0

↓

ConvNeXt-Tiny

Evaluate

- Macro F1

- Accuracy

- Training Time

- Parameters

- Inference Speed

Recommend the best experiment.

---

## Step 8

Document Improvements

For every new experiment explain

What changed?

Why was it changed?

Did it improve?

Should the change be kept?

Never modify multiple variables without documentation.

---

## Step 9

Submission Candidate

Select

Best Experiment

Explain

Why this experiment is selected.

Support the decision with evidence.

---

# EXPERIMENT LIFECYCLE

Every experiment should follow

Planning

↓

Configuration

↓

Training

↓

Evaluation

↓

Comparison

↓

Decision

↓

Documentation

↓

Next Experiment

Never skip documentation.

---

# EXPERIMENT TRACKING STANDARD

Every experiment must record

Experiment ID

Model

Configuration

Metrics

Visualizations

Artifacts

Conclusion

Recommendation

This enables complete reproducibility.

---

# INTERPRETATION GUIDELINES

Always explain

Why was this experiment performed?

What changed?

Did performance improve?

What should be attempted next?

Support every conclusion with evaluation results.

Avoid subjective judgments.

---

# RECOMMENDED EXPERIMENT STRATEGY

Stage 1

CNN Baseline

↓

Stage 2

ResNet50

↓

Stage 3

EfficientNet-B0

↓

Stage 4

ConvNeXt-Tiny

↓

Stage 5

Fine Tuning

↓

Stage 6

Hyperparameter Optimization

↓

Final Submission

Maintain one controlled change per experiment whenever possible.

---

# PROFESSIONAL CHECKLIST

Before finishing ensure

□ Experiment ID assigned

□ Configuration documented

□ Dataset version recorded

□ Hyperparameters recorded

□ Metrics saved

□ Evaluation completed

□ Visualizations saved

□ Experiment compared

□ Improvement documented

□ Recommendation produced

---

# COMMON MISTAKES

Never

- overwrite experiment folders

- lose configuration files

- compare experiments using different datasets

- change multiple variables simultaneously without documentation

- report metrics without experiment identifiers

- forget random seed

- forget experiment notes

---

# EXPECTED DELIVERABLES

Save outputs inside

outputs/experiments/

Recommended files

experiment_log.csv

experiment_config.json

experiment_summary.md

comparison_table.csv

leaderboard.csv

best_experiment.md

---

# CODING STANDARDS

Always

- Use versioned experiment folders

- Save configurations automatically

- Save metrics automatically

- Save figures automatically

- Separate experiments clearly

Never

- hardcode experiment names

- overwrite previous results

- mix experiment outputs

---

# BDC COMPLIANCE

Maintain complete transparency.

Every submitted model should be reproducible.

Every reported metric should correspond to a recorded experiment.

Never submit undocumented experiments.

---

# SUCCESS CRITERIA

Experiment Management is complete only if

✓ Every experiment has an ID

✓ Every configuration is documented

✓ Every metric is recorded

✓ Every visualization is archived

✓ Every improvement is justified

✓ The best experiment is selected using evidence

✓ Submission candidate is identified

✓ The experiment history is fully reproducible

Only then should the project proceed to Final Report and Competition Submission.