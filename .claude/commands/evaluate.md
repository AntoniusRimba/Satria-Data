# COMMAND: /evaluate

# PURPOSE

Act as an AI Research Engineer responsible for evaluating the performance of image classification models developed for the SATRIA DATA 2026 Big Data Challenge.

The objective of this stage is not merely to compute evaluation metrics, but to determine whether the trained model generalizes well, satisfies the competition objectives, and is ready for submission or further improvement.

Always follow the rules defined in `CLAUDE.md`.

---

# Global Instruction

Always:

- Follow CLAUDE.md
- Follow BDC Competition Rules
- Follow the project folder structure
- Explain theory before implementation
- Do not jump directly into coding
- Produce modular and reusable code
- Save outputs into the correct directory
- Explain every important design decision
- Maintain reproducibility

---

# CONTEXT

Project

- SATRIA DATA 2026
- Big Data Challenge
- Automatic Waste Sorting

Task

- Computer Vision
- Supervised Learning
- Multi-Class Image Classification

Target Classes

- Recyclable
- Electronic
- Organic

Primary Competition Metric

- Macro F1 Score

---

# INPUT

Expected Input

- Trained Model
- Best Model Checkpoint
- Validation Dataset
- Test Dataset (if permitted for inference only)
- Training History
- Configuration File

Do not retrain the model during this stage.

---

# OBJECTIVES

Evaluate the trained model by answering the following questions.

1. Does the model generalize well?

2. Which evaluation metric best represents the model performance?

3. Which classes are difficult to classify?

4. Is the model overfitting or underfitting?

5. Is the model ready for competition submission?

6. What improvements should be attempted next?

---

# WORKFLOW

## Step 1 — Evaluation Preparation

Verify

- Correct model checkpoint
- Dataset partition
- Label consistency
- Evaluation configuration

Ensure evaluation uses the appropriate dataset.

Never evaluate using the training dataset as the primary performance indicator.

---

## Step 2 — Model Prediction

Generate predictions using the trained model.

Collect

- Predicted labels
- Ground truth labels
- Prediction probabilities (if available)

Ensure inference is reproducible.

---

## Step 3 — Performance Metrics

Compute evaluation metrics.

Primary Metric

- Macro F1 Score

Secondary Metrics

- Accuracy
- Precision
- Recall

Additional Metrics (when appropriate)

- Per-class Precision
- Per-class Recall
- Per-class F1

Explain

- Why Macro F1 Score is the primary metric.
- Why relying only on Accuracy may be misleading.

Interpret every metric.

Do not simply display numerical values.

---

## Step 4 — Confusion Matrix Analysis

Generate a Confusion Matrix.

Analyze

- Frequently confused classes
- Correct predictions
- Misclassification patterns

Identify

- Which waste categories are most challenging.
- Possible causes of confusion.

Support conclusions using evidence.

---

## Step 5 — Classification Report

Generate a Classification Report.

Analyze

- Precision per class
- Recall per class
- F1 per class
- Macro Average
- Weighted Average

Explain the meaning of each important result.

---

## Step 6 — Learning Curve Analysis

Analyze

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

Determine whether the model exhibits

- Healthy convergence
- Overfitting
- Underfitting
- Training instability

Explain the evidence.

---

## Step 7 — Error Analysis

Inspect incorrectly classified samples.

Analyze

- Common failure patterns
- Background influence
- Similar object appearance
- Image quality issues
- Dataset limitations

Discuss

- Why these errors may occur.
- Possible preprocessing or modeling improvements.

Avoid unsupported speculation.

---

## Step 8 — Model Comparison

If multiple experiments exist,

compare

- Baseline CNN
- ResNet50
- EfficientNet-B0
- ConvNeXt-Tiny
- Other candidate models

Compare using

- Macro F1
- Accuracy
- Training Time
- Model Size
- Computational Cost
- Inference Efficiency

Recommend the best model based on evidence.

Do not choose a model solely because it has the highest Accuracy.

---

## Step 9 — Competition Readiness Assessment

Determine whether the model is suitable for submission.

Consider

- Macro F1 performance
- Generalization
- Stability
- Computational efficiency
- Reproducibility

If performance is unsatisfactory,

recommend the next experiment.

Examples

- Fine-Tuning
- Hyperparameter Tuning
- Better Augmentation
- Architecture Improvement

---

# EXPECTED OUTPUT

This stage should produce

- Macro F1 Score
- Accuracy
- Precision
- Recall
- Classification Report
- Confusion Matrix
- Learning Curves
- Error Analysis
- Model Comparison
- Recommendation

---

# DELIVERABLES

Save outputs inside

outputs/

Recommended artifacts

- evaluation_metrics.json
- classification_report.csv
- confusion_matrix.png
- learning_curve.png
- error_analysis.md
- model_comparison.csv
- evaluation_summary.md

---

# BDC RULES

Always comply with SATRIA DATA 2026 regulations.

Never

- tune hyperparameters using the test dataset
- retrain during evaluation
- manipulate evaluation metrics
- select models based on test leakage

Maintain fairness and reproducibility.

Remember

The official competition metric is

Macro F1 Score.

All evaluation decisions should prioritize this metric.

---

# IMPLEMENTATION STYLE

Always

- Explain evaluation objectives before coding.
- Interpret every metric.
- Support conclusions using evidence.
- Compare experiments objectively.
- Present results using tables and visualizations.
- Save all evaluation artifacts.
- Maintain reproducibility.

Never

- Report only Accuracy.
- Ignore Macro F1 Score.
- Skip error analysis.
- Draw conclusions without supporting evidence.
- Recommend a model without comparison.

---

# COMPLETION CRITERIA

The evaluation stage is considered complete only if

✓ Macro F1 Score has been reported.

✓ Secondary metrics have been analyzed.

✓ Confusion Matrix has been interpreted.

✓ Classification Report has been discussed.

✓ Learning Curves have been analyzed.

✓ Error Analysis has been completed.

✓ Model comparison has been performed (if multiple experiments exist).

✓ A recommendation for the next experiment or competition submission has been produced.

Only then may the project proceed to the Reporting or Submission stage.