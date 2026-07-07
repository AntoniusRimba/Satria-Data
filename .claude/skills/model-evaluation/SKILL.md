# SKILL: Model Evaluation

---

# PURPOSE

Provide professional guidance for evaluating Deep Learning image classification models following modern Machine Learning research practices.

This skill ensures that model evaluation is evidence-based, reproducible, and aligned with the official evaluation criteria of SATRIA DATA 2026.

The objective is not merely to compute metrics, but to assess whether the model generalizes well and is suitable for competition submission.

---

# CAPABILITY

This skill enables Claude to

- Evaluate trained models objectively
- Interpret evaluation metrics
- Analyze model generalization
- Detect overfitting and underfitting
- Compare multiple experiments
- Recommend the next experiment
- Select the best submission candidate

---

# WHEN TO USE

Use this skill

- after model training
- after transfer learning
- after fine tuning
- before submission
- before writing the experiment report

Never evaluate an untrained model.

---

# BDC CONTEXT

Competition

SATRIA DATA 2026

Category

Big Data Challenge

Problem

Automatic Waste Sorting

Task

Computer Vision

Image Classification

Classes

- Recyclable
- Electronic
- Organic

Official Evaluation Metric

Macro F1 Score

All evaluation decisions should prioritize Macro F1.

---

# IMPLEMENTATION OBJECTIVES

The evaluation stage should answer

1.

Does the model generalize well?

2.

Is Macro F1 satisfactory?

3.

Which classes are difficult?

4.

Does the model overfit?

5.

Which model performs best?

6.

Is the model ready for submission?

---

# IMPLEMENTATION WORKFLOW

Follow this order.

---

## Step 1

Load Evaluation Artifacts

Load

- Best checkpoint
- Validation dataset
- Test dataset (inference only)
- Training history
- Configuration

Verify

- model version
- dataset version
- experiment ID

---

## Step 2

Generate Predictions

Perform inference.

Collect

- predicted labels
- true labels
- prediction probabilities

Ensure reproducibility.

---

## Step 3

Primary Evaluation Metric

Compute

Macro F1 Score

Explain

- why Macro F1 is used
- why it is more appropriate than Accuracy
- relationship to class balance

This is the primary metric.

---

## Step 4

Secondary Metrics

Compute

- Accuracy
- Precision
- Recall

Also compute

Per-Class

- Precision
- Recall
- F1 Score

Interpret every metric.

Never report numbers without explanation.

---

## Step 5

Classification Report

Generate

Classification Report

Discuss

- strongest class
- weakest class
- imbalance effects
- precision-recall tradeoff

Focus on interpretation.

---

## Step 6

Confusion Matrix Analysis

Generate

Confusion Matrix

Identify

- most confused classes
- dominant errors
- systematic mistakes

Explain

possible causes

Examples

- similar object appearance
- background interference
- insufficient training samples

---

## Step 7

Learning Curve Analysis

Analyze

Training Loss

Validation Loss

Training Accuracy

Validation Accuracy

Macro F1

Determine

- healthy convergence
- overfitting
- underfitting
- unstable learning

Support conclusions with evidence.

---

## Step 8

Error Analysis

Inspect misclassified images.

Analyze

- object similarity
- lighting
- blur
- background
- occlusion
- annotation quality

Recommend improvements.

---

## Step 9

Model Comparison

Compare

- CNN Baseline
- ResNet50
- EfficientNet-B0
- ConvNeXt-Tiny

Compare

- Macro F1
- Accuracy
- Precision
- Recall
- Training Time
- Parameters
- Model Size
- Inference Time

Recommend the best model.

Selection must be evidence-based.

---

## Step 10

Submission Readiness

Determine

Is the model

Ready

or

Needs Improvement

Possible recommendations

- Fine Tuning

- Better Augmentation

- Hyperparameter Optimization

- Different Backbone

- More Training

Explain why.

---

# EVALUATION STANDARD

Always evaluate

Prediction

↓

Metrics

↓

Confusion Matrix

↓

Learning Curves

↓

Error Analysis

↓

Comparison

↓

Recommendation

Never skip interpretation.

---

# INTERPRETATION GUIDELINES

Always explain

Why did Macro F1 improve?

Why is one class weaker?

Why is the model overfitting?

Why should another architecture be tested?

Support every conclusion using evaluation evidence.

Avoid speculation.

---

# MODEL COMPARISON STANDARD

Compare models objectively.

Include

Model

Backbone

Parameters

Training Time

Macro F1

Accuracy

Inference Time

Memory Usage (if available)

Recommendation

The best model is

NOT

always the model with the highest Accuracy.

Prioritize

Macro F1

Generalization

Stability

Efficiency

---

# PROFESSIONAL CHECKLIST

Before finishing ensure

□ Predictions generated

□ Macro F1 computed

□ Secondary metrics computed

□ Classification Report generated

□ Confusion Matrix interpreted

□ Learning Curves analyzed

□ Error Analysis completed

□ Model Comparison completed

□ Recommendation produced

---

# COMMON MISTAKES

Never

- report only Accuracy

- ignore Macro F1

- skip Confusion Matrix

- skip Error Analysis

- compare only one metric

- ignore computational efficiency

- recommend a model without evidence

---

# EXPECTED DELIVERABLES

Save outputs inside

outputs/evaluation/

Recommended files

metrics.json

classification_report.csv

confusion_matrix.png

learning_curve.png

error_analysis.md

comparison_table.csv

evaluation_summary.md

---

# CODING STANDARDS

Always

- Use reusable evaluation functions

- Separate evaluation from training

- Save evaluation artifacts

- Use consistent metric calculations

- Keep evaluation reproducible

Never

- compute metrics manually when tested libraries exist

- duplicate evaluation logic

- modify model weights during evaluation

---

# BDC COMPLIANCE

Always comply with SATRIA DATA 2026.

Primary Metric

Macro F1 Score

Never

- tune hyperparameters using test data

- evaluate using training data only

- manipulate reported metrics

Maintain transparency.

---

# SUCCESS CRITERIA

Model Evaluation is complete only if

✓ Macro F1 reported

✓ Secondary metrics analyzed

✓ Classification Report interpreted

✓ Confusion Matrix interpreted

✓ Learning Curves analyzed

✓ Error Analysis completed

✓ Model Comparison completed

✓ Submission recommendation produced

Only then should the project proceed to Scientific Reporting or Submission.