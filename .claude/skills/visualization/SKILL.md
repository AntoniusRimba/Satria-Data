# SKILL: Visualization

---

# PURPOSE

Provide professional guidance for visualizing datasets, training processes, evaluation results, and experiment comparisons in Computer Vision projects.

Visualization should support analysis, interpretation, communication, and reproducibility rather than serving only aesthetic purposes.

All visualizations should help explain model behavior and support evidence-based decision making.

---

# CAPABILITY

This skill enables Claude to

- Visualize image datasets
- Visualize preprocessing results
- Monitor model training
- Present evaluation metrics
- Compare experiments visually
- Produce publication-quality figures
- Save reproducible visualization artifacts

---

# WHEN TO USE

Use this skill

- after preprocessing validation
- during model training
- after evaluation
- when comparing experiments
- when preparing reports

Visualization should accompany every important stage of the pipeline.

---

# BDC CONTEXT

Competition

SATRIA DATA 2026

Category

Big Data Challenge

Problem

Automatic Waste Sorting

Task

Image Classification

Target Classes

- Recyclable
- Electronic
- Organic

Visualization should communicate findings clearly for both technical reviewers and competition judges.

---

# VISUALIZATION OBJECTIVES

Every visualization should answer at least one question.

Examples

Can we understand the dataset?

↓

Can we monitor model learning?

↓

Can we identify model weaknesses?

↓

Can we compare experiments objectively?

Never generate figures without analytical purpose.

---

# IMPLEMENTATION WORKFLOW

Follow this order.

---

## Step 1

Dataset Visualization

Display

- sample images
- class examples
- image dimensions
- augmentation preview

Purpose

Verify dataset quality before training.

---

## Step 2

Preprocessing Visualization

Visualize

Original Image

↓

Resized Image

↓

Normalized Image

↓

Augmented Image

Purpose

Verify preprocessing correctness.

---

## Step 3

Training Visualization

Generate

Training Loss Curve

Validation Loss Curve

Training Accuracy Curve

Validation Accuracy Curve

Macro F1 Curve (if recorded)

Purpose

Observe convergence.

Detect

- overfitting
- underfitting
- unstable optimization

---

## Step 4

Evaluation Visualization

Generate

Confusion Matrix

Classification Performance

Per-Class Metrics

Purpose

Understand model strengths and weaknesses.

---

## Step 5

Error Analysis Visualization

Display

Misclassified Images

For every error include

- Ground Truth

- Prediction

- Confidence Score (if available)

Explain

possible causes.

---

## Step 6

Model Comparison Visualization

Compare experiments.

Possible charts

- Macro F1 comparison

- Accuracy comparison

- Training time comparison

- Parameter comparison

- Inference time comparison

Purpose

Support architecture selection.

---

## Step 7

Final Summary Visualization

Prepare

publication-ready figures

for

- notebook

- report

- presentation

Ensure consistent layout and labeling.

---

# RECOMMENDED VISUALIZATIONS

Dataset

✓ Sample Images

✓ Class Distribution

✓ Image Resolution Distribution

Preprocessing

✓ Before vs After

✓ Augmentation Preview

Training

✓ Loss Curve

✓ Accuracy Curve

✓ Macro F1 Curve

Evaluation

✓ Confusion Matrix

✓ Per-Class Performance

Error Analysis

✓ Misclassified Samples

Experiment

✓ Model Comparison Chart

---

# INTERPRETATION GUIDELINES

Never display figures without explanation.

Always explain

What does this figure show?

↓

Why is it important?

↓

What conclusion can be drawn?

↓

What should be improved?

Interpretation is more valuable than the figure itself.

---

# VISUALIZATION STANDARD

Every figure should contain

- Title

- Axis Labels

- Legend (when appropriate)

- Readable font size

- Consistent style

- Meaningful caption

Avoid cluttered figures.

One figure should communicate one main idea.

---

# PUBLICATION QUALITY

Figures should be

- clean

- readable

- high resolution

- consistent

- reproducible

Suitable for

- notebook

- technical report

- presentation

---

# PROFESSIONAL CHECKLIST

Before finishing ensure

□ Dataset visualized

□ Preprocessing verified visually

□ Training curves generated

□ Confusion Matrix generated

□ Error Analysis visualized

□ Experiment comparison visualized

□ Figures interpreted

□ Figures saved

---

# COMMON MISTAKES

Never

- generate unnecessary figures

- use unreadable labels

- overload one figure with too much information

- ignore interpretation

- compare experiments using inconsistent scales

- leave visualization undocumented

---

# EXPECTED DELIVERABLES

Save outputs inside

outputs/visualization/

Recommended files

dataset_preview.png

augmentation_preview.png

training_curve.png

loss_curve.png

macro_f1_curve.png

confusion_matrix.png

misclassified_samples.png

model_comparison.png

visualization_summary.md

---

# CODING STANDARDS

Always

- Save every figure automatically

- Use reusable plotting functions

- Keep visualization separate from training logic

- Generate deterministic figures

Never

- hardcode save paths

- duplicate plotting code

- mix visualization with model definition

---

# BDC COMPLIANCE

Visualization must accurately represent experimental results.

Never

- manipulate figures

- hide poor-performing metrics

- present misleading comparisons

Maintain transparency and scientific integrity.

---

# SUCCESS CRITERIA

Visualization is complete only if

✓ Dataset visualized

✓ Preprocessing validated visually

✓ Training curves generated

✓ Evaluation figures generated

✓ Error Analysis visualized

✓ Experiment comparison completed

✓ All figures interpreted

✓ Visualization artifacts saved

Only then should the project proceed to Experiment Management and Scientific Reporting.