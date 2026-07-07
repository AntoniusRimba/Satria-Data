# SKILL: Transfer Learning

---

# PURPOSE

Provide professional guidance for implementing Transfer Learning in image classification tasks using pretrained Deep Learning architectures.

This skill enables efficient development of high-performance image classification models while maintaining reproducibility, fairness, and compliance with SATRIA DATA 2026.

The objective is not merely to use pretrained models, but to apply Transfer Learning following modern Computer Vision best practices.

---

# CAPABILITY

This skill enables Claude to

- Load pretrained architectures
- Replace classification heads
- Freeze pretrained backbone
- Fine-tune selected layers
- Configure Transfer Learning pipelines
- Compare multiple pretrained backbones
- Document backbone information properly

---

# WHEN TO USE

Use this skill

- after the baseline CNN experiment
- when experimenting with pretrained architectures
- when limited training data exists
- when improving model performance
- before hyperparameter tuning

Applicable to

- ResNet
- DenseNet
- EfficientNet
- MobileNet
- ConvNeXt
- Vision Transformer
- Other pretrained Computer Vision architectures

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

Competition Rules

Public pretrained models are allowed.

However,

the pretrained model

MUST NOT

have been previously trained using

- official BDC training images
- official BDC testing images

The backbone used must be documented in the final report.

---

# IMPLEMENTATION OBJECTIVES

Transfer Learning should answer

1.

Can pretrained visual features improve performance?

2.

Which pretrained backbone performs best?

3.

Should the backbone remain frozen?

4.

Should fine tuning be applied?

5.

Does Transfer Learning outperform the baseline CNN?

---

# IMPLEMENTATION WORKFLOW

Follow this order.

---

## Step 1

Select Backbone

Choose one pretrained architecture.

Examples

- ResNet50
- EfficientNet-B0
- ConvNeXt-Tiny

Document

- model name
- pretrained source
- pretrained dataset

---

## Step 2

Load Pretrained Weights

Load

official pretrained weights

Examples

- ImageNet pretrained

Do not

use unofficial checkpoints unless justified.

Always document the weight source.

---

## Step 3

Replace Classification Head

Remove

the original classifier.

Replace

with a classifier matching

3 output classes

- Recyclable
- Electronic
- Organic

Only the classifier should initially be trainable.

---

## Step 4

Freeze Backbone

Initially

freeze feature extraction layers.

Train

only the new classifier.

Purpose

Allow the classifier to adapt to the new task before modifying pretrained representations.

---

## Step 5

Initial Training

Train

classification head only.

Monitor

- Validation Loss
- Macro F1

Determine whether frozen features are sufficient.

---

## Step 6

Fine Tuning

If validation performance plateaus,

gradually unfreeze

selected backbone layers.

Fine tuning should be performed gradually.

Avoid unfreezing the entire backbone immediately unless justified.

Document

- layers unfrozen
- learning rate
- fine tuning strategy

---

## Step 7

Training Configuration

Use

- lower learning rate than baseline CNN

Possible optimizer

- AdamW
- SGD

Apply scheduler when appropriate.

---

## Step 8

Validation

Evaluate after every epoch.

Monitor

- Accuracy
- Precision
- Recall
- Macro F1

Macro F1 remains the primary metric.

---

## Step 9

Checkpoint

Save

- frozen model
- fine-tuned model
- configuration
- backbone information

Never overwrite checkpoints.

---

## Step 10

Experiment Comparison

Compare against

- Custom CNN
- Other pretrained architectures

Evaluate

- Macro F1
- Accuracy
- Training Time
- Number of Parameters
- Model Size
- Inference Time

Recommend the best backbone using evidence.

---

# BACKBONE DOCUMENTATION

Every pretrained experiment must document

Architecture

Pretrained Dataset

Weight Version

Framework

Input Resolution

Number of Parameters

Frozen Layers

Fine Tuned Layers

Output Classes

This documentation is mandatory for reproducibility.

---

# FINE TUNING GUIDELINES

Recommended strategy

Stage 1

↓

Freeze Backbone

↓

Train Classifier

↓

Evaluate

↓

Stage 2

↓

Unfreeze Final Block

↓

Fine Tune

↓

Evaluate

↓

Stage 3 (Optional)

↓

Unfreeze More Layers

↓

Fine Tune

↓

Final Evaluation

Avoid aggressive fine tuning from the beginning.

---

# INTERPRETATION GUIDELINES

Always explain

Why this backbone was selected.

Why Transfer Learning is appropriate.

Why certain layers were frozen.

Why fine tuning was performed.

Never report only numerical improvements.

Interpret

- convergence
- generalization
- computational cost

---

# PROFESSIONAL CHECKLIST

Before finishing ensure

□ Backbone documented

□ Pretrained weights documented

□ Classification head replaced

□ Backbone frozen initially

□ Fine tuning strategy documented

□ Learning rate adjusted

□ Validation completed

□ Macro F1 analyzed

□ Checkpoints saved

□ Experiment summary produced

---

# COMMON MISTAKES

Never

- train using test images

- use pretrained weights trained on BDC data

- unfreeze every layer immediately

- forget to replace classifier

- ignore backbone documentation

- compare models using Accuracy only

- omit Macro F1 analysis

---

# EXPECTED DELIVERABLES

Save outputs inside

outputs/transfer_learning/

Recommended files

backbone_info.json

best_transfer_model.pt

fine_tuned_model.pt

training_history.csv

transfer_learning_summary.md

model_comparison.csv

---

# CODING STANDARDS

Always

- Modular implementation

- Separate backbone from classifier

- Configurable freezing strategy

- Configurable learning rate

- Configurable pretrained weights

- Reusable transfer learning pipeline

Never

- hardcode backbone

- duplicate training code

- mix transfer learning with preprocessing

- modify pretrained weights manually

---

# BDC COMPLIANCE

Always comply with SATRIA DATA 2026.

Allowed

- Public pretrained models

- Fine Tuning

- Hyperparameter Tuning

Required

- Document backbone architecture

- Document pretrained source

Never

- use pretrained weights trained on BDC data

- use external labeled datasets for additional training

Maintain transparency and fairness.

---

# SUCCESS CRITERIA

Transfer Learning is complete only if

✓ Backbone documented

✓ Pretrained weights documented

✓ Classification head replaced

✓ Frozen training completed

✓ Fine tuning evaluated (if applied)

✓ Macro F1 reported

✓ Experiment compared against baseline

✓ Best pretrained model selected

Only then should the project proceed to Model Evaluation or Submission.