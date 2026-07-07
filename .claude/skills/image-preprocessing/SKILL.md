# SKILL: Image Preprocessing

---

# PURPOSE

Provide professional guidance for preparing image datasets before model training.

This skill transforms raw images into standardized, clean, and reproducible inputs suitable for Deep Learning models while preserving fairness and compliance with SATRIA DATA 2026.

The preprocessing pipeline should be reusable across all candidate models.

---

# CAPABILITY

This skill enables Claude to:

- Build a reproducible preprocessing pipeline.
- Standardize image formats.
- Apply appropriate image transformations.
- Prepare datasets for CNN and Transfer Learning.
- Construct DataLoader efficiently.
- Separate training, validation, and testing workflows.
- Prevent data leakage.

---

# WHEN TO USE

Use this skill

- after Image EDA
- before model training
- before transfer learning
- before hyperparameter tuning

Never preprocess before understanding the dataset.

---

# BDC CONTEXT

Competition

SATRIA DATA 2026

Category

Big Data Challenge

Task

Computer Vision

Image Classification

Classes

- Recyclable
- Electronic
- Organic

Goal

Prepare images for robust and reproducible image classification without violating competition rules.

---

# IMPLEMENTATION OBJECTIVES

The preprocessing stage should answer the following questions.

1.

Should corrupted images be removed?

2.

Should image sizes be standardized?

3.

Should normalization be applied?

4.

Should augmentation be performed?

5.

Should augmentation be applied to every dataset split?

6.

How should the dataset be split?

7.

How should DataLoader be configured?

8.

How can data leakage be prevented?

---

# IMPLEMENTATION WORKFLOW

Follow this order.

---

## Step 1

Dataset Cleaning

Verify

- corrupted images
- invalid formats
- unreadable files
- duplicate images (if identified during EDA)

Remove only invalid data.

Document every removal.

---

## Step 2

Dataset Organization

Verify

- folder hierarchy
- class labels
- image naming consistency

Ensure every image belongs to the correct class.

---

## Step 3

Dataset Split

Split the dataset into

- Training Set
- Validation Set
- Test Set (if applicable)

Use

- stratified split whenever possible

Maintain class distribution.

Never use the test set during training or hyperparameter tuning.

---

## Step 4

Image Standardization

Standardize

- image size
- color channels
- tensor format

Examples

224 × 224

or

model-specific input size

Maintain consistency throughout the experiment.

---

## Step 5

Normalization

Normalize pixel values.

Apply normalization suitable for the selected model.

Examples

Custom CNN

- Scale pixel values to [0,1]

Transfer Learning

- Use pretrained normalization statistics (e.g., ImageNet mean and standard deviation)

Ensure consistency between training and inference.

---

## Step 6

Data Augmentation

Apply augmentation only to the Training Set.

Recommended augmentations

- Horizontal Flip
- Random Rotation
- Random Crop
- Random Resized Crop
- Color Jitter
- Random Brightness / Contrast
- Random Affine

Use augmentation conservatively.

Avoid transformations that change the semantic meaning of waste objects.

Never augment

- Validation Set
- Test Set

---

## Step 7

DataLoader Construction

Build DataLoader for

- Training
- Validation
- Testing

Configure

- batch size
- shuffle
- num_workers
- pin_memory (if supported)

Ensure reproducibility.

---

## Step 8

Pipeline Validation

Verify

- image dimensions
- tensor shapes
- class labels
- augmentation outputs

Visualize several transformed images.

Ensure preprocessing behaves as expected.

---

# INTERPRETATION GUIDELINES

Always explain

Why was this preprocessing step selected?

Examples

Correct

"Image resizing is required because CNN architectures expect a fixed input dimension."

Correct

"Normalization stabilizes optimization and accelerates convergence."

Correct

"Data augmentation improves generalization by exposing the model to more diverse visual variations."

Avoid applying transformations without justification.

---

# PREPROCESSING STANDARDS

Recommended order

Dataset Cleaning

↓

Dataset Split

↓

Resize

↓

Normalization

↓

Augmentation (Training Only)

↓

Tensor Conversion

↓

DataLoader

Maintain the same preprocessing pipeline for every experiment unless intentionally evaluating preprocessing strategies.

---

# DATA LEAKAGE PREVENTION

Always ensure

- Validation data is never augmented.
- Test data is never used during training.
- Hyperparameter tuning uses only the validation set.
- Duplicate images do not appear across different dataset splits.

Document the prevention strategy.

---

# PROFESSIONAL CHECKLIST

Before finishing preprocessing ensure

□ Invalid images removed

□ Dataset split completed

□ Class distribution preserved

□ Image size standardized

□ Normalization applied

□ Augmentation configured

□ DataLoader tested

□ Tensor shapes verified

□ Data leakage prevented

□ Pipeline documented

---

# COMMON MISTAKES

Never

- Apply augmentation to validation data.
- Apply augmentation to test data.
- Normalize inconsistently across experiments.
- Resize images differently within the same experiment.
- Use random split without preserving class balance.
- Ignore preprocessing reproducibility.
- Mix preprocessing logic inside the training loop.

---

# EXPECTED DELIVERABLES

Save outputs inside

outputs/preprocessing/

Recommended files

dataset_split.csv

preprocessing_config.json

augmentation_preview.png

tensor_shape_report.md

preprocessing_summary.md

---

# CODING STANDARDS

Always

- Use configurable preprocessing pipelines.
- Keep preprocessing independent from model code.
- Use reusable transformation functions.
- Use pathlib.
- Document preprocessing configuration.
- Validate preprocessing outputs before training.

Never

- Hardcode image paths.
- Duplicate preprocessing logic.
- Modify original datasets.
- Mix preprocessing with training logic.

---

# BDC COMPLIANCE

Always comply with SATRIA DATA 2026 rules.

Allowed

- Image resizing
- Normalization
- Data augmentation
- Stratified splitting

Never

- Use external labeled datasets.
- Use test images for tuning.
- Introduce data leakage.
- Modify official labels.

Maintain transparency and reproducibility.

---

# SUCCESS CRITERIA

Image Preprocessing is complete only if

✓ Dataset cleaned

✓ Dataset split completed

✓ Image standardization completed

✓ Normalization configured

✓ Augmentation implemented correctly

✓ DataLoader validated

✓ Data leakage prevented

✓ Preprocessing documented

Only then should the project proceed to Model Development.