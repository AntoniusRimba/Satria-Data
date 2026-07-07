# COMMAND: /preprocessing

# PURPOSE

Act as an AI Research Engineer assisting the Data Preprocessing stage for the SATRIA DATA 2026 Big Data Challenge project.

The objective of this stage is to transform the raw image dataset into a high-quality, standardized, and reproducible dataset suitable for machine learning model training.

Every preprocessing decision must be supported by findings from the previous Exploratory Data Analysis (EDA).

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

- Big Data Challenge 2026
- Image Classification for Automatic Waste Sorting

Task

- Supervised Learning
- Computer Vision
- Image Classification

Target Classes

- Recyclable
- Electronic
- Organic

---

# INPUT

Expected input

- Raw dataset
- EDA report
- Dataset directory structure

Do not begin preprocessing before understanding the EDA findings.

---

# OBJECTIVES

Perform preprocessing to answer the following questions.

1. What problems discovered during EDA need to be addressed?

2. How should the images be standardized for model training?

3. What preprocessing is truly necessary?

4. How can preprocessing improve model generalization without causing data leakage?

5. Is the dataset ready for CNN and Transfer Learning models?

---

# WORKFLOW

## Step 1 — Review EDA Findings

Review all important findings from the EDA stage.

Identify

- Image quality issues
- Class imbalance
- Resolution inconsistency
- Aspect ratio variation
- Dataset cleanliness
- Potential risks

Explain why each finding requires preprocessing.

Do not repeat analyses already completed during EDA.

---

## Step 2 — Data Cleaning

Inspect whether cleaning is required.

Possible tasks include

- Remove corrupted images
- Remove unreadable files
- Remove duplicated files (if applicable)
- Verify class folder consistency

Always explain why data cleaning is necessary.

Never modify class labels.

---

## Step 3 — Image Standardization

Determine how images should be standardized.

Possible preprocessing includes

- Image resizing
- Color space consistency
- Tensor conversion
- Pixel normalization

Explain

- Why each transformation is required
- Expected impact on model learning

Avoid unnecessary preprocessing.

---

## Step 4 — Data Augmentation

Determine whether augmentation is required.

Possible augmentation techniques include

- Horizontal Flip
- Rotation
- Random Crop
- Random Resize
- Brightness Adjustment
- Contrast Adjustment
- Color Jitter
- Random Erasing

Explain

- Why each augmentation is selected
- Which augmentation should not be used for this dataset
- Expected impact on generalization

Only apply augmentation to the training dataset.

Never augment validation or test datasets.

---

## Step 5 — Dataset Splitting

Prepare dataset partitions.

Split the dataset into

- Training Set
- Validation Set
- Test Set

Explain

- Split ratio
- Why the ratio is selected
- How data leakage is prevented

Ensure class distribution remains representative.

---

## Step 6 — Data Loading Pipeline

Prepare the input pipeline for model training.

Design

- Dataset Loader
- Transformation Pipeline
- Batch Processing
- Data Shuffling

Ensure the pipeline is modular and reusable.

---

## Step 7 — Validation of Preprocessing

Verify the preprocessing results.

Inspect

- Image dimensions
- Data type
- Pixel value range
- Label consistency
- Dataset balance after splitting

Visualize processed samples.

Confirm the processed images remain representative of the original dataset.

---

## Step 8 — Preprocessing Summary

Summarize all preprocessing decisions.

Explain

- What was performed
- Why it was necessary
- Expected influence on CNN training
- Expected influence on Transfer Learning models

Support every decision with evidence from EDA.

---

# EXPECTED OUTPUT

This stage should produce

- Clean dataset
- Standardized images
- Data augmentation pipeline
- Dataset split
- Data loading pipeline
- Preprocessing summary

---

# DELIVERABLES

Save outputs inside

data/

processed/

and

outputs/

Recommended artifacts include

- preprocessing configuration
- transformation summary
- processed sample visualization
- preprocessing report

---

# BDC RULES

Always comply with SATRIA DATA 2026 regulations.

Never

- use test data to determine preprocessing decisions
- apply augmentation to validation or test datasets
- modify original raw data
- introduce external labeled datasets
- create data leakage between dataset partitions

Maintain reproducibility throughout the preprocessing stage.

---

# IMPLEMENTATION STYLE

Always

- Explain preprocessing decisions before implementation.
- Base every preprocessing step on EDA findings.
- Build reusable preprocessing pipelines.
- Keep preprocessing configurable.
- Preserve reproducibility.
- Follow the project folder structure.
- Save preprocessing configurations.

Never

- Apply preprocessing without justification.
- Hardcode dataset paths.
- Duplicate preprocessing logic.
- Perform unnecessary image transformations.
- Continue to modeling before validating preprocessing outputs.

---

# COMPLETION CRITERIA

The preprocessing stage is considered complete only if

✓ All preprocessing decisions are supported by EDA.

✓ Images have been standardized.

✓ Training, validation, and test datasets have been prepared correctly.

✓ Data leakage has been prevented.

✓ Data augmentation is correctly applied only to the training set.

✓ The preprocessing pipeline is modular, reusable, and reproducible.

✓ The dataset is fully prepared for the modeling stage.

Only then may the project proceed to model development.