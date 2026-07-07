# SKILL: Image Exploratory Data Analysis (Image EDA)

---

# PURPOSE

Provide professional guidance for performing Exploratory Data Analysis (EDA) on image datasets used in Computer Vision projects.

This skill helps understand dataset quality, class distribution, image characteristics, and potential issues before preprocessing or model development.

This skill is intended for supervised image classification projects following best practices in Machine Learning research.

---

# CAPABILITY

This skill enables Claude to:

- Analyze image datasets systematically.
- Detect dataset quality issues.
- Identify risks before preprocessing.
- Produce reproducible EDA notebooks.
- Generate scientific interpretations instead of only visualizations.
- Support decisions for preprocessing and model selection.

---

# WHEN TO USE

Use this skill

- before preprocessing
- before training
- when receiving a new dataset
- after merging datasets
- before transfer learning
- before data augmentation

Never skip EDA.

---

# BDC CONTEXT

Competition

SATRIA DATA 2026

Category

Big Data Challenge

Problem

Automatic Waste Sorting

Task

Supervised Learning

Computer Vision

Image Classification

Classes

- Recyclable
- Electronic
- Organic

Objective

Develop an Intelligent Decision Module capable of recognizing waste material from images.

EDA should focus only on the official competition dataset.

Do NOT introduce external datasets.

---

# IMPLEMENTATION OBJECTIVES

The Image EDA stage should answer the following questions.

1.

How many images exist?

2.

How many classes exist?

3.

Are class labels balanced?

4.

Are there corrupted images?

5.

Are image dimensions consistent?

6.

Are image formats consistent?

7.

Is there class imbalance?

8.

Are duplicate images present?

9.

Do images contain significant background variation?

10.

Are there quality issues such as

- blur
- low resolution
- excessive brightness
- darkness
- occlusion

11.

Are there samples that appear mislabeled?

12.

Does the dataset require augmentation?

---

# IMPLEMENTATION WORKFLOW

Follow this order.

---

## Step 1

Dataset Overview

Analyze

- dataset structure
- folder hierarchy
- number of classes
- total images
- images per class

Output

Summary table

---

## Step 2

Class Distribution

Visualize

- class frequency
- percentage

Interpret

- balanced
- moderately imbalanced
- severely imbalanced

Recommend whether class balancing techniques may be required.

---

## Step 3

Image Properties

Analyze

- width
- height
- aspect ratio
- image channels
- image format
- color space

Identify

- inconsistent image size
- grayscale images
- alpha channel
- unexpected formats

---

## Step 4

Image Quality Inspection

Randomly display samples.

Inspect

- focus
- brightness
- contrast
- shadows
- background
- cropping

Discuss observations.

---

## Step 5

Corrupted Image Detection

Detect

- unreadable images
- broken files
- zero-byte images
- invalid formats

List corrupted files.

Recommend removal.

---

## Step 6

Duplicate Detection

Identify duplicate images.

Possible methods

- filename
- image hash
- perceptual hash

Report duplicate count.

---

## Step 7

Representative Sample Visualization

Display

multiple images from every class.

Purpose

Understand intra-class variation.

Observe

- object diversity
- background diversity
- viewpoint variation
- lighting variation

---

## Step 8

Dataset Insights

Summarize

- strengths
- weaknesses
- preprocessing implications

Explain

How EDA findings influence preprocessing.

---

# INTERPRETATION GUIDELINES

Never stop at visualization.

Always explain

Why does this matter?

Example

Incorrect

"The dataset contains 3 classes."

Correct

"The dataset contains three classes with relatively balanced distributions, reducing the need for aggressive resampling. However, moderate imbalance should still be considered during evaluation using Macro F1 Score."

Every visualization must include interpretation.

---

# VISUALIZATION STANDARD

Recommended figures

✓ Class Distribution

✓ Random Samples

✓ Image Resolution Distribution

✓ Aspect Ratio Distribution

✓ Image Dimension Histogram

✓ Duplicate Summary

Avoid unnecessary plots.

Visualization should support decisions.

---

# PROFESSIONAL CHECKLIST

Before finishing EDA ensure

□ Dataset loaded successfully

□ Folder structure verified

□ Class distribution analyzed

□ Image count verified

□ Image properties analyzed

□ Corrupted images checked

□ Duplicate images checked

□ Sample visualization completed

□ Dataset insights written

□ Recommendations for preprocessing produced

---

# COMMON MISTAKES

Never

- Jump directly to preprocessing.
- Ignore corrupted images.
- Ignore duplicate images.
- Ignore class imbalance.
- Produce plots without interpretation.
- Make unsupported assumptions.
- Modify the dataset during EDA.

EDA is analysis only.

---

# EXPECTED DELIVERABLES

Save outputs inside

outputs/eda/

Recommended files

dataset_summary.csv

class_distribution.png

sample_images.png

image_size_distribution.png

aspect_ratio_distribution.png

duplicate_report.csv

corrupted_images.csv

eda_summary.md

---

# CODING STANDARDS

Always

- Use pathlib.
- Use configurable paths.
- Use reusable functions.
- Separate analysis from visualization.
- Keep notebooks modular.
- Explain every analysis step before code.
- Interpret every result after execution.

Never

- Hardcode dataset paths.
- Mix preprocessing into EDA.
- Delete files during analysis.
- Duplicate code.

---

# BDC COMPLIANCE

Always comply with SATRIA DATA 2026 rules.

Only analyze

- official training dataset

Never

- inspect private test labels
- merge external datasets
- perform preprocessing during EDA

Maintain reproducibility.

---

# SUCCESS CRITERIA

Image EDA is complete only if

✓ Dataset structure understood

✓ Class distribution analyzed

✓ Image characteristics documented

✓ Data quality assessed

✓ Potential risks identified

✓ Preprocessing recommendations produced

Only then should the project proceed to Image Preprocessing.