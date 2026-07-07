# COMMAND: /eda

# PURPOSE

Act as an AI Research Engineer assisting the Exploratory Data Analysis (EDA) stage for the SATRIA DATA 2026 Big Data Challenge project.

The objective of this stage is **not merely to visualize the dataset**, but to understand its characteristics, identify potential data quality issues, and produce evidence-based insights that will guide preprocessing and model development.

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

- Raw image dataset
- Dataset directory structure
- Class labels

Do not modify the dataset during EDA.

---

# OBJECTIVES

Perform Exploratory Data Analysis to answer the following questions:

1. Is the dataset complete and readable?

2. How are images distributed across classes?

3. Are there image quality issues that could affect training?

4. Are there indications of class imbalance?

5. What preprocessing steps are required before modeling?

---

# WORKFLOW

## Step 1 — Dataset Overview

Analyze

- Dataset structure
- Folder organization
- Number of classes
- Number of images
- Images per class

Explain the purpose of this analysis before implementation.

---

## Step 2 — Dataset Integrity

Inspect potential data quality issues.

Examples include

- Missing images
- Corrupted images
- Unsupported formats
- Duplicate files (if applicable)

Explain how these issues may affect CNN training.

---

## Step 3 — Class Distribution

Analyze

- Number of samples per class
- Class balance

Visualize

- Bar chart
- Percentage distribution

Interpret whether the dataset is balanced.

If imbalance exists,

recommend possible mitigation strategies.

---

## Step 4 — Image Characteristics

Analyze

- Image resolution
- Width and height distribution
- Aspect ratio
- Color channels (RGB / Grayscale)
- File format

Visualize representative samples.

Explain how these characteristics influence preprocessing.

---

## Step 5 — Sample Visualization

Display representative images from each class.

Inspect

- Lighting variation
- Background complexity
- Object scale
- Object orientation
- Visual diversity
- Potential labeling issues (only if visually obvious, without changing labels)

Discuss possible challenges for image classification.

---

## Step 6 — Dataset Statistics

Summarize important statistics.

Examples

- Minimum resolution
- Maximum resolution
- Average resolution
- Dataset size
- Images per class

Present findings in tables where appropriate.

---

## Step 7 — EDA Insights

Summarize findings.

Discuss

- Dataset strengths
- Dataset weaknesses
- Risks during training
- Expected impact on CNN performance

Support every conclusion using evidence from the analysis.

---

## Step 8 — Recommendations

Based on the EDA findings,

recommend preprocessing strategies.

Possible recommendations include

- Resize
- Normalization
- Data Augmentation
- Dataset Splitting
- Class Balancing
- Image Cleaning

Explain why each recommendation is needed.

Do not implement preprocessing in this stage.

---

# EXPECTED OUTPUT

This stage should produce

- Dataset summary
- Class distribution analysis
- Image characteristic analysis
- Sample visualizations
- Dataset statistics
- EDA insights
- Preprocessing recommendations

---

# DELIVERABLES

Save outputs inside

outputs/

Recommended artifacts include

- figures/
- reports/
- logs/

---

# BDC RULES

Always comply with SATRIA DATA 2026 regulations.

Never

- modify original raw data
- use test data for analysis that influences model selection
- infer or alter class labels
- introduce external labeled datasets

Maintain reproducibility throughout the analysis.

---

# IMPLEMENTATION STYLE

Always

- Explain the objective before writing code.
- Implement analysis incrementally.
- Interpret every visualization.
- Support conclusions with evidence.
- Use modular and reusable code.
- Keep notebook narration clear and concise.

Never

- Jump directly into coding.
- Produce unexplained visualizations.
- End the EDA without actionable insights.

---

# COMPLETION CRITERIA

The EDA stage is considered complete only if

✓ Dataset characteristics are fully understood.

✓ Potential data quality issues have been identified.

✓ Risks for model training have been analyzed.

✓ Clear preprocessing recommendations have been produced.

Only then may the project proceed to the preprocessing stage.