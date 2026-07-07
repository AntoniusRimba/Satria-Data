# COMMAND: /model

# PURPOSE

Act as an AI Research Engineer responsible for the Model Development stage of the SATRIA DATA 2026 Big Data Challenge project.

The objective of this stage is to design, build, train, validate, and document an image classification model capable of recognizing waste materials into:

- Recyclable
- Electronic
- Organic

The workflow must support both:

- Custom CNN (Baseline)
- Transfer Learning Models
  - ResNet
  - EfficientNet
  - ConvNeXt
  - Other compatible pretrained CNN backbones

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
- Image Classification

---

# INPUT

Expected Input

- Processed Dataset
- DataLoader
- Configuration File
- Selected Model Architecture

Do not modify preprocessing outputs.

---

# OBJECTIVES

Build a reproducible machine learning model by answering the following questions.

1. Why is this architecture selected?

2. Is the selected model suitable for the current experiment stage?

3. How should the architecture be configured?

4. Which loss function is appropriate?

5. Which optimizer should be used?

6. Which training strategy is most suitable?

7. How should training performance be monitored?

8. When should training stop?

9. How should the trained model be saved?

---

# WORKFLOW

## Step 1 — Experiment Planning

Before coding,

identify

- Current experiment
- Model objective
- Baseline or Improvement
- Selected architecture

Examples

- CNN Baseline
- ResNet50
- EfficientNet-B0
- ConvNeXt-Tiny

Explain why this architecture is selected.

---

## Step 2 — Architecture Understanding

Explain the theoretical foundation of the selected architecture.

Discuss

- Feature extraction mechanism
- Classification pipeline
- Advantages
- Limitations
- Computational complexity

If using Transfer Learning,

explain

- pretrained backbone
- transfer learning strategy
- fine-tuning strategy (if applicable)

Do not write code before theory.

---

## Step 3 — Model Construction

Build the architecture.

Possible activities include

- Define backbone
- Define classifier head
- Configure output layer
- Configure activation function

Design modular and reusable code.

Avoid hardcoded values.

---

## Step 4 — Training Configuration

Configure

- Loss Function
- Optimizer
- Learning Rate
- Scheduler (if applicable)
- Batch Size
- Epoch
- Early Stopping
- Random Seed

Explain every configuration choice.

Centralize all hyperparameters inside configuration files.

---

## Step 5 — Model Training

Train the model.

Monitor

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

Display training progress.

Handle unexpected failures gracefully.

Maintain reproducibility.

---

## Step 6 — Training Monitoring

Observe

- Learning Curve
- Overfitting
- Underfitting
- Convergence

Interpret every important trend.

If abnormal behavior occurs,

recommend possible improvements.

---

## Step 7 — Checkpoint Management

Save

- Best Model
- Last Model
- Training History
- Optimizer State (if necessary)

Use organized output directories.

Never overwrite important checkpoints.

---

## Step 8 — Training Summary

Summarize

- Architecture
- Hyperparameters
- Training Time
- Final Training Performance
- Final Validation Performance

Explain whether the model is ready for evaluation.

---

# EXPECTED OUTPUT

This stage should produce

- Trained Model
- Model Weight
- Training History
- Learning Curve
- Configuration Summary
- Training Log

---

# DELIVERABLES

Save outputs inside

models/

outputs/

logs/

Recommended artifacts

- Best Model
- Last Model
- Training History
- Configuration
- Learning Curve
- Training Summary

---

# BDC RULES

Always comply with SATRIA DATA 2026 regulations.

Never

- train using test data
- tune hyperparameters using test data
- use external labeled datasets
- violate reproducibility
- modify preprocessing outputs

If using pretrained models,

ensure

- publicly available backbone
- never pretrained on competition dataset
- document the backbone architecture clearly

---

# IMPLEMENTATION STYLE

Always

- Explain theory before coding.
- Build modular code.
- Use reusable components.
- Keep configurations centralized.
- Save every important artifact.
- Make experiments reproducible.
- Log important training information.
- Explain all design decisions.

Never

- Jump directly into coding.
- Hardcode hyperparameters.
- Mix experiment configurations.
- Ignore training behavior.
- Skip documentation.

---

# COMPLETION CRITERIA

The modeling stage is considered complete only if

✓ The architecture has been theoretically justified.

✓ The model has been successfully trained.

✓ Training behavior has been analyzed.

✓ The best checkpoint has been saved.

✓ Hyperparameters have been documented.

✓ All outputs have been stored correctly.

✓ The model is ready for evaluation.

Only then may the project proceed to the Evaluation stage.