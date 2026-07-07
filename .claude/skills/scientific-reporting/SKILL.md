# SKILL: Scientific Reporting

---

# PURPOSE

Provide professional guidance for documenting Machine Learning experiments as scientific and technical reports suitable for AI competitions and research projects.

The objective is to transform experiment results into structured, reproducible, evidence-based documentation that clearly communicates methodology, findings, and conclusions.

This skill follows scientific writing principles rather than narrative storytelling.

---

# CAPABILITY

This skill enables Claude to

- Document Machine Learning workflows
- Explain methodological decisions
- Interpret experimental results
- Summarize findings objectively
- Produce competition-ready technical reports
- Produce reproducible documentation

---

# WHEN TO USE

Use this skill

- after experiment completion
- after evaluation
- after visualization
- before competition submission
- when updating experiment documentation

Scientific reporting should always be the final stage.

---

# BDC CONTEXT

Competition

SATRIA DATA 2026

Category

Big Data Challenge

Task

Computer Vision

Image Classification

Goal

Produce a transparent and reproducible technical report describing the complete Machine Learning pipeline.

---

# REPORT OBJECTIVES

Every report should answer

1.

What problem is being solved?

2.

Why was this approach selected?

3.

How was the model developed?

4.

How was the model evaluated?

5.

What evidence supports the conclusions?

6.

Why is this model recommended?

---

# REPORT STRUCTURE

Follow this order.

---

## 1.

Problem Background

Describe

- competition context
- environmental problem
- machine learning objective

Explain

why image classification is suitable.

---

## 2.

Problem Formulation

Define

Input

↓

Process

↓

Output

Explain

classification task

Target classes

Evaluation objective

---

## 3.

Dataset Description

Document

Dataset source

Number of images

Class labels

Class distribution

Image characteristics

Do not omit dataset limitations.

---

## 4.

Exploratory Data Analysis

Summarize

Dataset quality

Image distribution

Class balance

Potential challenges

Support statements using figures.

---

## 5.

Preprocessing

Describe

Cleaning

Resizing

Normalization

Augmentation

Dataset splitting

Explain

why each preprocessing step was applied.

---

## 6.

Model Development

Document

Architecture

Backbone

Training strategy

Transfer Learning (if applicable)

Fine Tuning (if applicable)

Configuration

Hyperparameters

Loss Function

Optimizer

Scheduler

Random Seed

Support methodological decisions.

---

## 7.

Experiment Design

Document

Baseline experiment

↓

Transfer Learning experiments

↓

Fine Tuning

↓

Comparison

Explain

what changed

between experiments.

---

## 8.

Evaluation

Present

Macro F1

Accuracy

Precision

Recall

Classification Report

Confusion Matrix

Learning Curves

Error Analysis

Interpret every result.

Avoid reporting numbers without explanation.

---

## 9.

Experiment Comparison

Compare

CNN

ResNet50

EfficientNet-B0

ConvNeXt-Tiny

Discuss

Strengths

Weaknesses

Computational Cost

Generalization

Recommendation

Support conclusions using evidence.

---

## 10.

Conclusion

Summarize

Main findings

Best architecture

Competition readiness

Future improvements

Do not introduce new information.

---

# WRITING PRINCIPLES

Always write

Objective

Scientific

Evidence-based

Reproducible

Concise

Professional

Avoid

Subjective opinions

Marketing language

Unsupported claims

---

# INTERPRETATION GUIDELINES

Every conclusion should be supported by

Evaluation Metrics

↓

Visualizations

↓

Experiment Comparison

↓

Scientific Reasoning

Never conclude using intuition alone.

---

# REPORT QUALITY STANDARD

The report should allow another researcher to

Understand

↓

Reproduce

↓

Verify

↓

Extend

the experiment independently.

---

# FIGURE INTEGRATION

Every figure should include

Figure Number

Title

Caption

Interpretation

Reference within the report

Do not insert figures without explanation.

---

# TABLE STANDARD

Tables should include

Configuration Summary

Hyperparameter Summary

Evaluation Metrics

Experiment Comparison

Submission Candidate

Use consistent formatting.

---

# PROFESSIONAL CHECKLIST

Before finishing ensure

□ Problem explained

□ Dataset documented

□ EDA summarized

□ Preprocessing documented

□ Model documented

□ Experiments documented

□ Metrics interpreted

□ Figures explained

□ Best model justified

□ Conclusions supported by evidence

---

# COMMON MISTAKES

Never

- describe code line-by-line

- report metrics without interpretation

- omit experiment configuration

- hide failed experiments

- exaggerate performance

- make unsupported claims

- write conclusions without evidence

---

# EXPECTED DELIVERABLES

Produce

Technical Report

Experiment Summary

Configuration Appendix

Model Comparison Table

Evaluation Summary

Figure Appendix

Artifact References

Ensure consistency with the submitted code.

---

# CODING STANDARDS

Documentation should always match

Code

Configuration

Outputs

Metrics

Figures

Experiments

Never document features that were not implemented.

---

# BDC COMPLIANCE

The report must

- document pretrained backbone (if used)

- explain methodology

- describe evaluation strategy

- support reproducibility

- remain consistent with submitted code

Maintain transparency throughout the report.

---

# SUCCESS CRITERIA

Scientific Reporting is complete only if

✓ The complete workflow is documented

✓ Every experiment is reproducible

✓ Every conclusion is evidence-based

✓ Every figure is interpreted

✓ Every metric is explained

✓ Every recommendation is justified

✓ The report is consistent with the submitted code

Only then is the project ready for final competition submission.