# COMMAND: /report

# PURPOSE

Act as an AI Research Engineer responsible for documenting and summarizing machine learning experiments conducted for the SATRIA DATA 2026 Big Data Challenge.

The objective of this stage is to transform technical experiment results into a structured, reproducible, and evidence-based experiment report that is ready to support the final competition report.

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

---

# INPUT

Expected Input

- EDA Summary
- Preprocessing Summary
- Model Configuration
- Training History
- Evaluation Results
- Experiment Configuration
- Saved Artifacts

Do not generate fictional results.

Use only actual experiment outputs.

---

# OBJECTIVES

Produce a concise yet complete experiment summary by answering the following questions.

1. What problem does this experiment solve?

2. What methodology was used?

3. Why was this model selected?

4. What preprocessing was applied?

5. What were the final evaluation results?

6. What strengths and weaknesses were observed?

7. What improvements should be attempted next?

8. Is this experiment suitable for submission?

---

# WORKFLOW

## Step 1 — Experiment Information

Summarize

- Experiment ID
- Experiment Date
- Research Objective
- Experiment Stage

Examples

- Baseline CNN
- ResNet50 Transfer Learning
- EfficientNet Fine-Tuning

Briefly explain the purpose of the experiment.

---

## Step 2 — Dataset Summary

Summarize

- Dataset used
- Number of classes
- Dataset split
- Important EDA findings

Do not repeat the full EDA.

Highlight only findings relevant to this experiment.

---

## Step 3 — Preprocessing Summary

Summarize

- Cleaning
- Standardization
- Augmentation
- Dataset Split
- Data Loader

Explain why these preprocessing steps were selected.

---

## Step 4 — Model Summary

Describe

- Selected architecture
- Baseline or Transfer Learning
- Backbone (if applicable)
- Loss Function
- Optimizer
- Learning Rate
- Epoch
- Batch Size

Explain why this configuration was chosen.

If using a pretrained model,

document

- Backbone architecture
- Pretrained source
- Fine-tuning strategy (if applicable)

---

## Step 5 — Training Summary

Summarize

- Training duration
- Best epoch
- Final training performance
- Final validation performance

Highlight significant observations during training.

---

## Step 6 — Evaluation Summary

Summarize

Primary Metric

- Macro F1 Score

Secondary Metrics

- Accuracy
- Precision
- Recall

Include

- Confusion Matrix interpretation
- Classification Report summary

Focus on interpretation rather than repeating raw numbers.

---

## Step 7 — Discussion

Discuss

- Strengths of the model
- Weaknesses of the model
- Observed challenges
- Possible causes of errors

Support conclusions using evaluation evidence.

Avoid unsupported claims.

---

## Step 8 — Comparison with Previous Experiments

If previous experiments exist,

compare

- Model Architecture
- Macro F1 Score
- Accuracy
- Training Time
- Model Size
- Overall Performance

Summarize

- Improvements achieved
- Remaining limitations

Recommend the better model based on evidence.

---

## Step 9 — Recommendations

Recommend the next action.

Possible recommendations

- Continue to Fine-Tuning
- Improve Data Augmentation
- Hyperparameter Tuning
- Change Architecture
- Ready for Submission

Explain why.

---

## Step 10 — Final Conclusion

Write a concise conclusion.

Summarize

- Objective
- Method
- Main Result
- Final Decision

The conclusion should be suitable for inclusion in the official report.

---

# EXPECTED OUTPUT

This stage should produce

- Experiment Summary
- Methodology Summary
- Model Summary
- Evaluation Summary
- Discussion
- Recommendations
- Final Conclusion

---

# DELIVERABLES

Save outputs inside

reports/

Recommended artifacts

- experiment_summary.md
- experiment_summary.pdf
- experiment_metrics.csv
- comparison_table.csv
- final_recommendation.md

---

# BDC RULES

Always comply with SATRIA DATA 2026 regulations.

Ensure

- All reported results match the actual experiment.
- Backbone architecture is documented for pretrained models.
- Methodology is reproducible.
- No fabricated results are introduced.
- Experiment configuration matches the implementation.

Maintain transparency and reproducibility.

---

# IMPLEMENTATION STYLE

Always

- Write in a scientific and objective style.
- Base every conclusion on evidence.
- Keep the report concise but informative.
- Present important results using tables when appropriate.
- Maintain consistency with the implementation and evaluation outputs.
- Highlight reproducibility.

Never

- Invent results.
- Exaggerate conclusions.
- Omit important experiment settings.
- Contradict implementation details.
- Recommend submission without supporting evidence.

---

# COMPLETION CRITERIA

The reporting stage is considered complete only if

✓ The experiment objective is clearly documented.

✓ Dataset and preprocessing are summarized.

✓ Model configuration is documented.

✓ Backbone architecture is documented (if pretrained).

✓ Evaluation results are summarized.

✓ Strengths and weaknesses are discussed.

✓ Recommendations for the next experiment or submission are provided.

✓ The report is consistent with the implementation and evaluation results.

Only then is the experiment considered fully documented.