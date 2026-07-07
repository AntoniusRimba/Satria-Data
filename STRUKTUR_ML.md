Smart Waste Classification/
├── .claude/
│   ├── commands/
│   │   ├── eda.md
│   │   ├── evaluate.md
│   │   ├── model.md
│   │   ├── preprocessing.md
│   │   └── report.md
│   └── skills/
│       ├── deep-learning-training/
│       │   └── SKILL.md
│       ├── experiment-management/
│       │   └── SKILL.md
│       ├── image-eda/
│       │   └── SKILL.md
│       ├── image-preprocessing/
│       │   └── SKILL.md
│       ├── model-evaluation/
│       │   └── SKILL.md
│       ├── scientific-reporting/
│       │   └── SKILL.md
│       ├── transfer-learning/
│       │   └── SKILL.md
│       └── visualization/
│           └── SKILL.md
├── .gitignore
├── CLAUDE.md
├── README.md
├── STRUKTUR_ML.md
├── configs/
│   ├── .gitkeep
│   ├── baseline.yaml
│   ├── default.yaml
│   ├── efficientnet.yaml
│   └── resnet50.yaml
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── BDC2026/
│   │       ├── submission.csv
│   │       ├── test/
│   │       └── train/
│   │           ├── 0_Recyclable/
│   │           ├── 1_Electronic/
│   │           └── 2_Organic/
│   └── splits/
│       ├── train/
│       │   └── .gitkeep
│       └── validation/
│           └── .gitkeep
├── docs/
│   ├── competition_rules.md
│   ├── experiment_log.md
│   ├── meeting_notes.md
│   ├── methodology.md
│   └── references.md
├── environment.yml
├── evaluate.py
├── experiments/
│   ├── .gitkeep
│   ├── 01_baseline/
│   │   └── .gitkeep
│   ├── 02_resnet50/
│   │   └── .gitkeep
│   ├── 03_resnet50_finetune/
│   │   └── .gitkeep
│   ├── 04_efficientnet/
│   │   └── .gitkeep
│   └── comparison/
│       └── .gitkeep
├── main.py
├── models/
│   ├── cnn/
│   │   └── .gitkeep
│   ├── hybrid/
│   │   └── .gitkeep
│   └── transfer_learning/
│       └── .gitkeep
├── notebooks/
│   ├── .gitkeep
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_collection.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── 05_cnn_baseline.ipynb
│   ├── 06_evaluation.ipynb
│   ├── 07_resnet50.ipynb
│   ├── 08_efficientnet.ipynb
│   ├── 09_comparison.ipynb
│   └── 10_final_submission.ipynb
├── outputs/
│   ├── artifacts/
│   │   ├── best_model.pth
│   │   ├── history.json
│   │   ├── metrics.json
│   │   └── submission.csv
│   ├── checkpoints/
│   │   └── .gitkeep
│   ├── figures/
│   │   └── .gitkeep
│   ├── logs/
│   │   └── .gitkeep
│   ├── reports/
│   │   └── .gitkeep
│   └── submission/
│       └── .gitkeep
├── predict.py
├── reports/
│   └── .gitkeep
├── requirements.txt
├── src/
│   ├── data_collection/
│   │   └── .gitkeep
│   ├── datasets/
│   │   └── .gitkeep
│   ├── eda/
│   │   └── .gitkeep
│   ├── evaluation/
│   │   └── .gitkeep
│   ├── inference/
│   │   └── .gitkeep
│   ├── models/
│   │   ├── .gitkeep
│   │   ├── __init__.py
│   │   ├── cnn/
│   │   │   ├── baseline.py
│   │   │   └── layers.py
│   │   └── transfer_learning/
│   │       ├── convnext_tiny.py
│   │       ├── efficientnet_b0.py
│   │       └── resnet50.py
│   ├── preprocessing/
│   │   └── .gitkeep
│   ├── training/
│   │   └── .gitkeep
│   └── utils/
│       ├── .gitkeep
│       └── seed.py
└── train.py