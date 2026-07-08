# Experiment 01 — Baseline CNN

## Identitas Eksperimen

| Field | Detail |
|---|---|
| Experiment ID | `baseline_cnn_exp01` |
| Model | Custom CNN (4 ConvBlocks, dari scratch) |
| Tanggal | 2026-07-08 |
| Dataset | SATRIA DATA 2026 BDC — 26.527 gambar training |
| Metrik Utama | Macro-averaged F1 Score |
| Tujuan | Membangun baseline terukur |

---

## Arsitektur

```
Input [B, 3, 224, 224]
  → ConvBlock 1: 3  → 32  ch  (Conv→BN→ReLU→MaxPool→Dropout2d)
  → ConvBlock 2: 32 → 64  ch  (Conv→BN→ReLU→MaxPool→Dropout2d)
  → ConvBlock 3: 64 → 128 ch  (Conv→BN→ReLU→MaxPool→Dropout2d)
  → ConvBlock 4: 128→ 256 ch  (Conv→BN→ReLU→MaxPool→Dropout2d)
  → AdaptiveAvgPool2d(1,1)
  → Flatten
  → Linear(256→128) → ReLU → Dropout(0.5)
  → Linear(128→3)
Output [B, 3]
```

---

## Hyperparameter

| Parameter | Nilai |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Loss | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping | patience=5, monitor=val_macro_f1 |
| Epochs | 30 |
| Batch Size | 32 |
| Input Size | 224×224 |
| WeightedRandomSampler | True |

---

## Preprocessing

- Resize(256) → RandomResizedCrop(224) [train] / CenterCrop(224) [val]
- Normalisasi: ImageNet mean/std
- Augmentasi: HorizontalFlip, Rotation±15°, ColorJitter
- Split: 80% train / 20% val (stratified)

---

## Hasil

> *Akan diisi setelah training selesai*

| Metrik | Nilai |
|---|---|
| Best Epoch | - |
| Val Macro F1 | - |
| Val Accuracy | - |
| Val Precision | - |
| Val Recall | - |
| Early Stopped | - |
| Training Time | - |

---

## Catatan

- Eksperimen ini bertujuan membangun baseline, bukan mencapai akurasi tertinggi.
- Hasil eksperimen ini menjadi acuan untuk membandingkan ResNet50 dan EfficientNet-B0.

---

## Output Files

| File | Path |
|---|---|
| Best Model | `outputs/checkpoints/baseline_cnn_exp01_best_model.pth` |
| Last Model | `outputs/checkpoints/baseline_cnn_exp01_last_model.pth` |
| Training History | `outputs/reports/baseline_cnn_exp01_training_history.csv` |
| Summary JSON | `outputs/reports/baseline_cnn_exp01_experiment_summary.json` |
| Training Curve | `outputs/figures/baseline_cnn_exp01_training_curve.png` |
