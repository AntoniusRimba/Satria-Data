# Scientific Report: Transfer Learning EfficientNet-B0 (Experiment 01)

**SATRIA DATA 2026 — Big Data Challenge: Automatic Waste Sorting**

---

## 1. Informasi Eksperimen

| Field | Detail |
|---|---|
| **Nama Eksperimen** | EfficientNet-B0 Feature Extraction — Experiment 01 |
| **Tujuan** | Mengevaluasi Transfer Learning EfficientNet-B0 sebagai kandidat backbone untuk klasifikasi material sampah BDC 2026 |
| **Tanggal Eksperimen** | *[Diisi setelah eksekusi notebook]* |
| **Model Utama** | `torchvision.models.efficientnet_b0` (IMAGENET1K_V1) |
| **Versi Eksperimen** | Experiment 01 — Feature Extraction |
| **Konfigurasi** | `configs/efficientnet.yaml` |
| **Notebook** | `notebooks/08_efficientnet.ipynb` |
| **Experiment Dir** | `experiments/04_efficientnet/exp_01/` |

---

## 2. Dataset

- **Sumber Data**: Dataset resmi Big Data Challenge (BDC) SATRIA DATA 2026
- **Jumlah Kelas**: 3 kelas (Recyclable, Electronic, Organic)
- **Total Gambar**: 26.527 gambar (terdeteksi pada saat eksperimen)

### Distribusi Dataset

| Kelas | Jumlah | Proporsi |
|---|---|---|
| Organic | 12.567 | 47.4% |
| Recyclable | 9.999 | 37.7% |
| Electronic | 3.961 | 14.9% |
| **Total** | **26.527** | **100%** |

> **Catatan Class Imbalance**: Kelas `Electronic` hanya memiliki ~14.9% dari total dataset. Rasio Organic:Electronic ≈ 3.17:1. Hal ini ditangani menggunakan `WeightedRandomSampler` pada training DataLoader.

### Train–Validation Split

| Split | Jumlah | Proporsi |
|---|---|---|
| Training | 21.219 | 80% |
| Validation | 5.305 | 20% |

- **Strategi**: Stratified Split — distribusi kelas dipertahankan di kedua set
- **Random Seed**: 42 (reproducible)
- **Implementasi**: `src/preprocessing/splitter.py` → `stratified_split()`

### Relevansi EDA

EDA tidak diulang karena dataset identik dengan eksperimen sebelumnya (CNN Baseline & ResNet50). Hasil EDA sebelumnya tetap valid:
- Format gambar: JPG/PNG, RGB — kompatibel
- Resolusi: bervariasi (multi-skala) → di-resize sebelum masuk model
- Anomali: terdapat beberapa gambar corrupt, ditangani oleh `WasteDataset` dengan graceful fallback

---

## 3. Data Preprocessing

Pipeline preprocessing yang digunakan adalah **reuse penuh** dari eksperimen sebelumnya — tidak ada perubahan. Hal ini disengaja untuk memastikan perbandingan antar model yang adil dan reproducible.

### Image Transforms

**Training Pipeline** (`get_train_transforms()`):
1. `RandomResizedCrop(224, scale=(0.8, 1.0))` — variasi skala dan posisi objek
2. `RandomHorizontalFlip(p=0.5)` — sampah tidak memiliki orientasi kanan-kiri inherent
3. `RandomRotation(degrees=15)` — simulasi variasi sudut kamera
4. `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)` — variasi pencahayaan
5. `ToTensor()` → `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

**Validation Pipeline** (`get_val_transforms()`):
1. `Resize(256)` → `CenterCrop(224)` — deterministik, tanpa augmentasi
2. `ToTensor()` → `Normalize(ImageNet stats)` — identik dengan training

### Normalisasi

Menggunakan statistik **ImageNet** (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Ini **wajib** untuk backbone pretrained EfficientNet-B0 agar distribusi input sesuai dengan kondisi saat pre-training.

### DataLoader

| Parameter | Nilai |
|---|---|
| Batch Size | 32 |
| Num Workers | 4 |
| Pin Memory | True |
| Drop Last (Train) | True |
| Sampler | `WeightedRandomSampler` (training) |

---

## 4. Arsitektur Model

### Konsep EfficientNet-B0 & Compound Scaling

EfficientNet diperkenalkan oleh Tan & Le (2019) dengan pendekatan **Compound Scaling** — menyeimbangkan tiga dimensi jaringan secara proporsional menggunakan koefisien φ (phi):

- **Depth** (kedalaman): lebih banyak layer
- **Width** (lebar): lebih banyak channel
- **Resolution** (resolusi input): gambar lebih besar

EfficientNet-B0 adalah baseline dengan φ=1.0 (depth=1.0×, width=1.0×, resolution=224×224). Ini menghasilkan rasio akurasi/parameter yang sangat efisien dibandingkan ResNet50.

### Blok Bangunan: MBConv (Mobile Inverted Bottleneck)

Setiap blok MBConv terdiri dari:
1. **Expansion Conv** (1×1) — perluas channel
2. **Depthwise Conv** (3×3 atau 5×5) — konvolusi per-channel
3. **SE Block** (Squeeze-and-Excitation) — channel attention
4. **Projection Conv** (1×1) — kompres kembali
5. **Skip Connection** (jika stride=1 dan input/output shape sama)

EfficientNet-B0 memiliki **16 MBConv blocks** dalam 7 stage.

### Struktur Backbone (Terimbeku)

```
Input: (B, 3, 224, 224)
  ↓
features[0] : Conv-BN-SiLU  (3 → 32 ch, stride=2)
features[1] : MBConv1  (32 → 16 ch, k=3, 1 block)
features[2] : MBConv6  (16 → 24 ch, k=3, 2 blocks)
features[3] : MBConv6  (24 → 40 ch, k=5, 2 blocks)
features[4] : MBConv6  (40 → 80 ch, k=3, 3 blocks)
features[5] : MBConv6  (80 → 112 ch, k=5, 3 blocks)
features[6] : MBConv6  (112 → 192 ch, k=5, 4 blocks)
features[7] : MBConv6  (192 → 320 ch, k=3, 1 block)
features[8] : Conv-BN-SiLU  (320 → 1280 ch)
  ↓
avgpool     : AdaptiveAvgPool2d(1, 1)  → (B, 1280, 1, 1)
  ↓
Flatten     → (B, 1280)
```

### Classification Head (Dilatih)

```
(B, 1280)
  ↓
Dropout(p=0.4)
  ↓
Linear(1280 → 3)
  ↓
Output Logits: (B, 3)
```

> **Perbedaan kritis dengan ResNet50**: Output backbone EfficientNet-B0 adalah **1.280-dim** (bukan 2.048-dim seperti ResNet50). Head API menggunakan `model.classifier` (bukan `model.fc`).

### Strategi Transfer Learning: Feature Extraction

Pada eksperimen ini, seluruh backbone (16 MBConv blocks + Conv layers) di-**freeze** — `requires_grad = False`. Hanya classification head yang dilatih.

**Alasan pemilihan Feature Extraction (bukan Full Fine-Tuning)**:
1. Dataset BDC ~21K training samples — risiko overfitting dengan full fine-tuning
2. Fitur ImageNet sudah sangat kaya dan representatif untuk domain visual umum
3. Sebagai baseline yang setara dengan ResNet50 Exp01 (strategi identik → perbandingan adil)
4. Full Fine-Tuning akan dilakukan pada Experiment 02 jika hasil belum optimal

### Ringkasan Parameter

| Komponen | Parameter | Status |
|---|---|---|
| Backbone (features + avgpool) | ~5.288.548 | ❄️ Frozen |
| Classification Head | ~3.843 | ✅ Trainable |
| **Total** | **~5.292.391** | — |

---

## 5. Hyperparameter Pelatihan

Konfigurasi dari `configs/efficientnet.yaml`:

| Parameter | Nilai | Alasan |
|---|---|---|
| **Epochs** | 20 | Konsisten dengan ResNet50 Exp01 → perbandingan fair |
| **Optimizer** | AdamW | Standar modern; weight decay terpisah dari gradient update |
| **Learning Rate** | 0.001 | LR relatif besar karena hanya head yang dilatih |
| **Weight Decay** | 0.0001 | Regularisasi L2 |
| **Scheduler** | ReduceLROnPlateau | Adaptif — reaksi terhadap perilaku val_macro_f1 aktual |
| **Scheduler Mode** | max | Monitor val_macro_f1 (higher is better) |
| **Scheduler Factor** | 0.5 | Kurangi LR 50% saat plateau |
| **Scheduler Patience** | 3 | Tunggu 3 epoch sebelum reduce |
| **Early Stopping Patience** | 5 | Stop jika val_macro_f1 tidak meningkat 5 epoch |
| **Early Stopping Monitor** | val_macro_f1 | Metrik utama kompetisi BDC |
| **Loss Function** | CrossEntropyLoss | Standar multi-class classification |
| **Batch Size** | 32 | Kompromi antara kecepatan dan stabilitas gradient |
| **Random Seed** | 42 | Reproducibility |
| **Dropout Head** | 0.4 | Regularisasi; lebih kecil dari ResNet50 (0.5) |

---

## 6. Hasil Evaluasi

> **Catatan**: Bagian ini wajib diisi setelah menjalankan `notebooks/08_efficientnet.ipynb`.

### Metrik Utama (Validation Set — Best Model)

| Metrik | Training | Validation |
|---|---|---|
| **Accuracy** | *[isi]* | *[isi]* |
| **Macro Precision** | *[isi]* | *[isi]* |
| **Macro Recall** | *[isi]* | *[isi]* |
| **Macro F1 Score** | *[isi]* | **[isi] ← METRIK UTAMA BDC** |
| **Best Epoch** | — | *[isi]* |

### Kinerja Per-Kelas (Validation Set)

| Kelas | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| **Recyclable** | *[isi]* | *[isi]* | *[isi]* | 2.000 |
| **Electronic** | *[isi]* | *[isi]* | *[isi]* | 792 |
| **Organic** | *[isi]* | *[isi]* | *[isi]* | 2.513 |

### Checkpoint Output

Tersimpan di `outputs/checkpoints/`:
- `efficientnet_b0_exp01_best_model.pth` — weights epoch terbaik
- `efficientnet_b0_exp01_last_model.pth` — weights epoch terakhir

### Laporan & Visualisasi Output

Tersimpan di `outputs/reports/` dan `outputs/figures/`:
- `efficientnet_b0_exp01_evaluation_metrics.json`
- `efficientnet_b0_exp01_training_history.csv`
- `efficientnet_b0_exp01_experiment_summary.json`
- `efficientnet_b0_exp01_full_summary.json`
- `efficientnet_b0_exp01_model_comparison.csv`
- `efficientnet_b0_exp01_learning_curves.png`
- `efficientnet_b0_exp01_confusion_matrix.png`
- `efficientnet_b0_exp01_per_class_metrics.png`
- `efficientnet_b0_exp01_model_comparison.png`

---

## 7. Analisis Objektif

> *Isi berdasarkan hasil aktual setelah eksekusi.*

### Proses Pembelajaran
- **Convergence**: *[Apakah loss turun stabil? Berapa epoch hingga konvergen?]*
- **LR Reduction**: *[Di epoch berapa scheduler mengurangi LR? Berapa kali?]*
- **Early Stopping**: *[Apakah dipicu? Di epoch berapa?]*

### Overfitting / Underfitting
- *[Analisis gap antara train_macro_f1 dan val_macro_f1]*
- *[Jika gap > 0.10 → indikasi overfitting meskipun backbone di-freeze]*
- *[Jika keduanya rendah → underfitting → fitur ImageNet kurang relevan untuk domain ini]*

### Kemampuan Generalisasi
- *[Seberapa baik val_macro_f1 mendekati train_macro_f1?]*
- *[Apakah WeightedRandomSampler berhasil menyeimbangkan prediksi kelas?]*

### Kelas Tersulit
- Berdasarkan Confusion Matrix: kelas *[X]* paling sering salah diklasifikasikan sebagai *[Y]*
- Faktor penyebab potensial: *[kemiripan warna/tekstur, variasi pencahayaan, dll.]*

---

## 8. Perbandingan dengan Model Sebelumnya

### vs CNN Baseline

| Komponen | CNN Baseline | EfficientNet-B0 (Feature Ext) | Peningkatan |
|---|---|---|---|
| **Macro F1 Score** | 0.2222 | *[isi]* | *[+/- isi]* |
| **Accuracy** | *[isi]* | *[isi]* | *[isi]* |
| **Total Params** | ~2.1 Juta | ~5.3 Juta | 2.5× lebih besar |
| **Trainable Params** | ~2.1 Juta | ~3.843 | Jauh lebih sedikit |
| **Strategi** | From Scratch | Feature Extraction | — |
| **Kecepatan Konvergensi** | *[isi]* | *[isi — lebih cepat]* | — |

**Interpretasi**: Penggunaan EfficientNet-B0 *[Berhasil / Gagal]* mengalahkan CNN Baseline secara signifikan. Fitur yang diekstrak dari ImageNet *[terbukti relevan / kurang relevan]* untuk domain sampah BDC.

### vs ResNet50 (Experiment 01)

| Komponen | ResNet50 (Exp01) | EfficientNet-B0 (Exp01) | Perbedaan |
|---|---|---|---|
| **Macro F1 Score** | *[isi]* | *[isi]* | *[+/- isi]* |
| **Accuracy** | *[isi]* | *[isi]* | *[+/- isi]* |
| **Total Params** | ~23.5 Juta | ~5.3 Juta | **4.4× lebih efisien** |
| **Trainable Params** | ~6.147 | ~3.843 | EfficientNet-B0 lebih ringan |
| **Backbone Output** | 2.048-dim | 1.280-dim | — |
| **Stabilitas Training** | *[isi]* | *[isi]* | — |
| **Kecepatan Konvergensi** | *[isi epoch]* | *[isi epoch]* | — |
| **Waktu Training/Epoch** | *[isi s]* | *[isi s]* | *[lebih cepat/lambat]* |

**Interpretasi**: EfficientNet-B0 *[Lebih baik / Seimbang / Lebih buruk]* dibandingkan ResNet50 dengan parameter yang jauh lebih sedikit. Ini *[membuktikan / tidak membuktikan]* keunggulan Compound Scaling untuk kasus ini.

---

## 9. Kelebihan Implementasi EfficientNet-B0

1. **Efisiensi Parameter**: ~5.3 Juta parameter total — 4.4× lebih efisien dari ResNet50 (~23.5 Juta) dengan performa yang kompetitif.
2. **Compound Scaling**: Keseimbangan depth-width-resolution menghasilkan representasi fitur yang lebih kaya per parameter dibandingkan scaling tunggal.
3. **Kecepatan Inference**: Lebih sedikit parameter → inference lebih cepat → lebih efisien untuk deployment.
4. **Depthwise Separable Conv**: MBConv menggunakan depthwise separable convolution yang jauh lebih efisien dari standard convolution.
5. **SE Block (Squeeze-and-Excitation)**: Mekanisme channel attention bawaan meningkatkan kemampuan selektivitas fitur.
6. *[Tambahkan berdasarkan hasil aktual]*

---

## 10. Kekurangan & Keterbatasan

1. **Bias Domain ImageNet**: Objek sampah di Indonesia mungkin berbeda visual dari objek dalam ImageNet — fitur statis mungkin tidak optimal.
2. **Feature Extraction Bottleneck**: Dengan backbone di-freeze, model tidak bisa menyesuaikan filter konvolusi terhadap tekstur material sampah yang spesifik.
3. **Kelas Electronic yang Sedikit**: Meskipun WeightedRandomSampler digunakan, kelas minoritas masih berisiko memiliki F1 yang rendah.
4. *[Kelemahan spesifik berdasarkan Confusion Matrix aktual]*
5. *[Potensi masalah lain yang ditemukan saat training]*

---

## 11. Kesimpulan

- **Tujuan eksperimen**: *[tercapai / belum sepenuhnya tercapai]*
- **EfficientNet-B0 vs CNN Baseline**: *[Lebih baik secara signifikan / Marginal / Lebih buruk]*
- **EfficientNet-B0 vs ResNet50**: *[Lebih baik / Seimbang / Lebih buruk]* dengan parameter 4.4× lebih sedikit
- **Efisiensi arsitektur**: EfficientNet-B0 *[terbukti / belum terbukti]* memberikan keunggulan efisiensi yang signifikan
- **Insight utama**: Strategi Feature Extraction dengan backbone ImageNet *[...]* untuk domain klasifikasi sampah BDC

---

## 12. Rekomendasi Eksperimen Berikutnya

Berdasarkan hasil eksperimen ini:

1. **EfficientNet-B0 Fine-Tuning (Experiment 02)**:
   - Unfreeze top 3 MBConv blocks (`features[6], features[7], features[8]`)
   - Gunakan LR lebih kecil (≤1e-4) untuk backbone, LR normal untuk head
   - Memanfaatkan `model.unfreeze_top_layers(n_blocks=3)` yang sudah tersedia

2. **Peningkatan Augmentasi**:
   - Tambahkan `RandomErasing` jika ditemukan overfitting
   - Coba `MixUp` atau `CutMix` untuk meningkatkan robustness

3. **ConvNeXt-Tiny (Experiment Berikutnya)**:
   - Kandidat selanjutnya dalam benchmark backbone
   - Arsitektur modern (2022) yang menggabungkan prinsip Vision Transformer ke CNN

4. **Pemilihan Model Terbaik**:
   - Bandingkan: CNN Baseline vs ResNet50 vs EfficientNet-B0 (Feature Ext & Fine-Tuned) vs ConvNeXt-Tiny
   - Pilih kandidat terbaik berdasarkan Macro F1 + efisiensi parameter + waktu inference

5. **Submission Preparation**:
   - Setelah model terbaik dipilih, lakukan Hyperparameter Tuning terbatas
   - Generate prediksi pada test set untuk submission BDC 2026

---

*Laporan ini dibuat mengikuti standar penulisan ilmiah SATRIA DATA 2026 Big Data Challenge.*
*Reproducibility: semua kode, konfigurasi, dan hasil tersimpan dalam repositori proyek.*
