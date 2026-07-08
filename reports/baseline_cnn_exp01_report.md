# Laporan Ilmiah — Eksperimen Baseline CNN
## SATRIA DATA 2026 — Big Data Challenge
### Klasifikasi Citra Sampah Otomatis

---

**Experiment ID :** `baseline_cnn_exp01`
**Tanggal :** 2026-07-08
**Tahap :** Experiment 01 — Baseline CNN
**Status :** ✅ Selesai diimplementasi

---

## Daftar Isi

1. [Latar Belakang & Tujuan](#1-latar-belakang--tujuan)
2. [Dataset](#2-dataset)
3. [Preprocessing](#3-preprocessing)
4. [Arsitektur CNN](#4-arsitektur-cnn)
5. [Konfigurasi Hyperparameter](#5-konfigurasi-hyperparameter)
6. [Hasil Training](#6-hasil-training)
7. [Hasil Evaluasi](#7-hasil-evaluasi)
8. [Analisis](#8-analisis)
9. [Kelebihan & Kekurangan](#9-kelebihan--kekurangan)
10. [Rekomendasi Eksperimen Berikutnya](#10-rekomendasi-eksperimen-berikutnya)
11. [Kesimpulan](#11-kesimpulan)

---

## 1. Latar Belakang & Tujuan

### 1.1 Konteks Kompetisi

Eksperimen ini merupakan bagian dari partisipasi dalam **SATRIA DATA 2026 — Big Data Challenge (BDC)**, kategori **Computer Vision**. Kompetisi ini meminta peserta untuk membangun sistem klasifikasi citra sampah yang mampu mengkategorikan gambar sampah secara otomatis ke dalam tiga kelas:

| Label | Kelas | Deskripsi |
|---|---|---|
| 0 | **Recyclable** | Sampah yang dapat didaur ulang (plastik, logam, kertas, kaca) |
| 1 | **Electronic** | Sampah elektronik (perangkat listrik, komponen elektronik) |
| 2 | **Organic** | Sampah organik (sisa makanan, dedaunan) |

### 1.2 Tujuan Eksperimen

Eksperimen ini bertujuan membangun **model CNN sederhana sebagai baseline** dengan fokus pada:

- Menetapkan **tolok ukur (baseline) terukur** sebelum eksperimen Transfer Learning
- Memastikan pipeline ML berjalan **end-to-end** dan **reproducible**
- Menghasilkan kode yang **modular dan dapat dikembangkan**
- Memberikan **referensi perbandingan** yang jelas untuk ResNet50 dan EfficientNet-B0

> **Catatan penting:** Target eksperimen ini **bukan** mencapai akurasi tertinggi, melainkan menghasilkan baseline yang bersih, terdokumentasi, dan dapat diinterpretasikan.

### 1.3 Metrik Evaluasi Resmi

Metrik utama BDC adalah **Macro-averaged F1 Score**:

$$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

Macro F1 dipilih karena:
- Memperlakukan setiap kelas **setara** terlepas dari frekuensinya
- Lebih representatif untuk dataset yang **tidak sempurna seimbang**
- Accuracy saja dapat menyesatkan jika satu kelas dominan

---

## 2. Dataset

### 2.1 Sumber & Struktur

Dataset disediakan oleh panitia BDC SATRIA DATA 2026. Dataset bersifat **tertutup** dan hanya dapat digunakan dalam konteks kompetisi ini.

```
data/raw/BDC2026/
├── train/
│   ├── 0_Recyclable/   ← Gambar training kelas Recyclable
│   ├── 1_Electronic/   ← Gambar training kelas Electronic
│   └── 2_Organic/      ← Gambar training kelas Organic
└── test/               ← Test set (tidak digunakan dalam preprocessing)
```

### 2.2 Statistik Dataset

| Aspek | Detail |
|---|---|
| Total gambar training | ±26.527 gambar |
| Jumlah kelas | 3 (Recyclable, Electronic, Organic) |
| Format gambar | JPEG, PNG, dan format lain |
| Variasi resolusi | Sangat beragam (temuan EDA) |

### 2.3 Temuan Kunci EDA

Berdasarkan analisis EDA (Notebook `03_eda.ipynb`), temuan utama yang relevan untuk eksperimen ini:

| Temuan | Implikasi |
|---|---|
| Resolusi gambar sangat bervariasi | Wajib resize ke dimensi tetap (224×224) |
| Aspect ratio beragam | Resize + CenterCrop (bukan stretch) untuk menghindari distorsi |
| Potensi gambar non-RGB | Konversi ke RGB saat loading |
| Variasi pencahayaan dan latar belakang | Augmentasi ColorJitter dan Rotation |
| Dataset memiliki potensi imbalance ringan | WeightedRandomSampler pada training |

---

## 3. Preprocessing

Seluruh pipeline preprocessing diimplementasikan di `src/preprocessing/` dan didokumentasikan di `notebooks/04_preprocessing.ipynb`.

### 3.1 Image Standardization

| Parameter | Nilai | Alasan |
|---|---|---|
| Target size | 224 × 224 px | Standar CNN; kompatibel dengan Transfer Learning |
| Resize intermediate | 256 px (sisi pendek) | Menghindari distorsi aspect ratio sebelum crop |
| Crop strategy | CenterCrop(224) | Area tengah mengandung objek utama |
| Color space | RGB (3 channel) | Konversi otomatis saat load via PIL |

### 3.2 Normalisasi

| Parameter | Nilai |
|---|---|
| Mean | [0.485, 0.456, 0.406] |
| Std | [0.229, 0.224, 0.225] |
| Alasan | Statistik ImageNet — menjaga konsistensi pipeline dengan Transfer Learning |

### 3.3 Augmentasi (Training Set Only)

| Teknik | Parameter | Alasan |
|---|---|---|
| RandomResizedCrop | scale=(0.8, 1.0), size=224 | Variasi skala objek |
| RandomHorizontalFlip | p=0.5 | Sampah tidak punya orientasi tetap |
| RandomRotation | ±15° | Variasi sudut kamera |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05 | Variasi kondisi cahaya |

> **Penting:** Augmentasi **hanya** diterapkan pada Training Set. Validation dan Test Set menggunakan pipeline deterministik (Resize → CenterCrop → Normalize).

### 3.4 Dataset Split

| Parameter | Nilai |
|---|---|
| Strategi | Stratified Split (sklearn) |
| Training Set | 80% |
| Validation Set | 20% |
| Metode | In-memory (tidak menyalin file) |
| Random Seed | 42 |

Stratified split dipilih untuk memastikan proporsi kelas yang representatif di training dan validation set, kritis untuk evaluasi Macro F1 yang akurat.

### 3.5 DataLoader Configuration

| Parameter | Training | Validation |
|---|---|---|
| Batch Size | 32 | 32 |
| Shuffle | WeightedRandomSampler | False (deterministik) |
| Num Workers | 4 | 4 |
| Pin Memory | True | True |
| Drop Last | True | False |

**WeightedRandomSampler** digunakan pada training loader untuk menangani class imbalance secara transparan — kelas yang lebih sedikit mendapat probabilitas sampling lebih tinggi.

---

## 4. Arsitektur CNN

### 4.1 Gambaran Umum

Model yang digunakan adalah **Custom CNN** yang dirancang dari nol (*from scratch*, tanpa pretrained weights). Arsitektur dirancang dengan prinsip:

1. **Sederhana** — mudah dipahami dan dijelaskan
2. **Modular** — menggunakan `ConvBlock` yang reusable
3. **Best practice CNN modern** — BN + Dropout + GlobalAvgPool
4. **Komparabel** — menjadi referensi yang jelas terhadap Transfer Learning

### 4.2 Diagram Arsitektur

```
─────────────────────────────────────────────────────────────
Input:  [B, 3, 224, 224]
─────────────────────────────────────────────────────────────

┌─ ConvBlock 1 ──────────────────────────────────────────────┐
│  Conv2d(3→32, kernel=3, padding=1, bias=False)             │
│  BatchNorm2d(32)                                           │
│  ReLU(inplace=True)                                        │
│  MaxPool2d(kernel=2, stride=2)                             │
│  Dropout2d(p=0.25)                                         │
│                                         Output: [B,32,112,112] │
└────────────────────────────────────────────────────────────┘

┌─ ConvBlock 2 ──────────────────────────────────────────────┐
│  Conv2d(32→64, kernel=3, padding=1, bias=False)            │
│  BatchNorm2d(64)                                           │
│  ReLU(inplace=True)                                        │
│  MaxPool2d(kernel=2, stride=2)                             │
│  Dropout2d(p=0.25)                                         │
│                                         Output: [B,64,56,56]   │
└────────────────────────────────────────────────────────────┘

┌─ ConvBlock 3 ──────────────────────────────────────────────┐
│  Conv2d(64→128, kernel=3, padding=1, bias=False)           │
│  BatchNorm2d(128)                                          │
│  ReLU(inplace=True)                                        │
│  MaxPool2d(kernel=2, stride=2)                             │
│  Dropout2d(p=0.25)                                         │
│                                         Output: [B,128,28,28]  │
└────────────────────────────────────────────────────────────┘

┌─ ConvBlock 4 ──────────────────────────────────────────────┐
│  Conv2d(128→256, kernel=3, padding=1, bias=False)          │
│  BatchNorm2d(256)                                          │
│  ReLU(inplace=True)                                        │
│  MaxPool2d(kernel=2, stride=2)                             │
│  Dropout2d(p=0.25)                                         │
│                                         Output: [B,256,14,14]  │
└────────────────────────────────────────────────────────────┘

  AdaptiveAvgPool2d(1, 1)              Output: [B, 256, 1, 1]
  Flatten                              Output: [B, 256]

┌─ Classifier Head ──────────────────────────────────────────┐
│  Linear(256 → 128)                                         │
│  ReLU(inplace=True)                                        │
│  Dropout(p=0.5)                                            │
│  Linear(128 → 3)                    Output: [B, 3] (logits)│
└────────────────────────────────────────────────────────────┘
─────────────────────────────────────────────────────────────
```

### 4.3 Alasan Pemilihan Setiap Komponen

| Komponen | Alasan Teknis |
|---|---|
| **4 ConvBlocks** | Cukup untuk menangkap fitur tekstur dan bentuk sampah. Lebih dalam = lebih kompleks dan lebih sulit dianalisis sebagai baseline. |
| **BatchNorm2d** | Mengurangi *Internal Covariate Shift* → training lebih stabil dan konvergensi lebih cepat. Bertindak sebagai regularisasi tambahan. |
| **MaxPool2d** | Reduksi dimensi spasial secara bertahap (224→112→56→28→14 px). Mempertahankan fitur paling dominan. |
| **Dropout2d(0.25)** | Regularisasi pada level channel (lebih efektif dari pixel-level dropout untuk data spasial). Mencegah overfitting di feature extractor. |
| **Channel scaling 32→64→128→256** | Pola standar CNN modern — layer awal mendeteksi fitur sederhana (tepi, warna), layer dalam mendeteksi pola kompleks (bentuk, objek). |
| **AdaptiveAvgPool2d** | Menggantikan Flatten besar — menghasilkan representasi fixed 256-dim. Parameter jauh lebih sedikit, lebih modern, lebih dekat ResNet/EfficientNet. |
| **Dropout(0.5) pada FC** | Regularisasi kuat pada feature vector 256-dim sebelum klasifikasi akhir. |
| **He Initialization** | Optimal untuk aktivasi ReLU — mempertahankan variance signal di setiap layer saat inisialisasi. |

### 4.4 Jumlah Parameter

| Komponen | Parameter (perkiraan) |
|---|---|
| Feature Extractor (4 ConvBlocks) | ~560K |
| Classifier Head | ~33K |
| **Total** | **~593K** |

> *Jumlah aktual tergantung output `model.get_model_info()` setelah dijalankan.*

---

## 5. Konfigurasi Hyperparameter

Semua hyperparameter tersimpan di `configs/baseline.yaml`.

### 5.1 Training Configuration

| Parameter | Nilai | Alasan |
|---|---|---|
| **Epochs** | 30 | Cukup untuk baseline; early stopping akan menghentikan lebih awal jika perlu |
| **Batch Size** | 32 | Keseimbangan memori GPU dan stabilitas gradient |
| **Optimizer** | **AdamW** | Default modern; weight decay terpisah dari update (lebih baik dari Adam biasa) |
| **Learning Rate** | 0.001 | Default stabil untuk AdamW |
| **Weight Decay** | 0.0001 | Regularisasi L2 ringan |
| **Loss Function** | **CrossEntropyLoss** | Standar untuk multi-class classification dengan kelas mutually exclusive |
| **Scheduler** | **ReduceLROnPlateau** | Adaptif: LR dikurangi 50% saat val_macro_f1 tidak meningkat selama 3 epoch |
| **Scheduler Factor** | 0.5 | LR baru = LR × 0.5 |
| **Scheduler Patience** | 3 epoch | |
| **Min LR** | 1e-5 | Batas bawah LR |
| **Early Stopping** | patience=5 | Monitor: val_macro_f1 (maximize) |
| **Random Seed** | 42 | Reproducibility |

### 5.2 Keputusan AdamW vs Adam

AdamW dipilih karena:
- Adam menggabungkan weight decay ke dalam gradient update (secara matematis tidak tepat)
- AdamW memisahkan weight decay → regularisasi lebih efektif
- AdamW menjadi default di banyak implementasi modern (HuggingFace, PyTorch Lightning)

### 5.3 Keputusan ReduceLROnPlateau vs StepLR

ReduceLROnPlateau dipilih karena:
- Bereaksi terhadap **perilaku aktual** training (bukan jadwal tetap)
- Lebih mudah dijelaskan: "LR turun saat model tidak berkembang"
- Tidak perlu menentukan `step_size` yang optimal di awal

---

## 6. Hasil Training

> ⚠️ *Section ini akan diisi setelah notebook `05_cnn_baseline.ipynb` dijalankan.*

### 6.1 Training Summary

| Metrik | Nilai |
|---|---|
| Total Epochs Dijalankan | _(isi setelah training)_ |
| Best Epoch | _(isi setelah training)_ |
| Early Stopped | _(Ya/Tidak)_ |
| Total Training Time | _(menit)_ |

### 6.2 Training Metrics (Best Epoch)

| Split | Loss | Accuracy | Macro F1 |
|---|---|---|---|
| Training | _(isi)_ | _(isi)_ | _(isi)_ |
| Validation | _(isi)_ | _(isi)_ | _(isi)_ |

### 6.3 Learning Curve

> *Gambar: `outputs/figures/baseline_cnn_exp01_training_curve.png`*

---

## 7. Hasil Evaluasi

> ⚠️ *Section ini akan diisi setelah notebook `06_evaluation.ipynb` dijalankan.*

### 7.1 Metrik Utama (Validation Set)

| Metrik | Nilai |
|---|---|
| ⭐ **Macro F1 Score** (BDC Primary) | _(isi setelah evaluasi)_ |
| Accuracy | _(isi)_ |
| Macro Precision | _(isi)_ |
| Macro Recall | _(isi)_ |

### 7.2 Classification Report

| Kelas | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| Recyclable | _(isi)_ | _(isi)_ | _(isi)_ | _(isi)_ |
| Electronic | _(isi)_ | _(isi)_ | _(isi)_ | _(isi)_ |
| Organic | _(isi)_ | _(isi)_ | _(isi)_ | _(isi)_ |
| **Macro Avg** | _(isi)_ | _(isi)_ | _(isi)_ | _(isi)_ |

### 7.3 Confusion Matrix

> *Gambar: `outputs/figures/baseline_cnn_exp01_confusion_matrix.png`*

**Interpretasi:**
> _(Isi setelah evaluasi: kelas mana yang paling sering tertukar, pola error yang ditemukan)_

### 7.4 Per-Class Metrics

> *Gambar: `outputs/figures/baseline_cnn_exp01_perclass_metrics.png`*

**Kelas terkuat:** _(isi)_ — F1 = _(isi)_
**Kelas terlemah:** _(isi)_ — F1 = _(isi)_

---

## 8. Analisis

### 8.1 Analisis Learning Curve

> *(Isi setelah training — berdasarkan gambar `baseline_cnn_exp01_learning_curve_eval.png`)*

Panduan interpretasi (template):

| Kondisi | Indikator |
|---|---|
| **Healthy convergence** | Val loss turun bersama train loss; gap kecil |
| **Overfitting** | Train loss jauh lebih rendah dari val loss; gap membesar di epoch akhir |
| **Underfitting** | Keduanya masih tinggi; model belum cukup belajar |
| **Unstable learning** | Loss naik-turun tidak konsisten |

**Observasi:** _(Isi setelah training)_
**Diagnosa:** _(Healthy / Overfitting / Underfitting)_
**Bukti:** _(Generalization gap = train_f1 - val_f1 = ?)_

### 8.2 Analisis Error

> *(Isi setelah evaluasi — berdasarkan gambar `baseline_cnn_exp01_error_analysis.png`)*

**Pola kesalahan yang diamati:** _(isi)_
**Kemungkinan penyebab:**
- _(contoh: kelas Electronic dan Recyclable memiliki tampilan visual yang tumpang tindih)_
- _(contoh: variasi pencahayaan pada kelas Organic)_

### 8.3 Analisis Imbalance

WeightedRandomSampler diterapkan untuk menangani potensi class imbalance. Efektivitasnya dapat dilihat dari:
- Perbandingan F1 per kelas: apakah kelas minoritas memiliki F1 yang wajar?
- Diagonal confusion matrix: apakah recall semua kelas cukup tinggi?

---

## 9. Kelebihan & Kekurangan

### 9.1 Kelebihan

| # | Kelebihan | Penjelasan |
|---|---|---|
| 1 | **Sederhana & Interpretable** | Arsitektur 4-block mudah dipahami dan dijelaskan di laporan kompetisi |
| 2 | **Modular** | `ConvBlock` reusable; mudah dimodifikasi untuk eksperimen berikutnya |
| 3 | **Reproducible** | Seed fixed, konfigurasi tersimpan di YAML, split tersimpan di CSV |
| 4 | **Pipeline lengkap** | Dari raw data → preprocessing → training → evaluasi → laporan |
| 5 | **Best practice** | BN, Dropout, GlobalAvgPool, WeightedSampler, AdamW, ReduceLROnPlateau |
| 6 | **Training dari scratch** | Tidak bergantung pada pretrained weights → valid untuk analisis kemampuan arsitektur |

### 9.2 Kekurangan

| # | Kekurangan | Penjelasan |
|---|---|---|
| 1 | **Kapasitas terbatas** | ~593K parameter jauh lebih kecil dari ResNet50 (~25M) → ekspresivitas terbatas |
| 2 | **Fitur visual sederhana** | Tanpa pretrained knowledge, model harus belajar dari nol → butuh lebih banyak data dan epoch |
| 3 | **Tidak ada attention mechanism** | Tidak mampu fokus pada region penting seperti arsitektur modern |
| 4 | **Rentan terhadap class confusion** | Kelas dengan tampilan visual mirip (e.g., Electronic vs Recyclable) lebih sulit dibedakan |
| 5 | **Bukan state-of-the-art** | Desain by design — ini adalah baseline, bukan model terbaik |

---

## 10. Rekomendasi Eksperimen Berikutnya

Berdasarkan analisis eksperimen baseline ini, rekomendasi eksperimen selanjutnya adalah:

### 10.1 Prioritas Utama: Transfer Learning

| Eksperimen | Model | Alasan |
|---|---|---|
| **Experiment 02** | ResNet50 | Backbone terbukti; 25M parameter; skip connections mencegah vanishing gradient |
| **Experiment 03** | EfficientNet-B0 | Efisien; rasio parameter/akurasi terbaik; cocok untuk dataset ukuran sedang |

**Justifikasi:** Transfer Learning memanfaatkan fitur visual yang telah dipelajari dari ImageNet (jutaan gambar). Untuk dataset sampah, fitur seperti tekstur plastik, logam, dan organik kemungkinan sudah direpresentasikan dengan baik oleh backbone pretrained.

### 10.2 Perbaikan Potensial (jika F1 Baseline rendah)

| Area | Saran |
|---|---|
| Augmentasi | Tambahkan `RandomErasing` atau `MixUp` untuk kelas terlemah |
| Arsitektur | Tambahkan `ConvBlock 5` dengan 512 filter (test lebih dalam) |
| Loss | Ganti `CrossEntropyLoss` dengan `FocalLoss` jika imbalance signifikan |
| Regularisasi | Tambahkan `Label Smoothing` (ε=0.1) |

### 10.3 Target Metrik Eksperimen Berikutnya

| Eksperimen | Target Macro F1 |
|---|---|
| Baseline CNN (ini) | Referensi |
| ResNet50 | > Baseline + margin signifikan |
| EfficientNet-B0 | > ResNet50 atau sebanding dengan efisiensi lebih baik |

---

## 11. Kesimpulan

Eksperimen ini berhasil membangun **Custom CNN Baseline** untuk klasifikasi citra sampah dalam kompetisi SATRIA DATA 2026 BDC. Model dengan arsitektur 4 ConvBlock (32→64→128→256 channels) + GlobalAvgPool + FC head dilatih dari nol menggunakan AdamW, CrossEntropyLoss, dan ReduceLROnPlateau.

Pipeline implementasi bersifat **modular, reproducible, dan terdokumentasi** — mencakup preprocessing, augmentasi, training, evaluasi, dan laporan. Setiap keputusan teknis didukung oleh temuan EDA dan best practice deep learning.

Model baseline ini memberikan **referensi terukur** yang akan digunakan untuk membandingkan performa Transfer Learning (ResNet50 dan EfficientNet-B0) pada eksperimen berikutnya.

> *Nilai metrik aktual (Macro F1, Accuracy, dll.) akan diisi setelah notebook 05 dan 06 dijalankan.*

---

## Referensi Artifact

| Artifact | Path |
|---|---|
| Konfigurasi | `configs/baseline.yaml` |
| EDA Notebook | `notebooks/03_eda.ipynb` |
| Preprocessing Notebook | `notebooks/04_preprocessing.ipynb` |
| Training Notebook | `notebooks/05_cnn_baseline.ipynb` |
| Evaluation Notebook | `notebooks/06_evaluation.ipynb` |
| Model Source | `src/models/cnn/baseline.py` |
| Training Source | `src/training/trainer.py` |
| Best Checkpoint | `outputs/checkpoints/baseline_cnn_exp01_best_model.pth` |
| Training History | `outputs/reports/baseline_cnn_exp01_training_history.csv` |
| Evaluation Metrics | `outputs/reports/baseline_cnn_exp01_evaluation_metrics.json` |
| Confusion Matrix | `outputs/figures/baseline_cnn_exp01_confusion_matrix.png` |
| Learning Curve | `outputs/figures/baseline_cnn_exp01_training_curve.png` |
| Error Analysis | `outputs/figures/baseline_cnn_exp01_error_analysis.png` |

---

*Laporan ini dibuat secara otomatis menggunakan pipeline scientific reporting proyek Smart Waste Classification.*
*Tanggal generate: 2026-07-08 | Experiment ID: baseline_cnn_exp01 | Seed: 42*
