# Transfer Learning ResNet50 — Experiment 01

# Tujuan

Membangun model klasifikasi citra berbasis **Transfer Learning ResNet50** untuk kasus klasifikasi material sampah pada Big Data Challenge SATRIA DATA 2026.

Eksperimen ini bertujuan mengevaluasi sejauh mana penggunaan **backbone ResNet50 pretrained** mampu meningkatkan performa dibandingkan **CNN Baseline** dalam mengklasifikasikan citra sampah ke dalam tiga kategori:

- Recyclable
- Electronic
- Organic

Selain itu, eksperimen ini menjadi langkah awal dalam mengevaluasi efektivitas pendekatan **Transfer Learning** sebelum dilakukan Fine-Tuning, Hyperparameter Tuning, dan perbandingan dengan backbone lain seperti EfficientNet-B0 dan ConvNeXt-Tiny.

---

# Scope

## Model

- ResNet50 (Pretrained ImageNet)
- Transfer Learning
- Mengganti Classification Head sesuai jumlah kelas BDC
- Belum menggunakan Fine-Tuning penuh (Full Fine-Tuning dilakukan pada eksperimen berikutnya)
- Tidak menggunakan Attention Module
- Tidak menggunakan Ensemble Learning

---

## Dataset

Dataset resmi Big Data Challenge (BDC) SATRIA DATA 2026

- Train digunakan untuk Training dan Validation
- Validation digunakan untuk evaluasi selama pelatihan
- Test hanya digunakan pada proses inferensi akhir (Submission)

---

## Paradigma

- Supervised Learning
- Image Classification
- Computer Vision
- Deep Learning
- Transfer Learning

---

# Target

Menghasilkan implementasi ResNet50 yang:

- Berjalan tanpa error
- Modular
- Clean Code
- Reproducible
- Mudah dianalisis
- Mengikuti best practice PyTorch
- Mematuhi aturan kompetisi SATRIA DATA 2026
- Dapat dibandingkan secara objektif dengan CNN Baseline

Target utama eksperimen ini adalah memperoleh **peningkatan performa awal menggunakan Transfer Learning**, bukan langsung memperoleh skor leaderboard tertinggi.

---

# Tahapan yang Akan Dilakukan

Business Understanding
        │
        ▼
Data Collection
        │
        ▼
EDA
        │
        ▼
Preprocessing
        │
        ▼
Transfer Learning (ResNet50)
        │
        ▼
Evaluation
        │
        ▼
Experiment Report

---

# Expected Output

Eksperimen ini diharapkan menghasilkan:

- Model ResNet50 yang berhasil dilatih
- History Training
- Model Checkpoint Terbaik
- Accuracy
- Precision
- Recall
- Macro F1 Score
- Classification Report
- Confusion Matrix
- Training Curve
- Validation Curve
- Ringkasan hasil eksperimen untuk dibandingkan dengan CNN Baseline

---

# Catatan

Eksperimen ini merupakan implementasi pertama menggunakan pendekatan **Transfer Learning**.

Fokus utama adalah membangun implementasi yang benar, terdokumentasi, dan dapat dijadikan dasar untuk eksperimen lanjutan seperti:

- ResNet50 Fine-Tuning
- Hyperparameter Tuning
- EfficientNet-B0
- ConvNeXt-Tiny
- Perbandingan antar model
- Pemilihan model terbaik untuk submission akhir.