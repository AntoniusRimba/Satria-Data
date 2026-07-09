# Transfer Learning EfficientNet-B0 — Experiment 01

# Tujuan

Membangun model klasifikasi citra berbasis **Transfer Learning EfficientNet-B0** untuk kasus klasifikasi material sampah pada Big Data Challenge SATRIA DATA 2026.

Eksperimen ini bertujuan mengevaluasi sejauh mana penggunaan **backbone EfficientNet-B0 pretrained** mampu meningkatkan performa dibandingkan **CNN Baseline** dan **ResNet50** dalam mengklasifikasikan citra sampah ke dalam tiga kategori:

- Recyclable
- Electronic
- Organic

Selain itu, eksperimen ini menjadi bagian dari proses evaluasi berbagai backbone **Transfer Learning** sebelum dilakukan Fine-Tuning, Hyperparameter Tuning, serta pemilihan model terbaik untuk submission akhir.

---

# Scope

## Model

- EfficientNet-B0 (Pretrained ImageNet)
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

Menghasilkan implementasi EfficientNet-B0 yang:

- Berjalan tanpa error
- Modular
- Clean Code
- Reproducible
- Mudah dianalisis
- Mengikuti best practice PyTorch
- Mematuhi aturan kompetisi SATRIA DATA 2026
- Dapat dibandingkan secara objektif dengan CNN Baseline dan ResNet50

Target utama eksperimen ini adalah memperoleh **peningkatan performa awal menggunakan EfficientNet-B0**, serta mengevaluasi apakah backbone ini mampu memberikan performa yang lebih baik dibandingkan kandidat sebelumnya.

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

Transfer Learning (EfficientNet-B0)

│

▼

Evaluation

│

▼

Experiment Report

---

# Expected Output

Eksperimen ini diharapkan menghasilkan:

- Model EfficientNet-B0 yang berhasil dilatih
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
- Ringkasan hasil eksperimen untuk dibandingkan dengan CNN Baseline dan ResNet50

---

# Catatan

Eksperimen ini merupakan implementasi pertama menggunakan backbone **EfficientNet-B0**.

Fokus utama adalah membangun implementasi yang benar, terdokumentasi, modular, dan dapat dijadikan dasar untuk eksperimen lanjutan seperti:

- EfficientNet-B0 Fine-Tuning
- Hyperparameter Tuning
- ConvNeXt-Tiny
- Perbandingan antar model
- Pemilihan model terbaik untuk submission akhir