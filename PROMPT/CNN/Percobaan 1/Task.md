Baseline CNN — Experiment 01
Tujuan

Membangun model CNN sederhana sebagai baseline pertama untuk kasus klasifikasi citra sampah pada Big Data Challenge SATRIA DATA 2026.

Eksperimen ini bertujuan memperoleh performa awal yang nantinya akan dijadikan acuan dalam mengevaluasi peningkatan performa menggunakan model Transfer Learning (ResNet50, EfficientNet-B0, dan ConvNeXt-Tiny).

Scope

Model:

Custom CNN
Tidak menggunakan pretrained
Tidak menggunakan transfer learning
Tidak menggunakan attention module
Tidak menggunakan ensemble

Dataset:

Dataset resmi BDC
Train digunakan untuk training dan validation
Test hanya digunakan saat inferensi akhir

Paradigma:

Supervised Learning
Image Classification
Computer Vision
Target

Menghasilkan baseline yang:

berjalan tanpa error
modular
reproducible
mudah dianalisis
memenuhi aturan SATRIA DATA

Bukan mengejar akurasi tertinggi.

Tahapan yang akan dilakukan
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
CNN Baseline
        │
        ▼
Evaluation
        │
        ▼
Experiment Report