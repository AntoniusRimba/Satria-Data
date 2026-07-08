# RESNET50 — EXPERIMENT 01

Saat ini kita akan memulai implementasi eksperimen pertama menggunakan **Transfer Learning ResNet50** pada proyek Big Data Challenge SATRIA DATA 2026.

Tahap eksperimen yang akan dilakukan adalah:

**ResNet50 (Experiment 01)**

## Tujuan Eksperimen

Membangun model klasifikasi citra berbasis **Transfer Learning ResNet50** untuk mengidentifikasi jenis material sampah ke dalam tiga kategori:

- Recyclable
- Electronic
- Organic

Eksperimen ini bertujuan untuk mengevaluasi peningkatan performa dibandingkan **CNN Baseline** dengan memanfaatkan backbone ResNet50 yang telah dilatih sebelumnya pada dataset ImageNet.

Model yang dibangun akan menjadi kandidat utama dalam proses pemilihan model terbaik sebelum dilakukan tahap Fine-Tuning dan Hyperparameter Tuning.

---

## Aturan Implementasi

Ikuti seluruh aturan proyek yang terdapat pada **CLAUDE.md**.

Ikuti SOP pada command **/model**.

Gunakan seluruh best practice pada skill **transfer-learning** dan **deep-learning-training**.

Pastikan implementasi mengikuti seluruh aturan kompetisi **SATRIA DATA 2026**.

---

## Kriteria Implementasi

- Modular
- Clean Code
- Reproducible
- Mudah dikembangkan
- Menggunakan PyTorch
- Menggunakan backbone pretrained **ResNet50**
- Mematuhi aturan penggunaan pretrained model pada kompetisi

---

## Sebelum Menulis Kode

Lakukan tahapan berikut terlebih dahulu:

1. Analisis kembali tujuan eksperimen dan hubungannya dengan CNN Baseline.
2. Jelaskan konsep Transfer Learning serta alasan pemilihan ResNet50 sebagai backbone.
3. Jelaskan arsitektur ResNet50 yang akan digunakan beserta modifikasi pada classification head.
4. Jelaskan strategi transfer learning yang akan digunakan (Feature Extraction atau Fine-Tuning) beserta alasannya.
5. Jelaskan hyperparameter awal yang akan digunakan.
6. Jelaskan output yang diharapkan.
7. Buat rencana implementasi secara bertahap.

---

## Catatan Penting

Eksperimen ini **bukan bertujuan memperoleh leaderboard terbaik**, tetapi membangun implementasi **ResNet50 pertama yang benar, modular, terdokumentasi, dan mudah dianalisis**.

Model ini nantinya akan dibandingkan secara objektif dengan:

- CNN Baseline
- EfficientNet-B0
- ConvNeXt-Tiny

sebelum dipilih sebagai kandidat terbaik untuk tahap optimasi berikutnya.

---

Jangan langsung menghasilkan seluruh kode.

Mulailah dengan analisis, perencanaan implementasi, serta penjelasan teori yang relevan.

Tunggu persetujuan setelah rencana implementasi selesai.