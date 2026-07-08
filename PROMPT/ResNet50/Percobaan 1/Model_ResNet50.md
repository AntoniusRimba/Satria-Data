# Transfer Learning ResNet50 — Experiment 01

## Tahap

Modeling

---

## Status

EDA dan Data Preprocessing telah selesai serta disetujui.

---

## Tujuan

Membangun model klasifikasi citra berbasis **Transfer Learning ResNet50** sebagai kandidat utama setelah CNN Baseline pada kasus klasifikasi material sampah Big Data Challenge SATRIA DATA 2026.

Eksperimen ini bertujuan mengevaluasi peningkatan performa dengan memanfaatkan representasi visual yang telah dipelajari oleh ResNet50 pada dataset ImageNet, kemudian diadaptasikan untuk mengklasifikasikan tiga kategori sampah:

- Recyclable
- Electronic
- Organic

---

## Instruksi

Ikuti seluruh SOP pada command **/model**.

Ikuti seluruh aturan proyek yang terdapat pada **CLAUDE.md**.

Gunakan seluruh best practice pada skill:

- transfer-learning
- deep-learning-training

Pastikan implementasi mengikuti seluruh aturan kompetisi **SATRIA DATA 2026**.

---

## Ruang Lingkup

Fokus implementasi meliputi:

- Memuat backbone pretrained ResNet50
- Mengganti classification head sesuai jumlah kelas
- Menyusun arsitektur model secara modular
- Menyusun pipeline training
- Menentukan loss function
- Menentukan optimizer
- Menentukan learning rate
- Menentukan scheduler (jika diperlukan)
- Menentukan checkpoint terbaik
- Menyimpan model terbaik

---

## Strategi Training

Eksperimen ini menggunakan strategi:

**Transfer Learning (Feature Extraction)**

Dengan ketentuan:

- Menggunakan pretrained ResNet50
- Backbone dibekukan (freeze)
- Hanya classification head yang dilatih
- Belum melakukan Fine-Tuning seluruh layer

Strategi Fine-Tuning akan dilakukan pada eksperimen berikutnya.

---

## Target

Menghasilkan implementasi ResNet50 yang:

- Berjalan tanpa error
- Modular
- Clean Code
- Reproducible
- Mudah dikembangkan
- Mengikuti best practice PyTorch
- Dapat dibandingkan secara objektif dengan CNN Baseline

Target utama bukan memperoleh skor leaderboard tertinggi, melainkan membangun implementasi Transfer Learning yang benar sebagai dasar eksperimen lanjutan.

---

## Sebelum Implementasi

Sebelum menulis kode:

1. Jelaskan kembali konsep Transfer Learning.
2. Jelaskan alasan pemilihan ResNet50 sebagai backbone.
3. Jelaskan struktur arsitektur ResNet50 secara konseptual.
4. Jelaskan bagian mana yang akan dibekukan (freeze) dan bagian mana yang akan dilatih.
5. Jelaskan alasan pemilihan hyperparameter awal.
6. Jelaskan output yang diharapkan.
7. Buat rencana implementasi secara bertahap.

Baru setelah itu lakukan implementasi.

---

## Output yang Diharapkan

Menghasilkan:

- Arsitektur ResNet50 yang modular
- Pipeline training
- Model checkpoint terbaik
- History training
- Training log
- Konfigurasi hyperparameter
- Dokumentasi implementasi

Seluruh output mengikuti struktur proyek yang telah ditetapkan.

---

## Catatan Penting

Eksperimen ini merupakan implementasi pertama menggunakan **Transfer Learning ResNet50**.

Gunakan konfigurasi yang stabil dan mudah dianalisis.

Belum melakukan:

- Fine-Tuning
- Hyperparameter Tuning
- Ensemble Learning
- Attention Module

Fokus utama adalah membangun baseline Transfer Learning yang kuat sebelum masuk ke tahap optimasi.

---

## Setelah Selesai

Setelah model berhasil dilatih:

- Jangan melakukan evaluasi lanjutan.
- Jangan melakukan Fine-Tuning.
- Jangan mengubah hyperparameter.
- Tunggu evaluasi dan persetujuan sebelum melanjutkan ke tahap **Evaluation**.