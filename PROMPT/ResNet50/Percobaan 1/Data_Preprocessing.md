# Transfer Learning ResNet50 — Experiment 01

## Tahap

Data Preprocessing

---

## Status

EDA telah selesai dan telah disetujui.

---

## Tujuan

Mempersiapkan dataset agar sesuai dengan kebutuhan implementasi **Transfer Learning ResNet50** tanpa mengubah informasi penting pada citra.

Tahap preprocessing bertujuan menghasilkan pipeline preprocessing yang bersih, konsisten, reproducible, serta kompatibel dengan backbone pretrained ResNet50 sehingga dapat digunakan secara optimal pada proses pelatihan model.

---

## Instruksi

Ikuti seluruh SOP pada command **/preprocessing**.

Ikuti seluruh aturan proyek yang terdapat pada **CLAUDE.md**.

Gunakan seluruh best practice pada skill **image-preprocessing**.

Pastikan seluruh preprocessing mematuhi aturan kompetisi **SATRIA DATA 2026**.

---

## Ruang Lingkup

Fokus preprocessing meliputi:

- Validasi struktur dataset
- Train–Validation Split
- Image Transformation
- Resize sesuai input ResNet50
- Normalisasi menggunakan statistik ImageNet
- Data Augmentation (khusus data training)
- Penyusunan DataLoader
- Konfigurasi pipeline preprocessing yang modular dan reproducible

---

## Batasan

Pada tahap ini:

- Jangan membangun model ResNet50.
- Jangan melakukan training.
- Jangan melakukan evaluasi model.
- Jangan menggunakan data test untuk proses preprocessing.
- Jangan melakukan preprocessing yang menyebabkan data leakage.

---

## Output yang Diharapkan

Menghasilkan pipeline preprocessing yang:

- Modular
- Reproducible
- Mudah digunakan kembali
- Siap digunakan oleh ResNet50

Output meliputi:

- Pipeline Transform Training
- Pipeline Transform Validation
- DataLoader Training
- DataLoader Validation
- Dokumentasi preprocessing
- Ringkasan konfigurasi preprocessing

Seluruh output mengikuti struktur proyek yang telah ditetapkan.

---

## Cara Berpikir

Sebelum melakukan implementasi:

1. Analisis kembali hasil EDA.
2. Tentukan preprocessing yang benar-benar diperlukan.
3. Jelaskan alasan setiap preprocessing.
4. Pastikan preprocessing kompatibel dengan ResNet50 pretrained.
5. Jelaskan dampak preprocessing terhadap performa model.
6. Baru lakukan implementasi.

---

## Catatan Penting

Gunakan preprocessing yang mengikuti best practice Transfer Learning.

Normalisasi, ukuran input, dan transformasi harus sesuai dengan karakteristik backbone **ResNet50 pretrained**.

Hindari preprocessing yang dapat menghilangkan informasi penting dari objek sampah.

---

## Setelah Selesai

Jangan melanjutkan ke tahap pembangunan model.

Tunggu evaluasi dan persetujuan sebelum melanjutkan ke implementasi **Transfer Learning ResNet50**.