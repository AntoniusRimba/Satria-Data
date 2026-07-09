# Transfer Learning EfficientNet-B0 — Experiment 01

## Tahap

Data Preprocessing

---

## Status

EDA telah selesai dan telah disetujui.

Pipeline preprocessing dasar telah tersedia dari eksperimen sebelumnya (CNN Baseline dan ResNet50).

---

## Tujuan

Memastikan pipeline preprocessing yang telah dibangun sebelumnya tetap sesuai dan kompatibel dengan implementasi **Transfer Learning EfficientNet-B0**.

Tahap ini bertujuan melakukan **verifikasi dan penyesuaian seperlunya** terhadap pipeline preprocessing agar sesuai dengan kebutuhan backbone pretrained EfficientNet-B0 tanpa mengubah prinsip dasar preprocessing maupun menimbulkan data leakage.

---

## Instruksi

Ikuti seluruh SOP pada command **/preprocessing**.

Ikuti seluruh aturan proyek yang terdapat pada **CLAUDE.md**.

Gunakan seluruh best practice pada skill **image-preprocessing**.

Pastikan seluruh preprocessing mematuhi aturan kompetisi **SATRIA DATA 2026**.

---

## Ruang Lingkup

Lakukan verifikasi terhadap pipeline preprocessing yang telah tersedia, meliputi:

- Validasi struktur dataset
- Train–Validation Split
- Image Transformation
- Resize sesuai input EfficientNet-B0
- Normalisasi menggunakan statistik ImageNet
- Data Augmentation (khusus data training)
- Penyusunan DataLoader
- Konfigurasi pipeline preprocessing yang modular dan reproducible

Apabila seluruh pipeline telah sesuai, gunakan kembali implementasi preprocessing yang telah ada tanpa membangun ulang.

---

## Batasan

Pada tahap ini:

- Jangan membangun model EfficientNet-B0.
- Jangan melakukan training.
- Jangan melakukan evaluasi model.
- Jangan menggunakan data test untuk proses preprocessing.
- Jangan melakukan preprocessing yang menyebabkan data leakage.
- Jangan membuat pipeline preprocessing baru apabila pipeline sebelumnya masih dapat digunakan.

---

## Output yang Diharapkan

Menghasilkan pipeline preprocessing yang:

- Modular
- Reproducible
- Dapat digunakan kembali
- Kompatibel dengan EfficientNet-B0

Output meliputi:

- Konfirmasi pipeline preprocessing yang digunakan
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

1. Tinjau kembali hasil EDA.
2. Verifikasi pipeline preprocessing yang telah dibangun pada eksperimen sebelumnya.
3. Identifikasi apakah terdapat penyesuaian yang diperlukan untuk EfficientNet-B0.
4. Jelaskan alasan setiap preprocessing yang digunakan.
5. Pastikan seluruh preprocessing kompatibel dengan EfficientNet-B0 pretrained.
6. Jelaskan dampak preprocessing terhadap performa model.
7. Baru lakukan implementasi apabila memang diperlukan.

---

## Catatan Penting

Gunakan preprocessing yang mengikuti best practice **Transfer Learning**.

Normalisasi harus mengikuti statistik **ImageNet**.

Ukuran input dan transformasi harus sesuai dengan karakteristik **EfficientNet-B0 pretrained**.

Prioritaskan penggunaan kembali (**reusable pipeline**) dibanding membangun pipeline preprocessing baru.

Hindari preprocessing yang dapat menghilangkan informasi penting dari objek sampah.

---

## Setelah Selesai

Apabila pipeline preprocessing telah sesuai, lanjutkan ke tahap implementasi **Transfer Learning EfficientNet-B0**.

Jangan melakukan pembangunan model sebelum proses verifikasi preprocessing selesai.