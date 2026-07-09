# Scientific Report: Transfer Learning ResNet50 (Experiment 01)

**SATRIA DATA 2026 - Big Data Challenge: Automatic Waste Sorting**

---

## 1. Informasi Eksperimen
- **Nama Eksperimen**: ResNet50 Feature Extraction (Baseline Transfer Learning)
- **Tujuan**: Mengevaluasi performa arsitektur ResNet50 menggunakan bobot *pretrained* ImageNet sebagai ekstraktor fitur statis, dibandingkan dengan model CNN buatan awal.
- **Tanggal Eksperimen**: *[Diisi setelah eksekusi]*
- **Model Utama**: `torchvision.models.resnet50`
- **Versi**: Eksperimen 01

---

## 2. Dataset
- **Sumber Data**: Dataset Kompetisi BDC SATRIA DATA 2026
- **Jumlah Kelas**: 3 Kelas (Recyclable, Electronic, Organic)
- **Distribusi Data**: *[Tulis ringkasan distribusi dari EDA, misal: Kelas terbanyak adalah Organic (X%), kelas paling sedikit adalah Electronic (Y%)]*
- **Train-Validation Split**: Stratified Split 80% Training, 20% Validation.

---

## 3. Data Preprocessing
Pipeline *preprocessing* dan augmentasi disusun untuk menyesuaikan input standar ImageNet:
- **Resize**: Resolusi diubah ke `256x256`, kemudian di-crop / di-*center crop* menjadi `224x224`.
- **Normalization**: Menggunakan mean `[0.485, 0.456, 0.406]` dan standar deviasi `[0.229, 0.224, 0.225]` sesuai standar pretrained ImageNet.
- **Data Augmentation**: *[Sebutkan jika ada, misalnya: RandomHorizontalFlip, RandomRotation]*
- **Dataloader**: Batch size `32`, menggunakan `WeightedRandomSampler` untuk mengatasi *imbalanced data*.

---

## 4. Arsitektur Model
- **Backbone**: `ResNet50` (Residual Network dengan kedalaman 50 layer).
- **Pretrained Weights**: `IMAGENET1K_V2` (atau V1). Membawa representasi fitur visual yang kaya.
- **Strategi Transfer Learning**: **Feature Extraction**. Seluruh layer pada backbone (layer 1-4) di-*freeze* (`requires_grad = False`). Total ~23.5 Juta parameter statis.
- **Classification Head**: Mengganti layer `fc` bawaan dengan `Sequential(Dropout(0.5), Linear(2048, 3))`. Head ini diinisialisasi secara acak dan menjadi satu-satunya bagian yang dilatih (trainable, ~6K parameter).

---

## 5. Hyperparameter Pelatihan
Konfigurasi diambil dari `configs/resnet50.yaml`:
- **Epoch**: *[isi, misal 20]*
- **Batch Size**: 32
- **Optimizer**: AdamW
- **Learning Rate Awal**: 0.001
- **Loss Function**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau (monitor='val_macro_f1', mode='max', factor=0.5, patience=3)
- **Early Stopping**: Patience = 5
- **Random Seed**: 42

---

## 6. Hasil Evaluasi
*(Bagian ini wajib diisi berdasarkan output dari Stage 4 / File Metrik)*

| Metrik Evaluasi | Training | Validation |
|---|---|---|
| **Accuracy** | *[isi]* | *[isi]* |
| **Macro Precision** | *[isi]* | *[isi]* |
| **Macro Recall** | *[isi]* | *[isi]* |
| **Macro F1 Score** | *[isi]* | **[isi - METRIK UTAMA]** |

**Kinerja per Kelas (Validation):**
- **Recyclable**: F1 = *[isi]*
- **Electronic**: F1 = *[isi]*
- **Organic**: F1 = *[isi]*

---

## 7. Analisis Objektif
- **Proses Belajar**: *[Jelaskan apakah loss turun dengan stabil atau berfluktuasi]*
- **Overfitting/Underfitting**: *[Apakah selisih loss training dan validation terlampau jauh? Atau sudah konvergen dengan baik?]*
- **Kemampuan Generalisasi**: *[Seberapa baik model memprediksi data yang belum pernah dilihat]*
- **Kelas Tersulit**: Berdasarkan *Confusion Matrix*, kelas *[X]* paling sering disalahartikan sebagai kelas *[Y]*. Faktor penyebab potensial: *[Bentuk objek mirip, pencahayaan, dll]*

---

## 8. Perbandingan dengan CNN Baseline
| Komponen | CNN Baseline | ResNet50 (Feature Ext) |
|---|---|---|
| **Macro F1 Score** | *[isi]* | *[isi]* |
| **Stabilitas Training** | *[Lebih lambat/cepat]* | *[Cepat konvergen di head]* |
| **Kompleksitas** | Model kecil (~2-5M param) | Model besar (~23M param) |

**Interpretasi:** Penggunaan `ResNet50` *[Berhasil / Gagal]* mengalahkan CNN Baseline. Fitur yang diekstrak ImageNet *[terbukti relevan / kurang relevan]* untuk domain sampah BDC.

---

## 9. Kelebihan Implementasi Ini
1. **Kecepatan Training**: Karena backbone di-freeze, kalkulasi gradient hanya dilakukan di layer terakhir. Sangat cepat meskipun arsitektur besar.
2. **Kualitas Ekstraksi Fitur**: ResNet50 memiliki *Receptive Field* yang luas dan kemampuan mendeteksi *texture/edges* yang sangat kuat.
3. *[Tambahkan kelebihan berdasarkan grafik]*

---

## 10. Kekurangan & Keterbatasan
1. **Bias Domain ImageNet**: Objek sampah di Indonesia mungkin memiliki visual yang berbeda dengan objek di ImageNet, sehingga layer fitur statis mungkin kurang optimal.
2. *[Kelemahan berdasarkan metrik]*
3. *[Potensi bottleneck pada single Linear layer]*

---

## 11. Kesimpulan Akhir
- **Tujuan eksperimen** *[tercapai / belum sepenuhnya tercapai]*.
- **Secara komparatif**, ResNet50 *[mendominasi / seimbang dengan]* CNN Baseline.
- **Insight utama**: Strategi *Feature Extraction* sangat bergantung pada representasi data asli.

---

## 12. Rekomendasi Eksperimen Berikutnya
Berdasarkan keterbatasan di atas, rekomendasi eksperimen selanjutnya adalah:
1. **ResNet50 Fine-Tuning (Eksperimen 02)**: Membuka kunci (unfreeze) beberapa *layer* terakhir di backbone agar model bisa menyesuaikan filter konvolusinya secara spesifik terhadap tekstur material sampah.
2. **Augmentasi Tambahan**: Menambahkan *ColorJitter* atau *RandomErasing* jika ditemukan indikasi *overfitting* pada kelas tertentu.
3. Eksplorasi kandidat *Transfer Learning* lain (EfficientNet) jika Fine-Tuning masih *stuck*.
