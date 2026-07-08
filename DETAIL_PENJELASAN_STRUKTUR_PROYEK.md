📁 Smart Waste Classification/

Folder utama proyek.

Berisi seluruh source code, notebook, konfigurasi, dataset, hasil eksperimen, dan dokumentasi.

🤖 .claude/

Berisi seluruh konfigurasi Claude Code sehingga AI bekerja mengikuti SOP proyek.

Bukan bagian dari kode Machine Learning, tetapi "otak AI assistant" proyek.

📁 commands/

Berisi SOP tiap pipeline implementasi.

Claude akan mengikuti isi file ini ketika kamu menjalankan command.

/eda
/preprocessing
/model
/evaluate
/report
📁 skills/

Berisi knowledge base.

Misalnya

image-eda/

berisi best practice EDA.

transfer-learning/

berisi best practice Transfer Learning.

Claude akan menggunakan skill tersebut ketika mengerjakan pipeline terkait.

📁 artifacts/

Berisi artefak utama hasil training.

Misalnya

best_model.pth

model terbaik.

metrics.json

hasil evaluasi.

history.json

history training.

Folder ini berisi hasil akhir yang penting.

📁 configs/

Berisi seluruh konfigurasi eksperimen.

Misalnya

baseline.yaml

Konfigurasi CNN.

resnet50.yaml

Konfigurasi ResNet.

Isinya misalnya

batch size
learning rate
epoch
optimizer
scheduler

Keuntungannya

tidak perlu mengubah kode ketika ingin mencoba konfigurasi lain.

📁 data/

Berisi seluruh dataset.

Tidak ada kode di sini.

📁 raw/

Dataset asli dari panitia.

Contoh

train/

test/

submission.csv

Tidak boleh diubah.

📁 processed/

Kalau suatu saat preprocessing menghasilkan dataset permanen.

Contoh

resize permanen
remove corrupt image
convert grayscale

Kalau preprocessing dilakukan on-the-fly menggunakan PyTorch, folder ini bisa tetap kosong.

📁 train/

Opsional.

Kalau nanti kalian memutuskan menyimpan hasil split training.

📁 validation/

Opsional.

Berisi validation hasil split.

📁 test/

Opsional.

Biasanya kosong karena test berasal dari panitia.

📁 experiments/

Folder dokumentasi eksperimen.

Misalnya

01_baseline/

berisi seluruh hasil CNN Baseline.

02_resnet50/

hasil ResNet.

Isinya bisa berupa

config.yaml

metrics.json

notes.md

figure.png

Jadi seluruh eksperimen terdokumentasi.

📁 notebooks/

Notebook penelitian.

Ini bukan source code utama.

Notebook digunakan untuk

eksplorasi
analisis
dokumentasi eksperimen

Urutannya mengikuti roadmap penelitian.

01_business_understanding

Memahami kasus.

02_data_collection

Memahami dataset.

03_eda

EDA.

04_preprocessing

Preprocessing.

05_cnn_baseline

CNN Baseline.

dan seterusnya.

📁 outputs/

Semua output otomatis.

checkpoints/

Model setiap epoch.

Misalnya

epoch10.pth

epoch20.pth
figures/

Semua gambar.

Misalnya

Confusion Matrix

Loss Curve

Accuracy Curve

ROC Curve
logs/

Log training.

Misalnya

training.log
reports/

Ringkasan otomatis.

Misalnya

metrics.json

summary.csv
submission/

Berisi

submission.csv

yang siap di-upload ke leaderboard.

📁 reports/

Berbeda dengan outputs.

Kalau

outputs/reports

adalah hasil otomatis.

Sedangkan

reports/

adalah laporan yang ditulis manusia.

Misalnya

Laporan Akhir.pdf

Metodologi.md
📁 src/

Ini adalah inti proyek.

Semua source code berada di sini.

📁 data_collection/

Kode untuk membaca dataset.

Misalnya

download.py

extract.py
📁 datasets/

Custom Dataset PyTorch.

Misalnya

WasteDataset
📁 eda/

Seluruh fungsi EDA.

Misalnya

plot_class_distribution()

show_samples()
📁 preprocessing/

Seluruh preprocessing.

Misalnya

transform.py

augmentation.py
📁 models/

Semua arsitektur model.

📁 cnn/

CNN Baseline.

Misalnya

baseline.py

layers.py
📁 transfer_learning/

Semua model pretrained.

Misalnya

resnet50.py

efficientnet_b0.py

convnext_tiny.py
📁 training/

Kode training.

Misalnya

trainer.py

early_stopping.py
📁 evaluation/

Kode evaluasi.

Misalnya

metrics.py

confusion_matrix.py
📁 inference/

Kode prediksi.

Misalnya

predict.py
📁 utils/

Fungsi umum.

Misalnya

seed.py

logger.py

device.py
📄 train.py

Entry point untuk training.

Misalnya

python train.py
📄 evaluate.py

Menjalankan evaluasi model.

📄 predict.py

Melakukan inferensi pada data test.

Menghasilkan

submission.csv
📄 main.py

Opsional.

Biasanya digunakan untuk menjalankan pipeline lengkap.

📄 README.md

Penjelasan proyek.

Dibaca manusia.

📄 CLAUDE.md

Instruksi proyek untuk Claude Code.

Dibaca AI.

📄 STRUKTUR_ML.md

Dokumentasi internal mengenai arsitektur proyek, pipeline, dan konvensi pengembangan.

📄 requirements.txt

Daftar package Python yang digunakan.

📄 environment.yml

Konfigurasi environment Conda agar proyek dapat direproduksi.

📄 .gitignore

Daftar file yang tidak akan diunggah ke Git.

Contohnya

__pycache__/

*.pth

.ipynb_checkpoints/