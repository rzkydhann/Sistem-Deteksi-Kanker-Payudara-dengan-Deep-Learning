# Sistem Deteksi Kanker Payudara dengan Deep Learning

Sistem klasifikasi citra mamografi untuk mendeteksi kanker payudara (benign/malignant) menggunakan Convolutional Neural Network (CNN) dan Flask web framework.

## 🎯 Fitur Utama

- ✅ Klasifikasi citra mammografi (Benign vs Malignant)
- ✅ Transfer Learning dengan ResNet50
- ✅ Web interface yang user-friendly
- ✅ Real-time prediction dengan confidence score
- ✅ Data augmentation untuk meningkatkan akurasi
- ✅ Visualisasi hasil prediksi

## 📁 Struktur Project

```
breast-cancer-detection/
├── app.py                      # Flask application
├── train_model.py              # Script training model
├── requirements.txt            # Dependencies
├── templates/
│   └── index.html             # Web interface
├── static/
│   └── uploads/               # Folder untuk gambar yang diupload
├── models/
│   └── breast_cancer_model.h5 # Model yang sudah ditraining
└── dataset/                   # Dataset mammografi
    ├── benign/                # Gambar benign
    └── malignant/             # Gambar malignant
```

## 📋 Requirements

### Software Requirements:
- **Python**: 3.8 - 3.11 (Recommended: 3.10)
- **pip**: Package installer for Python
- **Virtual Environment**: venv atau conda (optional tapi recommended)

### Hardware Requirements:
- **RAM**: Minimum 8GB (Recommended: 16GB+)
- **Storage**: Minimum 5GB free space
- **GPU**: Optional (NVIDIA dengan CUDA support untuk training lebih cepat)
- **Processor**: Multi-core processor recommended

### Library Dependencies:
```
flask==3.0.0              # Web framework
tensorflow==2.15.0        # Deep learning framework
numpy==1.24.3            # Numerical computing
opencv-python==4.8.1.78  # Image processing
pillow==10.1.0           # Image handling
matplotlib==3.8.2        # Plotting & visualization
scikit-learn==1.3.2      # Machine learning utilities
pandas==2.1.4            # Data manipulation
werkzeug==3.0.1          # WSGI utilities
```

### Dataset Requirements:
- **Format**: PNG atau JPG
- **Minimum jumlah**: 300-500 gambar per kelas (benign/malignant)
- **Recommended**: 1000+ gambar per kelas untuk hasil optimal
- **Resolusi**: Bebas (akan di-resize ke 224x224)
- **Color**: Grayscale atau RGB (akan dikonversi otomatis)

## 🚀 Instalasi

### 1. Clone atau Download Project

```bash
git clone <repository-url>
cd breast-cancer-detection
```

### 2. Buat Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Jika menggunakan GPU (NVIDIA):**
```bash
pip install tensorflow-gpu==2.15.0
```

### 3. Siapkan Dataset

Buat struktur folder dataset seperti ini:

```
dataset/
├── benign/
│   ├── benign_image_1.png
│   ├── benign_image_2.png
│   └── ...
└── malignant/
    ├── malignant_image_1.png
    ├── malignant_image_2.png
    └── ...
```

**Sumber Dataset Mammografi yang Bisa Digunakan:**

- [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) - Curated Breast Imaging Subset of DDSM
- [INbreast](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset) - Full-Field Digital Mammography
- [Mini-MIAS](http://peipa.essex.ac.uk/info/mias.html) - Mammographic Image Analysis Society
- [Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) - Alternative dataset

### 4. Training Model

```bash
python train_model.py
```

Script ini akan:
- Load dan preprocess dataset
- Split data menjadi train/validation/test
- Training model dengan data augmentation
- Save model terbaik ke `models/breast_cancer_model.h5`
- Generate grafik training history

**Proses training akan memakan waktu tergantung:**
- Jumlah data (rekomendasi minimal 500 gambar per kelas)
- Spesifikasi komputer (GPU sangat disarankan)
- Estimasi: 30-60 menit untuk dataset sedang

### 5. Jalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di: `http://localhost:5000`

## 💻 Penggunaan

1. Buka browser dan akses `http://localhost:5000`
2. Klik area upload atau drag & drop gambar mammografi
3. Klik tombol "Analisis Gambar"
4. Sistem akan menampilkan:
   - Hasil prediksi (Benign/Malignant)
   - Confidence score (%)
   - Gambar yang dianalisis
   - Disclaimer medis

## 🔧 Konfigurasi

### Ubah Ukuran Gambar (di `train_model.py` dan `app.py`)

```python
IMG_SIZE = (224, 224)  # Default untuk ResNet50
```

### Ubah Parameter Training

```python
BATCH_SIZE = 32
EPOCHS = 50
```

### Pilih Model Architecture

Di `train_model.py`, ada 2 pilihan:

1. **ResNet50 Transfer Learning** (untuk dataset besar >500 gambar)
2. **Simple CNN** (untuk dataset kecil <500 gambar)

Model otomatis dipilih berdasarkan ukuran dataset.

## 📊 Evaluasi Model

Setelah training, cek file `models/training_history.png` untuk melihat:
- Accuracy vs Epoch
- Loss vs Epoch

Metrik evaluasi yang ditampilkan:
- **Accuracy**: Akurasi keseluruhan
- **Precision**: Seberapa tepat prediksi positif
- **Recall**: Seberapa lengkap mendeteksi positif
- **F1-Score**: Harmonic mean precision & recall

## ⚠️ Catatan Penting

1. **Disclaimer Medis**: Sistem ini BUKAN pengganti diagnosis medis profesional. Selalu konsultasikan dengan dokter spesialis.

2. **Kualitas Data**: Akurasi model sangat bergantung pada:
   - Kualitas gambar mammografi
   - Jumlah dan keseimbangan dataset
   - Preprocessing yang tepat

3. **GPU Recommended**: Training akan jauh lebih cepat dengan GPU. Install TensorFlow-GPU jika tersedia.

4. **Data Augmentation**: Sudah diterapkan untuk mengatasi overfitting:
   - Rotation
   - Width/height shift
   - Horizontal flip
   - Zoom

## 🔍 Troubleshooting

### Model tidak dimuat
```
Error: Model tidak ditemukan di models/breast_cancer_model.h5
```
**Solusi**: Jalankan `python train_model.py` terlebih dahulu

### Dataset kosong
```
ERROR: Dataset tidak ditemukan!
```
**Solusi**: Pastikan struktur folder dataset benar dan ada gambar di folder benign/malignant

### Error saat upload
```
Format file tidak didukung!
```
**Solusi**: Gunakan format JPG atau PNG, maksimal 10MB

### Memory Error saat training
**Solusi**: 
- Kurangi `BATCH_SIZE`
- Kurangi ukuran `IMG_SIZE`
- Gunakan model Simple CNN

## 📈 Meningkatkan Performa

1. **Tambah Data**: Semakin banyak data training, semakin baik
2. **Fine-tuning**: Unfreeze beberapa layer ResNet50
3. **Ensemble**: Gabungkan beberapa model
4. **Preprocessing**: Tambahkan normalisasi khusus mammografi
5. **Class Weight**: Jika data imbalanced, gunakan class weights

## 📝 Lisensi

Project ini untuk tujuan edukasi dan penelitian. Tidak untuk penggunaan medis komersial tanpa validasi klinis yang proper.

## 👨‍💻 Kontribusi

Silakan berkontribusi dengan:
- Menambah fitur baru
- Memperbaiki bug
- Meningkatkan dokumentasi
- Menambah dataset

## 📧 Support

Jika ada pertanyaan atau masalah, silakan buka issue di repository ini.

---

**Dibuat dengan ❤️ untuk membantu deteksi dini kanker payudara**
