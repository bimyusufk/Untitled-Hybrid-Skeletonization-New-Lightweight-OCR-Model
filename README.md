# TopoGrad-Net — Morphological Gradient OCR untuk Edge Computing

> Model OCR ringan berbasis CNN + Morphological Gradient + Fitur Topologi untuk pengenalan karakter alfanumerik (62 kelas: 0–9, a–z, A–Z).

---

## 📊 Performa Model

| Metrik | Nilai |
|---|---|
| **Strict Accuracy** | **84.95%** |
| **Tolerant Accuracy** (case-insensitive) | **91.83%** |
| **Total Parameter** | 1,168,382 (~1.17M) |
| **Latensi Inferensi** | ~2.15 ms/gambar (CPU) |
| **Input** | Grayscale 64×64 → Morphological Gradient |

---

## 🏗️ Arsitektur

TopoGrad-Net menggunakan pendekatan **dual-branch** yang menggabungkan fitur visual CNN dengan fitur topologi/geometris:

```
Input Gambar (64×64)
    │
    ├── [Preprocessing] Otsu Binarization + Hole Filling
    │         │
    │         ├── [Morphological Gradient] kernel 2×2
    │         │         │
    │         │    Visual Branch (CNN)
    │         │    ├── Conv2d(1→32) + BN + ReLU + MaxPool  → 32×32
    │         │    ├── Conv2d(32→64) + BN + ReLU + MaxPool → 16×16
    │         │    └── Conv2d(64→128) + BN + ReLU + MaxPool → 8×8
    │         │         │
    │         │    FC(8192→128) + BN + Dropout(0.4)
    │         │         │ → Visual Embedding (128-dim)
    │         │
    │         └── [Fitur Topologi] 12 Fitur
    │               ├── 5 Region Properties (Euler, Eccentricity,
    │               │   Aspect Ratio, Extent, Solidity)
    │               └── 7 Hu Moments (skala log, invarian
    │                   translasi/rotasi/skala)
    │                    │ → Topology Vector (12-dim)
    │
    └── [Fusion] Concatenate(128 + 12 = 140-dim)
              │
              FC(140→128) + BN + Dropout(0.3)
              │
              FC(128→62) → Output Prediksi (62 kelas)
```

---

## 📁 Struktur Repositori

```
.
├── super_hybrid_benchmarking.py   # Script utama: training + evaluasi
├── datasets/
│   ├── annotations.csv            # Metadata label gambar Chars74K
│   └── raw/                       # Folder gambar grayscale asli
├── ocr_evaluation_outputs_super_hybrid/
│   ├── SuperHybrid_Gradient.pth   # Bobot model terbaik (pre-trained)
│   ├── classification_report_*.txt
│   ├── confusion_matrix_*.png
│   ├── training_curves_*.png
│   ├── super_hybrid_results.json
│   └── super_hybrid_summary.csv
├── requirements.txt               # Daftar dependensi Python
├── README.md                      # Dokumentasi ini
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone Repositori

```bash
git clone <URL_REPOSITORI>
cd skeletonization_image_processing
```

### 2. Buat Virtual Environment & Install Dependensi

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

**Dependensi utama:**
- Python ≥ 3.10
- PyTorch ≥ 2.0
- OpenCV (`opencv-python-headless`) ≥ 4.8
- scikit-image ≥ 0.22
- scikit-learn ≥ 1.3
- pandas, matplotlib, tqdm, scipy

### 3. Siapkan Dataset

Letakkan dataset [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) di dalam folder `datasets/`:

```
datasets/
├── annotations.csv    # Format: Folder Name, Label
└── raw/
    ├── Sample001/     # Folder per kelas karakter
    │   ├── img001.png
    │   ├── img002.png
    │   └── ...
    ├── Sample002/
    └── ...
```

File `annotations.csv` berisi dua kolom: `Folder Name` dan `Label`, yang memetakan setiap folder ke karakter yang diwakilinya.

### 4. Training Model

Untuk melatih model dari awal:

```bash
python super_hybrid_benchmarking.py
```

**Konfigurasi default:**
- **Epochs**: 50 (bisa diubah via environment variable `OCR_EPOCHS`)
- **Batch size**: 64
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-2)
- **Scheduler**: CosineAnnealing
- **Early stopping**: patience=10
- **Augmentasi online**: Rotasi acak ±10°, translasi acak ±10%

**Untuk dry-run (verifikasi cepat):**

```bash
set DRY_RUN=true
python super_hybrid_benchmarking.py
```

**Untuk mengatur jumlah epoch secara manual:**

```bash
set OCR_EPOCHS=30
python super_hybrid_benchmarking.py
```

### 5. Hasil Output

Setelah training selesai, semua output tersimpan di folder `ocr_evaluation_outputs_super_hybrid/`:

| File | Deskripsi |
|---|---|
| `SuperHybrid_Gradient.pth` | Bobot model terbaik |
| `classification_report_*.txt` | Laporan presisi, recall, F1 per kelas |
| `confusion_matrix_*.png` | Visualisasi confusion matrix |
| `training_curves_*.png` | Grafik loss & akurasi per epoch |
| `prediction_samples_*.png` | Sampel prediksi visual |
| `super_hybrid_results.json` | Hasil metrik dalam format JSON |
| `super_hybrid_summary.csv` | Tabel ringkasan perbandingan semua model |

---

## 🔮 Inference & Client GUI

Kami menyediakan dua cara untuk menguji model TopoGrad-Net pada gambar baru: via Command Line Interface (CLI) atau Desktop GUI Client.

### 1. Menggunakan Command Line Interface (CLI)

Gunakan script `inference.py` untuk melakukan prediksi cepat pada satu file gambar:

```bash
python inference.py --image path/ke/gambar_karakter.png
```

Pilihan parameter:
- `--image`: Path ke gambar karakter input (wajib).
- `--model`: Path ke file bobot model `.pth` (default: `ocr_evaluation_outputs_super_hybrid/SuperHybrid_Gradient.pth`).

### 2. Menggunakan Desktop GUI Client (Tkinter)

Untuk menggunakan aplikasi interface berbasis desktop (GUI) yang interaktif, jalankan:

```bash
python gui_app.py
```

Fitur GUI Desktop:
- **Pilih & Buka Gambar**: Mengunggah gambar karakter secara instan melalui file dialog.
- **Visualisasi Preprocessing**: Menampilkan visualisasi Citra Asli (Raw), Citra Biner (Otsu + Hole Filling), dan Citra Gradien Morfologi (Input model) secara berdampingan.
- **Prediksi Cepat**: Menampilkan hasil huruf/angka prediksi model secara dinamis beserta nilai akurasi keyakinan (confidence score).
- **Vektor Fitur Topologi**: Menampilkan nilai numerik ekstraksi dari 12 fitur topologi & Hu Moments dalam bentuk tabel scrollable.

---

## 🛠️ Membuat Standalone Executable (.exe)

Anda dapat mengemas (compile) aplikasi Desktop GUI di atas menjadi file executable standalone (`.exe`) Windows sehingga aplikasi dapat dijalankan tanpa perlu menginstal Python di komputer target.

Jalankan script pembangun otomatis:

```bash
python build_exe.py
```

Script ini akan otomatis menginstal `pyinstaller` (jika belum ada) dan membuat executable di direktori:
`dist/TopoGrad_OCR.exe`

> **PENTING**: Ketika memindahkan atau mendistribusikan file `TopoGrad_OCR.exe`, pastikan folder bobot `ocr_evaluation_outputs_super_hybrid/` berada di satu direktori yang sama dengan file `.exe` tersebut agar model dapat dimuat dengan sukses.

---

## 🔬 Pipeline Preprocessing

Setiap gambar melewati pipeline berikut sebelum masuk ke model:

1. **Baca gambar** → grayscale
2. **Resize** → 64×64 piksel
3. **Otsu Binarization** → konversi otomatis ke hitam-putih (dengan deteksi background otomatis)
4. **Conditional Hole Filling** → menutup lubang kecil ≤35 piksel (mempertahankan lubang penting seperti pada huruf "o", "a", "e")
5. **Morphological Gradient** → `cv2.morphologyEx(MORPH_GRADIENT, kernel_2x2)` untuk mengekstrak tepi struktural
6. **Normalisasi** → rentang [-1.0, 1.0]

---

## 📚 Detail Teknis

### Fitur Topologi (12 dimensi)

| # | Fitur | Sumber | Deskripsi |
|---|---|---|---|
| 1 | Euler Number | Region Props | Jumlah objek minus jumlah lubang |
| 2 | Eccentricity | Region Props | Rasio panjang sumbu minor/mayor (0=lingkaran, 1=garis) |
| 3 | Aspect Ratio | Region Props | Rasio lebar/tinggi bounding box |
| 4 | Extent | Region Props | Rasio area objek/area bounding box |
| 5 | Solidity | Region Props | Rasio area objek/area convex hull |
| 6–12 | Hu Moments 1–7 | cv2.HuMoments | Momen invarian terhadap translasi, rotasi, dan skala (skala log) |

### Augmentasi Online (saat training)

- **Rotasi acak**: ±10 derajat
- **Translasi acak**: ±10% dari ukuran gambar (±6.4 piksel pada 64×64)

---

## ⚖️ Lisensi

Proyek ini dikembangkan untuk keperluan akademis pada mata kuliah Pengolahan dan Analisis Citra Digital, Universitas Padjadjaran.