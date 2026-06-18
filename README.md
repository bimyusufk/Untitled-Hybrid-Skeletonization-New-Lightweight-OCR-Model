# TopoGrad-Net — Morphological Gradient OCR

> Repositori implementasi model **TopoGrad-Net** untuk pengenalan karakter alfanumerik (62 kelas) berbasis CNN, Preprocessing Morfologi, dan Fitur Topologi.
>
> Proyek UAS Pengolahan dan Analisis Citra Digital (PACD) — Universitas Padjadjaran.

---

## 🚀 Cara Cepat Uji Coba (GUI Client)

Untuk memudahkan pengujian secara visual, telah disediakan aplikasi client berbasis desktop (GUI). Dosen penguji dapat langsung mencoba model dengan langkah berikut:

### 1. Jalankan Aplikasi GUI
Pastikan virtual environment aktif dan library sudah diinstal (lihat bagian [Instalasi](#-instalasi-virtual-environment) di bawah), lalu jalankan:

```bash
python gui_app.py
```

### 2. Cara Menggunakan
1. Klik tombol **"Open Image File"**.
2. Pilih file gambar karakter alfanumerik (dapat menggunakan gambar di folder `datasets/raw/...`).
3. Aplikasi akan menampilkan visualisasi tahapan pipeline secara real-time:
   - **Raw Image**: Gambar masukan asli.
   - **Otsu + Hole Filling**: Hasil binerisasi otomatis dan penambalan lubang kecil.
   - **Morphological Gradient**: Ekstraksi tepi struktural (input visual CNN).
   - **Prediction Output**: Karakter hasil prediksi beserta nilai keyakinan (Confidence %).
   - **12-dim Topological Features**: Nilai numerik 5 Region Properties + 7 Hu Moments.

---

## 🛠️ Instalasi (Virtual Environment)

Sebelum menjalankan aplikasi atau melatih model, siapkan environment Python dengan langkah berikut:

```bash
# 1. Buat Virtual Environment
python -m venv venv

# 2. Aktifkan Virtual Environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install Dependensi Utama
pip install -r requirements.txt
```

*Catatan: Memerlukan Python ≥ 3.10.*

---

## 📊 Pelatihan & Evaluasi Ulang (Command Line)

Jika ingin melatih kembali model dari awal atau memverifikasi metrik evaluasi penuh:

```bash
python super_hybrid_benchmarking.py
```

*   **Verifikasi Cepat (Dry-Run)**: Jika ingin memverifikasi kode tanpa menunggu training penuh (hanya 2 epoch):
    ```bash
    # Windows:
    $env:DRY_RUN="true"; python super_hybrid_benchmarking.py
    
    # Linux/macOS:
    DRY_RUN=true python super_hybrid_benchmarking.py
    ```

Semua hasil evaluasi (laporan akurasi per kelas, confusion matrix, kurva training) akan otomatis disimpan di folder `ocr_evaluation_outputs_super_hybrid/`.

---

## 🏗️ Struktur Folder Utama

```
.
├── gui_app.py                     # Aplikasi client desktop GUI (Tkinter)
├── super_hybrid_benchmarking.py   # Master script training & evaluasi
├── requirements.txt               # Daftar pustaka minimal yang dibutuhkan
├── README.md                      # Dokumentasi ini
├── datasets/
│   ├── annotations.csv            # Metadata pemetaan folder & label
│   └── raw/                       # Dataset citra grayscale asli
└── ocr_evaluation_outputs_super_hybrid/
    └── SuperHybrid_Gradient.pth   # Bobot model terbaik terlatih (TopoGrad-Net)
```