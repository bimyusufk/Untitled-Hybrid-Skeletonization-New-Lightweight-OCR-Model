# Laporan Hasil Ablasi Preprocessing Tepi
## Analisis Pengaruh Metode Ekstraksi Kontur Terhadap Kinerja SuperHybridCNN

### Protokol Eksperimen
- **Dataset**: Chars74K (64x64, Grayscale, Preprocessed on-the-fly)
- **Split**: Train 6164 | Val 770 | Test 771 (seed=42)
- **Epochs**: 50 (early stopping patience=10)
- **Model**: SuperHybridCNN (1.16M params, 12 topology/Hu features)

### Hasil Perbandingan Komparatif

| Model                 | Mode      |   Parameters |   Strict Acc (%) |   Tolerant Acc (%) |   Latency (ms) |
|:----------------------|:----------|-------------:|-----------------:|-------------------:|---------------:|
| SuperHybrid_Raw       | RAW       |    1,168,382 |            84.95 |              90.92 |         0.0405 |
| SuperHybrid_Gradient  | GRADIENT  |    1,168,382 |            86.25 |              92.74 |         0.0417 |
| SuperHybrid_Canny     | CANNY     |    1,168,382 |            85.21 |              91.7  |         0.0415 |
| SuperHybrid_Sobel     | SOBEL     |    1,168,382 |            85.21 |              91.57 |         0.0404 |
| SuperHybrid_Laplacian | LAPLACIAN |    1,168,382 |            85.47 |              92.09 |         0.0411 |

