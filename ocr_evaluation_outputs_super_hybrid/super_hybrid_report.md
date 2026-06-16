# Laporan Benchmarking Super Hybrid CNN
## Strategi Mengalahkan Model Berat (ResNet-18 & Proposed_1M) dengan Model Ringan Ultra Cepat

### Protokol Eksperimen
- **Dataset**: Chars74K (64x64, Grayscale, Preprocessed on-the-fly)
- **Split**: Train 6164 | Val 770 | Test 771 (seed=42)
- **Epochs**: 2 (early stopping patience=2)
- **Online Augmentation**: Random Rotation (+/-10 deg), Random Translation (+/-10%)
- **Fitur Geometris/Topologi**: 12 Fitur (5 Properti + 7 Hu Moments)

### Hasil Perbandingan Komparatif

| Model                        | Input Type             | Hybrid         | Parameters   |   Strict Acc (%) |   Tolerant Acc (%) |   Latency (ms) |
|:-----------------------------|:-----------------------|:---------------|:-------------|-----------------:|-------------------:|---------------:|
| SuperHybrid_Binary           | Clean Binary           | YES (12 feats) | 1,168,382    |             1.56 |               1.56 |         1.4029 |
| SuperHybrid_Gradient         | Morphological Gradient | YES (12 feats) | 1,168,382    |             0    |               0    |         2.0112 |
| Gradient_CNN_Hybrid_Baseline | Morphological Gradient | YES (5 feats)  | 556,500      |             0    |               0    |         0.7157 |
| Proposed_1M_Raw_Baseline     | Clean Binary           | NO             | 1,074,987    |             0    |               0    |         1.3023 |

### Kesimpulan & Temuan Utama
1. **Akurasi**: Apakah penskalaan lebar saluran (`[32, 64, 128]`) ditambah 12 fitur geometris (termasuk Hu moments) dan augmentasi online berhasil mengalahkan model dilated Proposed_1M dan SOTA ResNet-18?
2. **Latensi**: Memverifikasi keunggulan latensi model penskalaan non-dilasi yang diproyeksikan berada di kisaran ~0.02 ms (sub-milidetik).
