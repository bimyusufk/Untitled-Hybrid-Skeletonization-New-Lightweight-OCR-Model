# Laporan Benchmarking Pengolahan Morfologi & Fitur Hybrid Topologi
## Optimasi Latensi dan Simplifikasi Arsitektur OCR untuk Edge Device

### Protokol Eksperimen
- **Dataset**: Chars74K (64x64, Grayscale, Preprocessed on-the-fly)
- **Split**: Train 6164 | Val 770 | Test 771 (seed=42)
- **Epochs**: 50 (early stopping patience=10)
- **Fitur Topologi**: Euler, Eccentricity, Aspect Ratio, Extent, Solidity

### Hasil Perbandingan Komparatif

| Model               | Input Type             | Hybrid (Topologi)   |   Parameters |   Strict Acc (%) |   Tolerant Acc (%) |   Latency (ms) |
|:--------------------|:-----------------------|:--------------------|-------------:|-----------------:|-------------------:|---------------:|
| Shallow_CNN_Only    | Clean Binary           | NO                  |      556,190 |            78.47 |              85.73 |         0.0101 |
| Shallow_CNN_Hybrid  | Clean Binary           | YES                 |      556,500 |            79.51 |              86.38 |         0.0113 |
| Gradient_CNN_Only   | Morphological Gradient | NO                  |      556,190 |            78.73 |              85.73 |         0.0095 |
| Gradient_CNN_Hybrid | Morphological Gradient | YES                 |      556,500 |            79.9  |              86.9  |         0.0084 |
| Proposed_1M_Raw     | Clean Binary           | NO                  |    1,074,987 |            84.18 |              91.96 |         2.7774 |

### Analisis Temuan
1. **Efisiensi Latensi**: Shallow CNN Only/Hybrid (~550k) berjalan dalam sub-milidetik, jauh melampaui Proposed_1M yang lambat akibat konvolusi dilasi.
2. **Peran Fitur Topologi**: Membandingkan model `Shallow_CNN_Only` vs `Shallow_CNN_Hybrid` untuk melihat kontribusi konkret 5 fitur geometris dalam meningkatkan akurasi model kecil.
3. **Pengaruh Gradient (Outline)**: Membandingkan performa model dengan input kontur terstandardisasi (`Gradient_CNN`) vs biner padat (`Shallow_CNN`).
