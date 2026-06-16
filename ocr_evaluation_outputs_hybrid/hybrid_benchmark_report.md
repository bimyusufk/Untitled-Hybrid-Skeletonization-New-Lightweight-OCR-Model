# Laporan Benchmarking Pengolahan Morfologi & Fitur Hybrid Topologi
## Optimasi Latensi dan Simplifikasi Arsitektur OCR untuk Edge Device

### Protokol Eksperimen
- **Dataset**: Chars74K (64x64, Grayscale, Preprocessed on-the-fly)
- **Split**: Train 6164 | Val 770 | Test 771 (seed=42)
- **Epochs**: 2 (early stopping patience=2)
- **Fitur Topologi**: Euler, Eccentricity, Aspect Ratio, Extent, Solidity

### Hasil Perbandingan Komparatif

| Model               | Input Type             | Hybrid (Topologi)   | Parameters   |   Strict Acc (%) |   Tolerant Acc (%) |   Latency (ms) |
|:--------------------|:-----------------------|:--------------------|:-------------|-----------------:|-------------------:|---------------:|
| Shallow_CNN_Only    | Clean Binary           | NO                  | 556,190      |             0    |               0    |         0.6111 |
| Shallow_CNN_Hybrid  | Clean Binary           | YES                 | 556,500      |             1.56 |               1.56 |         0.5648 |
| Gradient_CNN_Only   | Morphological Gradient | NO                  | 556,190      |             1.56 |               3.12 |         0.7607 |
| Gradient_CNN_Hybrid | Morphological Gradient | YES                 | 556,500      |             6.25 |               6.25 |         0.7191 |
| Proposed_1M_Raw     | Clean Binary           | NO                  | 1,074,987    |             1.56 |               3.12 |         1.1724 |

### Analisis Temuan
1. **Efisiensi Latensi**: Shallow CNN Only/Hybrid (~550k) berjalan dalam sub-milidetik, jauh melampaui Proposed_1M yang lambat akibat konvolusi dilasi.
2. **Peran Fitur Topologi**: Membandingkan model `Shallow_CNN_Only` vs `Shallow_CNN_Hybrid` untuk melihat kontribusi konkret 5 fitur geometris dalam meningkatkan akurasi model kecil.
3. **Pengaruh Gradient (Outline)**: Membandingkan performa model dengan input kontur terstandardisasi (`Gradient_CNN`) vs biner padat (`Shallow_CNN`).
