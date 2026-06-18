# Laporan Hasil Validasi Silang Dataset (Cross-Dataset Validation)
## Analisis Ketergeneralisasian Model SuperHybridCNN Menggunakan Chars74K dan EMNIST ByClass

### Hasil Perbandingan Komparatif

| Experiment              | Train Source      | Test Target   |   Strict Acc (%) |   Tolerant Acc (%) |
|:------------------------|:------------------|:--------------|-----------------:|-------------------:|
| 1. In-Domain (Baseline) | Chars74K          | Chars74K      |            84.82 |              91.05 |
| 2. Cross-Domain A       | EMNIST            | Chars74K      |             7.65 |              12.45 |
| 3. Cross-Domain B       | Chars74K          | EMNIST        |             6.38 |               9.09 |
| 4. Combined Train       | Chars74K + EMNIST | Chars74K      |            81.06 |              88.72 |

### Temuan Utama
1. **Ketergeneralisasian domain**: Apakah model yang dilatih pada EMNIST dapat mengenali karakter dari Chars74K dengan baik?
2. **Efek Penggabungan**: Apakah penggabungan dataset EMNIST dan Chars74K dapat memberikan peningkatan akurasi umum pada Chars74K?
