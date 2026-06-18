# Laporan Hasil Ablasi Komponen (Component Ablation)
## Analisis Pengaruh Masing-Masing Modul dalam Pipa Pemrosesan Terhadap Akurasi OCR

### Hasil Perbandingan Komparatif

| Config                |   Parameters |   Strict Acc (%) |   Tolerant Acc (%) |   Latency (ms) |
|:----------------------|-------------:|-----------------:|-------------------:|---------------:|
| A1_CNN_Only_Raw       |    1,150,078 |            80.03 |              86.51 |         0.0408 |
| A2_CNN_Only_Gradient  |    1,150,078 |            80.29 |              87.55 |         0.0406 |
| A3_CNN_RegionProps    |    1,167,486 |            80.8  |              87.81 |         0.0407 |
| A4_CNN_HuMoments      |    1,167,742 |            81.58 |              87.55 |         0.0433 |
| A5_CNN_12Feats        |    1,168,382 |            80.8  |              87.29 |         0.041  |
| A6_CNN_12Feats_Aug    |    1,168,382 |            85.6  |              91.7  |         0.0394 |
| A7_CNN_12Feats_Aug_HF |    1,168,382 |            86.25 |              92.09 |         0.0406 |

