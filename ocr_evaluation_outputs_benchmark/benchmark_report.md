# Laporan Benchmarking Terkontrol (Matriks 3x2)
## Klasifikasi Karakter Terisolasi untuk Edge Device (Raw vs. Skeletonized)

### Protokol Eksperimen
- **Dataset**: Chars74K Paired (Raw & Skeletonized, 64x64, Grayscale)
- **Split**: Train 6164 | Val 770 | Test 771 (seed=42)
- **Epochs**: 50 (early stopping patience=10)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-2)
- **Scheduler**: CosineAnnealingLR
- **Loss**: CrossEntropyLoss

### Hasil Perbandingan Komparatif

| Model       | Input Type   | Referensi                  |   Parameters |   Strict Acc (%) |   Tolerant Acc (%) |   Latency (ms) |
|:------------|:-------------|:---------------------------|-------------:|-----------------:|-------------------:|---------------:|
| ResNet18    | Raw          | He et al., CVPR 2016       |   11,202,046 |            80.29 |              87.81 |         0.0415 |
| ResNet18    | Skeleton     | He et al., CVPR 2016       |   11,202,046 |            75.62 |              82.36 |         0.0363 |
| MobileNetV3 | Raw          | Howard et al., ICCV 2019   |    4,281,166 |            73.15 |              79.64 |         0.0838 |
| MobileNetV3 | Skeleton     | Howard et al., ICCV 2019   |    4,281,166 |            63.29 |              70.95 |         0.0666 |
| Proposed_1M | Raw          | Custom (SE + Dilated Conv) |    1,074,987 |            84.31 |              91.18 |         2.3683 |
| Proposed_1M | Skeleton     | Custom (SE + Dilated Conv) |    1,074,987 |            79.9  |              87.55 |         2.4464 |

### Referensi Arsitektur

| Model | Referensi | Deskripsi Arsitektur |
|---|---|---|
| **ResNet-18** | He et al., CVPR 2016 | Standar industri mid-weight, residual connection |
| **MobileNetV3-Large** | Howard et al., ICCV 2019 | Standar industri edge-optimized dengan MBConv |
| **Proposed_1M** | Custom | Model usulan: dilated conv + SE attention blocks |
