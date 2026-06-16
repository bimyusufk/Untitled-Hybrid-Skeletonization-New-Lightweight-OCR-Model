# Laporan Benchmarking Terkontrol (Matriks 3x2)
## Klasifikasi Karakter Terisolasi untuk Edge Device (Raw vs. Skeletonized)

### Protokol Eksperimen
- **Dataset**: Chars74K Paired (Raw & Skeletonized, 64x64, Grayscale)
- **Split**: Train 6164 | Val 770 | Test 771 (seed=42)
- **Epochs**: 2 (early stopping patience=2)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-2)
- **Scheduler**: CosineAnnealingLR
- **Loss**: CrossEntropyLoss

### Hasil Perbandingan Komparatif

| Model       | Input Type   | Referensi                  | Parameters   |   Strict Acc (%) |   Tolerant Acc (%) |   Latency (ms) |
|:------------|:-------------|:---------------------------|:-------------|-----------------:|-------------------:|---------------:|
| ResNet18    | Raw          | He et al., CVPR 2016       | 11,202,046   |             1.56 |               3.12 |         6.6661 |
| ResNet18    | Skeleton     | He et al., CVPR 2016       | 11,202,046   |             4.69 |               6.25 |         5.9165 |
| MobileNetV3 | Raw          | Howard et al., ICCV 2019   | 4,281,166    |             1.56 |               1.56 |         3.0286 |
| MobileNetV3 | Skeleton     | Howard et al., ICCV 2019   | 4,281,166    |             6.25 |               6.25 |         4.8798 |
| Proposed_1M | Raw          | Custom (SE + Dilated Conv) | 1,074,987    |             1.56 |               3.12 |         2.7027 |
| Proposed_1M | Skeleton     | Custom (SE + Dilated Conv) | 1,074,987    |             6.25 |               6.25 |         2.3315 |

### Referensi Arsitektur

| Model | Referensi | Deskripsi Arsitektur |
|---|---|---|
| **ResNet-18** | He et al., CVPR 2016 | Standar industri mid-weight, residual connection |
| **MobileNetV3-Large** | Howard et al., ICCV 2019 | Standar industri edge-optimized dengan MBConv |
| **Proposed_1M** | Custom | Model usulan: dilated conv + SE attention blocks |
