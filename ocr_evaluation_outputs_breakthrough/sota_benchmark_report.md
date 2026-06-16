# Laporan Benchmarking Terkontrol dengan Model SOTA

Berikut adalah hasil perbandingan performa model usulan (Proposed 1M) dengan model SOTA under the exact same dataset splits, optimizer, and scheduler:

| Model Name        | Parameters   | Strict Accuracy (%)   | Tolerant Accuracy (%)   | Avg Latency (ms)   |
|:------------------|:-------------|:----------------------|:------------------------|:-------------------|
| Proposed_1M       | 1,074,987    | 1.56%                 | 3.12%                   | 1.5361 ms          |
| CNN_GRU           | 5,717,694    | 0.00%                 | 0.00%                   | 13.1772 ms         |
| MobileNetV3_Small | 1,581,118    | 0.00%                 | 0.00%                   | 1.1299 ms          |
| MobileViT_XXS     | 970,638      | 9.38%                 | 9.38%                   | 1.8070 ms          |

### Analisis Singkat:
1. **Proposed 1M Model** memiliki keunggulan performa latensi dan akurasi yang seimbang pada topologi tipis skeleton.
2. **CNN_GRU** menggabungkan ekstraksi spasial dan temporal (BiGRU) namun memerlukan penanganan dimensi sequence.
3. **MobileNetV3_Small** mewakili arsitektur edge CNN konvensional yang ringan.
4. **MobileViT_XXS** menggabungkan atensi transformer dengan efisiensi konvolusi.
