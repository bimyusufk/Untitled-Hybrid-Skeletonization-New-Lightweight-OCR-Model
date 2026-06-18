# Laporan Analisis Signifikansi Statistik (Statistical Significance Report)
## Verifikasi Keunggulan SuperHybrid_Gradient Terhadap Model Baseline Melalui Multi-Seed Runs

### Hasil Perbandingan Komparatif

| Model                        | Accuracy (Seed Runs)                   | Mean ± Std    | Wilcoxon p-val   | Paired T-test p-val   |
|:-----------------------------|:---------------------------------------|:--------------|:-----------------|:----------------------|
| SuperHybrid_Gradient         | 84.82%, 87.42%, 86.12%, 85.47%, 86.64% | 86.10 ± 0.90% | -                | -                     |
| SuperHybrid_Binary           | 84.95%, 87.42%, 85.86%, 84.57%, 85.34% | 85.63 ± 0.99% | 0.2500           | 0.1635                |
| Gradient_CNN_Hybrid_Baseline | 78.99%, 82.36%, 80.03%, 79.90%, 79.25% | 80.10 ± 1.19% | 0.0625           | 0.0001                |
| Proposed_1M_Raw_Baseline     | 85.34%, 85.99%, 85.60%, 82.88%, 83.27% | 84.62 ± 1.28% | 0.1250           | 0.1015                |

### Metodologi Analisis Statistik
- **Jumlah Seeds**: 5 (Seeds: [42, 123, 456, 789, 1024])
- **Uji Statistik**: Wilcoxon signed-rank test (Uji non-parametrik berpasangan) dan Paired t-test (Uji parametrik berpasangan).
- **Tingkat Signifikansi (Alpha)**: 0.05. Jika p-value < 0.05, perbedaan performa dianggap signifikan secara statistik.
