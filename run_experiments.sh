#!/bin/bash
# =========================================================================
# RUN ALL PIPELINE EXPERIMENTS
# =========================================================================
set -e

# Configurations
export OCR_EPOCHS=${OCR_EPOCHS:-50}
export DRY_RUN=${DRY_RUN:-False}
export NUM_WORKERS=${NUM_WORKERS:-8}

echo "=================================================="
echo " RUNNING HYBRID OCR RESEARCH PIPELINE ON GPU"
echo " DRY_RUN: $DRY_RUN, EPOCHS: $OCR_EPOCHS"
echo "=================================================="

echo "[Phase 1] Running Edge Preprocessing Ablation..."
python3 edge_preprocessing_ablation.py

echo "[Phase 2] Running Component Ablation..."
python3 component_ablation.py

echo "[Phase 3] Running Cross-Dataset Validation..."
python3 cross_dataset_validation.py

echo "[Phase 4] Running Statistical Significance..."
python3 statistical_significance.py

echo "[Phase 5B] Running Feature Map Visualization..."
python3 feature_map_analysis.py

echo "[Phase 5E] Running Edge Profiling..."
python3 edge_profiling.py

echo "=================================================="
echo " ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=================================================="
