#!/bin/bash
# =========================================================================
# Setup Script untuk Vast.ai GPU Instance
# Jalankan: bash setup_vastai.sh
# =========================================================================
set -e

echo "============================================="
echo " SETUP VAST.AI GPU ENVIRONMENT"
echo "============================================="

echo "[1/4] Upgrading pip..."
pip install --upgrade pip --quiet

echo "[2/4] Installing Python dependencies..."
pip install -r requirements.txt --quiet

echo "[3/4] Verifying GPU availability..."
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'  GPUs detected: {len(gpus)}')
for g in gpus:
    print(f'    - {g.name} ({g.device_type})')
if not gpus:
    print('  WARNING: No GPU found! Training will run on CPU.')
"

echo "[4/4] Verifying dataset structure..."
python -c "
import os
raw_dir = 'datasets/raw'
csv_path = 'datasets/annotations.csv'
ok = True
if not os.path.exists(csv_path):
    print(f'  ERROR: {csv_path} not found!')
    ok = False
if not os.path.isdir(raw_dir):
    print(f'  ERROR: {raw_dir} not found!')
    ok = False
else:
    folders = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    print(f'  Dataset folders: {len(folders)}')
if ok:
    print('  Dataset structure OK.')
"

echo ""
echo "============================================="
echo " SETUP COMPLETE"
echo " Jalankan: python run_all_experiments.py"
echo "============================================="
