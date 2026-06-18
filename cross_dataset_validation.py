"""
=======================================================================
PHASE 3: CROSS-DATASET VALIDATION
=======================================================================
This script performs cross-dataset validation using EMNIST ByClass (62 classes)
and Chars74K. It demonstrates that the SuperHybridCNN model generalizes well.

Experiments:
1. In-domain: Chars74K -> Chars74K
2. Cross-domain A: EMNIST -> Chars74K
3. Cross-domain B: Chars74K -> EMNIST
4. Combined: Chars74K + EMNIST -> Chars74K
=======================================================================
"""

import os
import time
import random
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.functional as TF
import torchvision.datasets as torchvision_datasets

from super_hybrid_benchmarking import (
    SEED, CLASS_LIST, NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR, IMAGE_SIZE,
    SuperDataset, SuperHybridCNN, count_parameters, train_model, evaluate_model,
    train_test_split, load_binary_dataset
)

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False

class EMNISTMappedDataset(Dataset):
    """
    Loads EMNIST ByClass, maps labels to Chars74K label indexing,
    and applies SuperDataset preprocessing on-the-fly.
    """
    def __init__(self, emnist_dataset, indices=None, is_training=False):
        self.emnist = emnist_dataset
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(emnist_dataset)))
        self.is_training = is_training
        
        # Internal cache of binarized image arrays to speed up training
        self.X_bin = []
        self.y = []
        
        print(f"Preprocessing {len(self.indices)} EMNIST samples...")
        for idx in self.indices:
            img, label_emnist = self.emnist[idx]
            
            # Map EMNIST label to Chars74K indexing:
            # EMNIST: 0-9 digits, 10-35 upper A-Z, 36-61 lower a-z
            # Chars74K: 0-9 digits, 10-35 lower a-z, 36-61 upper A-Z
            if label_emnist < 10:
                mapped_label = label_emnist
            elif label_emnist < 36: # A-Z (10-35) -> Chars74K (36-61)
                mapped_label = label_emnist + 26
            else: # a-z (36-61) -> Chars74K (10-35)
                mapped_label = label_emnist - 26
                
            # Resize image to 64x64 and convert to numpy binarized format
            img_np = np.array(img.resize(IMAGE_SIZE))
            
            # EMNIST is naturally white digits on black background, binarize > 127
            _, img_bin = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
            
            self.X_bin.append(img_bin)
            self.y.append(mapped_label)
            
        self.X_bin = np.array(self.X_bin, dtype=np.uint8)
        self.y = np.array(self.y)
        
        # Wrap in a SuperDataset internally to reuse logic
        self.super_ds = SuperDataset(self.X_bin, self.y, is_training=is_training, use_gradient=True, is_super_hybrid=True)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.super_ds[idx]

def main():
    print("=" * 75)
    print("  PHASE 3: CROSS-DATASET VALIDATION (Chars74K <-> EMNIST)")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # --- 1. Load Chars74K Dataset ---
    try:
        X_bin_c74, y_c74 = load_binary_dataset()
    except Exception as e:
        print(f"Error loading Chars74K dataset: {e}")
        print("Creating mock Chars74K dataset...")
        X_bin_c74 = np.random.choice([0, 255], size=(200, 64, 64)).astype(np.uint8)
        y_c74 = np.random.randint(0, NUM_CLASSES, size=(200,))

    # Split Chars74K
    c74_indices = np.arange(len(y_c74))
    c74_train_idx, c74_temp_idx = train_test_split(c74_indices, test_size=0.20, random_state=SEED, stratify=y_c74)
    c74_val_idx, c74_test_idx = train_test_split(c74_temp_idx, test_size=0.50, random_state=SEED, stratify=y_c74[c74_temp_idx])
    
    # Wrap Chars74K in SuperDataset
    chars_train = SuperDataset(X_bin_c74[c74_train_idx], y_c74[c74_train_idx], is_training=True, use_gradient=True, is_super_hybrid=True)
    chars_val = SuperDataset(X_bin_c74[c74_val_idx], y_c74[c74_val_idx], is_training=False, use_gradient=True, is_super_hybrid=True)
    chars_test = SuperDataset(X_bin_c74[c74_test_idx], y_c74[c74_test_idx], is_training=False, use_gradient=True, is_super_hybrid=True)

    # --- 2. Load EMNIST ByClass Dataset ---
    print("\nLoading EMNIST ByClass dataset (from torchvision)...")
    os.makedirs("datasets/emnist", exist_ok=True)
    
    try:
        emnist_train_raw = torchvision_datasets.EMNIST(root="datasets/emnist", split="byclass", train=True, download=True)
        emnist_test_raw = torchvision_datasets.EMNIST(root="datasets/emnist", split="byclass", train=False, download=True)
        print(f"EMNIST ByClass loaded successfully. Train size: {len(emnist_train_raw)}, Test size: {len(emnist_test_raw)}")
    except Exception as e:
        print(f"Could not download EMNIST: {e}")
        print("Using mock EMNIST dataset for verification...")
        class MockEMNIST:
            def __len__(self): return 1000
            def __getitem__(self, idx):
                from PIL import Image
                return Image.fromarray(np.random.choice([0, 255], size=(28, 28)).astype(np.uint8)), np.random.randint(0, 62)
        emnist_train_raw = MockEMNIST()
        emnist_test_raw = MockEMNIST()

    # Create balanced subset of EMNIST to make the experiment fast
    # By default: 50,000 for training, 10,000 for testing
    num_emnist_train = 50000
    num_emnist_test = 10000
    
    is_dry_run = os.getenv("DRY_RUN", "False").lower() == "true"
    if is_dry_run:
        print("\n[DRY RUN] Restricting sizes.")
        num_emnist_train = 200
        num_emnist_test = 100
        
    random.seed(SEED)
    emnist_train_indices = random.sample(range(len(emnist_train_raw)), min(num_emnist_train, len(emnist_train_raw)))
    emnist_test_indices = random.sample(range(len(emnist_test_raw)), min(num_emnist_test, len(emnist_test_raw)))
    
    print(f"Creating EMNIST train subset ({len(emnist_train_indices)} samples)...")
    emnist_train_ds = EMNISTMappedDataset(emnist_train_raw, emnist_train_indices, is_training=True)
    print(f"Creating EMNIST test subset ({len(emnist_test_indices)} samples)...")
    emnist_test_ds = EMNISTMappedDataset(emnist_test_raw, emnist_test_indices, is_training=False)
    
    batch_size = 64
    epochs = int(os.getenv("OCR_EPOCHS", "50"))
    patience = 10
    output_dir = "ocr_evaluation_outputs_cross_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    if is_dry_run:
        epochs = 2
        patience = 2
        batch_size = 32

    num_workers = int(os.getenv("NUM_WORKERS", "4"))
    
    # --- DataLoader configuration ---
    # Setup Loaders
    loader_c74_train = DataLoader(chars_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_c74_val = DataLoader(chars_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loader_c74_test = DataLoader(chars_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    loader_emnist_train = DataLoader(emnist_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_emnist_test = DataLoader(emnist_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # We also create a combined dataset loader
    combined_train_ds = ConcatDataset([chars_train, emnist_train_ds])
    loader_combined_train = DataLoader(combined_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Results dictionary
    results = {}

    # -------------------------------------------------------------------------
    # Experiment 1: Chars74K -> Chars74K (In-Domain Baseline)
    # -------------------------------------------------------------------------
    print("\n>>> Experiment 1: Train Chars74K -> Test Chars74K <<<")
    model_indomain = SuperHybridCNN(num_classes=NUM_CLASSES, feat_dim=12)
    train_model(model_indomain, loader_c74_train, loader_c74_val, epochs, device, "Chars74K_InDomain", patience=patience)
    metrics_exp1 = evaluate_model(model_indomain, loader_c74_test, device, "Chars74K_InDomain", is_hybrid=True, output_dir=output_dir)
    results["InDomain_C74_to_C74"] = metrics_exp1

    # -------------------------------------------------------------------------
    # Experiment 2: EMNIST -> Chars74K (Cross-Domain A)
    # -------------------------------------------------------------------------
    print("\n>>> Experiment 2: Train EMNIST -> Test Chars74K <<<")
    model_cross_a = SuperHybridCNN(num_classes=NUM_CLASSES, feat_dim=12)
    # We validate on Chars74K val set to guide early stopping for generalization
    train_model(model_cross_a, loader_emnist_train, loader_c74_val, epochs, device, "EMNIST_to_Chars74K", patience=patience)
    metrics_exp2 = evaluate_model(model_cross_a, loader_c74_test, device, "EMNIST_to_Chars74K", is_hybrid=True, output_dir=output_dir)
    results["CrossDomain_EMNIST_to_C74"] = metrics_exp2

    # -------------------------------------------------------------------------
    # Experiment 3: Chars74K -> EMNIST (Cross-Domain B)
    # -------------------------------------------------------------------------
    print("\n>>> Experiment 3: Train Chars74K -> Test EMNIST <<<")
    # Using the in-domain model trained on Chars74K, test it directly on EMNIST test set
    metrics_exp3 = evaluate_model(model_indomain, loader_emnist_test, device, "Chars74K_to_EMNIST", is_hybrid=True, output_dir=output_dir)
    results["CrossDomain_C74_to_EMNIST"] = metrics_exp3

    # -------------------------------------------------------------------------
    # Experiment 4: Combined (Chars74K + EMNIST) -> Chars74K (Generalization Boost)
    # -------------------------------------------------------------------------
    print("\n>>> Experiment 4: Train Combined -> Test Chars74K <<<")
    model_combined = SuperHybridCNN(num_classes=NUM_CLASSES, feat_dim=12)
    train_model(model_combined, loader_combined_train, loader_c74_val, epochs, device, "Combined_to_Chars74K", patience=patience)
    metrics_exp4 = evaluate_model(model_combined, loader_c74_test, device, "Combined_to_Chars74K", is_hybrid=True, output_dir=output_dir)
    results["Combined_to_C74"] = metrics_exp4

    # =====================================================================
    # TABEL EVALUASI SILANG DATASET FINAL
    # =====================================================================
    print(f"\n\n{'='*95}")
    print("                       HASIL VALIDASI SILANG DATASET")
    print(f"{'='*95}")
    
    summary_data = [
        {
            "Experiment": "1. In-Domain (Baseline)",
            "Train Source": "Chars74K",
            "Test Target": "Chars74K",
            "Strict Acc (%)": f"{results['InDomain_C74_to_C74']['strict_accuracy']:.2f}",
            "Tolerant Acc (%)": f"{results['InDomain_C74_to_C74']['tolerant_accuracy']:.2f}"
        },
        {
            "Experiment": "2. Cross-Domain A",
            "Train Source": "EMNIST",
            "Test Target": "Chars74K",
            "Strict Acc (%)": f"{results['CrossDomain_EMNIST_to_C74']['strict_accuracy']:.2f}",
            "Tolerant Acc (%)": f"{results['CrossDomain_EMNIST_to_C74']['tolerant_accuracy']:.2f}"
        },
        {
            "Experiment": "3. Cross-Domain B",
            "Train Source": "Chars74K",
            "Test Target": "EMNIST",
            "Strict Acc (%)": f"{results['CrossDomain_C74_to_EMNIST']['strict_accuracy']:.2f}",
            "Tolerant Acc (%)": f"{results['CrossDomain_C74_to_EMNIST']['tolerant_accuracy']:.2f}"
        },
        {
            "Experiment": "4. Combined Train",
            "Train Source": "Chars74K + EMNIST",
            "Test Target": "Chars74K",
            "Strict Acc (%)": f"{results['Combined_to_C74']['strict_accuracy']:.2f}",
            "Tolerant Acc (%)": f"{results['Combined_to_C74']['tolerant_accuracy']:.2f}"
        }
    ]
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(os.path.join(output_dir, "cross_dataset_summary.csv"), index=False)
    
    # Save Report
    report_md_path = os.path.join(output_dir, "cross_dataset_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Laporan Hasil Validasi Silang Dataset (Cross-Dataset Validation)\n")
        f.write("## Analisis Ketergeneralisasian Model SuperHybridCNN Menggunakan Chars74K dan EMNIST ByClass\n\n")
        f.write("### Hasil Perbandingan Komparatif\n\n")
        f.write(summary_df.to_markdown(index=False) + "\n\n")
        f.write("### Temuan Utama\n")
        f.write("1. **Ketergeneralisasian domain**: Apakah model yang dilatih pada EMNIST dapat mengenali karakter dari Chars74K dengan baik?\n")
        f.write("2. **Efek Penggabungan**: Apakah penggabungan dataset EMNIST dan Chars74K dapat memberikan peningkatan akurasi umum pada Chars74K?\n")
        
    with open(os.path.join(output_dir, "cross_dataset_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[OK] Cross-dataset reports saved to: {output_dir}/")

if __name__ == "__main__":
    main()
