"""
=======================================================================
PHASE 2: COMPONENT ABLATION STUDY
=======================================================================
Systematic ablation study with 7 additive configurations:
- A1: CNN Only (Raw)                - Raw input, no features, no augmentation, no hole filling
- A2: CNN Only (Gradient)           - Morph gradient input, no features, no augmentation, no hole filling
- A3: CNN + 5 Region Props          - Morph gradient input, 5 region props, no augmentation, no hole filling
- A4: CNN + 7 Hu Moments            - Morph gradient input, 7 Hu moments, no augmentation, no hole filling
- A5: CNN + 12 Features             - Morph gradient input, 12 features, no augmentation, no hole filling
- A6: CNN + 12 Features + Aug       - Morph gradient input, 12 features, online augmentation, no hole filling
- A7: CNN + 12 Features + Aug + HF  - Morph gradient input, 12 features, online augmentation, hole filling (Full pipeline)
=======================================================================
"""

import os
import time
import random
import json
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from super_hybrid_benchmarking import (
    SEED, CLASS_LIST, NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR, IMAGE_SIZE,
    count_parameters, train_model, evaluate_model, train_test_split
)

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False

# Custom Preprocessing with optional Hole Filling
def preprocess_image_custom(raw_path, use_hole_fill=True):
    img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Border average check to handle black vs white background
    top_border = img_resized[0, :]
    bottom_border = img_resized[-1, :]
    left_border = img_resized[:, 0]
    right_border = img_resized[:, -1]
    all_border = np.concatenate([top_border, bottom_border, left_border, right_border])
    avg_border = np.mean(all_border)
    
    if avg_border > 127:
        _, img_bin = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, img_bin = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    if use_hole_fill:
        img_bool = img_bin > 0
        all_filled = ndimage.binary_fill_holes(img_bool)
        only_holes = np.logical_xor(all_filled, img_bool)
        labeled_holes, num_features = ndimage.label(only_holes)
        small_holes_mask = np.zeros_like(img_bool)
        
        for slice_index in range(1, num_features + 1):
            hole_area = np.sum(labeled_holes == slice_index)
            if hole_area <= 35:
                small_holes_mask = np.logical_or(small_holes_mask, (labeled_holes == slice_index))
                
        img_clean_bin = np.logical_or(img_bool, small_holes_mask).astype(np.uint8) * 255
        return img_clean_bin
    else:
        return img_bin

def load_dataset_ablation(csv_path="datasets/annotations.csv", raw_base_dir="datasets/raw", use_hole_fill=True):
    print(f"Loading raw dataset (Hole Fill={use_hole_fill}) from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotation file not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    X_bin_list = []
    y_list = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading (Hole Fill={use_hole_fill})"):
        folder_name = row['Folder Name']
        label_char = str(row['Label'])
        
        raw_folder = os.path.join(raw_base_dir, folder_name)
        if not os.path.exists(raw_folder):
            continue
            
        for img_name in sorted(os.listdir(raw_folder)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                raw_path = os.path.join(raw_folder, img_name)
                img_bin = preprocess_image_custom(raw_path, use_hole_fill=use_hole_fill)
                if img_bin is not None:
                    X_bin_list.append(img_bin)
                    y_list.append(CHAR_TO_IDX[label_char])
                    
    X_bin = np.array(X_bin_list, dtype=np.uint8)
    y = np.array(y_list)
    print(f"Dataset loaded: {X_bin.shape[0]} samples.")
    return X_bin, y

# Extract custom subset of features
def extract_ablation_features(img_bin, feat_type):
    """
    feat_type: "none" (0 feats), "region" (5 region props), "hu" (7 Hu Moments), "all" (12 feats)
    """
    if feat_type == "none":
        return np.array([], dtype=np.float32)
        
    binary = img_bin > 0
    
    # 5 region props
    region_feats = np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    if feat_type in ["region", "all"]:
        labeled = label(binary)
        props = regionprops(labeled)
        if len(props) > 0:
            props = sorted(props, key=lambda x: x.area, reverse=True)
            main_prop = props[0]
            euler = float(main_prop.euler_number)
            eccentricity = float(main_prop.eccentricity)
            minr, minc, maxr, maxc = main_prop.bbox
            h = max(1, maxr - minr)
            w = max(1, maxc - minc)
            aspect_ratio = float(w / h)
            extent = float(main_prop.extent)
            solidity = float(main_prop.solidity)
            
            euler = max(-5.0, min(5.0, euler))
            region_feats = np.array([euler, eccentricity, aspect_ratio, extent, solidity], dtype=np.float32)
            
    if feat_type == "region":
        return region_feats
        
    # 7 Hu Moments
    hu_feats = np.zeros(7, dtype=np.float32)
    if feat_type in ["hu", "all"]:
        moments = cv2.moments(img_bin)
        hu = cv2.HuMoments(moments).flatten()
        hu_log = []
        for i in range(7):
            val = hu[i]
            abs_val = abs(val)
            if abs_val > 0:
                log_val = -1.0 * np.sign(val) * np.log10(abs_val)
            else:
                log_val = 0.0
            if np.isnan(log_val) or np.isinf(log_val):
                log_val = 0.0
            hu_log.append(log_val)
        hu_feats = np.array(hu_log, dtype=np.float32)
        
    if feat_type == "hu":
        return hu_feats
        
    return np.concatenate([region_feats, hu_feats])

class ComponentAblationDataset(Dataset):
    def __init__(self, X_bin, y, is_training=False, use_gradient=True, feat_type="all", use_augmentation=False):
        self.X_bin = X_bin
        self.y = y
        self.is_training = is_training
        self.use_gradient = use_gradient
        self.feat_type = feat_type
        self.use_augmentation = use_augmentation
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        img_bin = self.X_bin[idx].copy()
        
        # Online augmentation (Rotation & Translation)
        if self.is_training and self.use_augmentation:
            img_tensor = torch.tensor(img_bin, dtype=torch.float32).unsqueeze(0)
            angle = random.uniform(-10.0, 10.0)
            img_tensor = TF.rotate(img_tensor, angle)
            max_dx = int(0.1 * 64)
            max_dy = int(0.1 * 64)
            dx = random.randint(-max_dx, max_dx)
            dy = random.randint(-max_dy, max_dy)
            img_tensor = TF.affine(img_tensor, angle=0, translate=[dx, dy], scale=1.0, shear=0)
            img_bin = (img_tensor.squeeze(0).numpy() > 127).astype(np.uint8) * 255
            
        # Extract features based on config
        feats = extract_ablation_features(img_bin, self.feat_type)
        
        # Morphological Gradient
        if self.use_gradient:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img_out = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel)
        else:
            img_out = img_bin
            
        # Normalize to [-1.0, 1.0]
        img_out_norm = (img_out.astype(np.float32) / 255.0 - 0.5) / 0.5
        img_out_tensor = torch.tensor(img_out_norm, dtype=torch.float32).unsqueeze(0)
        
        feats_tensor = torch.tensor(feats, dtype=torch.float32)
        label_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        
        return img_out_tensor, feats_tensor, label_tensor

class SuperHybridCNNAblation(nn.Module):
    def __init__(self, num_classes=62, in_channels=1, feat_dim=12):
        super().__init__()
        # Conv Block 1: 64x64 -> 32x32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv Block 2: 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv Block 3: 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Initial FC
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        
        # Fusion Classifier Head
        self.feat_dim = feat_dim
        if self.feat_dim > 0:
            self.fc2 = nn.Linear(128 + feat_dim, 128)
            self.bn5 = nn.BatchNorm1d(128)
            self.drop2 = nn.Dropout(0.3)
            self.fc_out = nn.Linear(128, num_classes)
        else:
            self.fc_out = nn.Linear(128, num_classes)
            
    def forward(self, x_img, x_feats=None):
        x = self.pool1(self.relu(self.bn1(self.conv1(x_img))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.drop1(x)
        
        if self.feat_dim > 0:
            combined = torch.cat([x, x_feats], dim=1)
            x_fused = self.relu(self.bn5(self.fc2(combined)))
            x_fused = self.drop2(x_fused)
            out = self.fc_out(x_fused)
        else:
            out = self.fc_out(x)
        return out

def main():
    print("=" * 75)
    print("  PHASE 2: COMPONENT ABLATION STUDY")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load raw dataset without hole fill (for configs A1-A6)
    try:
        X_bin_no_hf, y = load_dataset_ablation(use_hole_fill=False)
        X_bin_hf, _ = load_dataset_ablation(use_hole_fill=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating mock dataset for verification...")
        X_bin_no_hf = np.random.choice([0, 255], size=(200, 64, 64)).astype(np.uint8)
        X_bin_hf = X_bin_no_hf.copy()
        y = np.random.randint(0, NUM_CLASSES, size=(200,))

    # Split Data
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y[temp_idx])
    
    # Splitted Datasets
    X_tr_no_hf, y_tr = X_bin_no_hf[train_idx], y[train_idx]
    X_val_no_hf, y_val = X_bin_no_hf[val_idx], y[val_idx]
    X_te_no_hf, y_te = X_bin_no_hf[test_idx], y[test_idx]
    
    X_tr_hf = X_bin_hf[train_idx]
    X_val_hf = X_bin_hf[val_idx]
    X_te_hf = X_bin_hf[test_idx]
    
    batch_size = 64
    epochs = int(os.getenv("OCR_EPOCHS", "50"))
    patience = 10
    output_dir = "ocr_evaluation_outputs_ablation_components"
    os.makedirs(output_dir, exist_ok=True)
    
    is_dry_run = os.getenv("DRY_RUN", "False").lower() == "true"
    if is_dry_run:
        print("\n[DRY RUN] Restricting to 2 epochs with mini subsets.")
        epochs = 2
        patience = 2
        X_tr_no_hf, y_tr = X_tr_no_hf[:128], y_tr[:128]
        X_val_no_hf, y_val = X_val_no_hf[:64], y_val[:64]
        X_te_no_hf, y_te = X_te_no_hf[:64], y_te[:64]
        X_tr_hf = X_tr_hf[:128]
        X_val_hf = X_val_hf[:64]
        X_te_hf = X_te_hf[:64]
        batch_size = 32
        
    # Define Ablation Configs
    configs = [
        ("A1_CNN_Only_Raw", X_tr_no_hf, X_val_no_hf, X_te_no_hf, False, "none", False),
        ("A2_CNN_Only_Gradient", X_tr_no_hf, X_val_no_hf, X_te_no_hf, True, "none", False),
        ("A3_CNN_RegionProps", X_tr_no_hf, X_val_no_hf, X_te_no_hf, True, "region", False),
        ("A4_CNN_HuMoments", X_tr_no_hf, X_val_no_hf, X_te_no_hf, True, "hu", False),
        ("A5_CNN_12Feats", X_tr_no_hf, X_val_no_hf, X_te_no_hf, True, "all", False),
        ("A6_CNN_12Feats_Aug", X_tr_no_hf, X_val_no_hf, X_te_no_hf, True, "all", True),
        ("A7_CNN_12Feats_Aug_HF", X_tr_hf, X_val_hf, X_te_hf, True, "all", True)
    ]
    
    results = {}
    
    num_workers = int(os.getenv("NUM_WORKERS", "4"))
    
    for config_name, x_tr, x_val, x_te, use_grad, feat_type, use_aug in configs:
        print(f"\nTraining Ablation Config: {config_name}")
        
        # Dataloaders
        train_loader = DataLoader(ComponentAblationDataset(x_tr, y_tr, is_training=True, use_gradient=use_grad, feat_type=feat_type, use_augmentation=use_aug),
                                  batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(ComponentAblationDataset(x_val, y_val, is_training=False, use_gradient=use_grad, feat_type=feat_type, use_augmentation=use_aug),
                                batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        test_loader = DataLoader(ComponentAblationDataset(x_te, y_te, is_training=False, use_gradient=use_grad, feat_type=feat_type, use_augmentation=use_aug),
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        
        feat_dim = 0
        if feat_type == "region":
            feat_dim = 5
        elif feat_type == "hu":
            feat_dim = 7
        elif feat_type == "all":
            feat_dim = 12
            
        model = SuperHybridCNNAblation(num_classes=NUM_CLASSES, feat_dim=feat_dim)
        is_hybrid = feat_dim > 0
        
        # Verify forward pass
        model = model.to(device)
        model.eval()
        dummy_img = torch.randn(2, 1, 64, 64).to(device)
        dummy_feats = torch.randn(2, max(1, feat_dim)).to(device)
        with torch.no_grad():
            if is_hybrid:
                _ = model(dummy_img, dummy_feats[:, :feat_dim])
            else:
                _ = model(dummy_img)
        print(f"  [OK] Forward pass verification successful for {config_name}")
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            device=device,
            model_name=config_name,
            patience=patience,
            is_hybrid=is_hybrid
        )
        
        # Save training curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_title(f"Loss: {config_name}", fontsize=12, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history["train_acc"], label="Train Acc", linewidth=2)
        axes[1].plot(history["val_acc"], label="Val Acc", linewidth=2)
        axes[1].set_title(f"Accuracy: {config_name}", fontsize=12, fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"training_curves_{config_name}.png"), dpi=160, bbox_inches="tight")
        plt.close()
        
        # Evaluate
        eval_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            model_name=config_name,
            is_hybrid=is_hybrid,
            output_dir=output_dir
        )
        
        results[config_name] = {
            "params": count_parameters(model),
            "strict_accuracy": eval_metrics["strict_accuracy"],
            "tolerant_accuracy": eval_metrics["tolerant_accuracy"],
            "avg_latency_ms": eval_metrics["avg_latency_ms"]
        }

    # =====================================================================
    # TABEL ABLASI KOMPONEN FINAL
    # =====================================================================
    print(f"\n\n{'='*95}")
    print("                       HASIL ABLASI KOMPONEN")
    print(f"{'='*95}")
    
    summary_data = []
    for m_name, res in results.items():
        summary_data.append({
            "Config": m_name,
            "Parameters": f"{res['params']:,}",
            "Strict Acc (%)": f"{res['strict_accuracy']:.2f}",
            "Tolerant Acc (%)": f"{res['tolerant_accuracy']:.2f}",
            "Latency (ms)": f"{res['avg_latency_ms']:.4f}"
        })
        
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(os.path.join(output_dir, "component_ablation_summary.csv"), index=False)
    
    # Save Report
    report_md_path = os.path.join(output_dir, "component_ablation_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Laporan Hasil Ablasi Komponen (Component Ablation)\n")
        f.write("## Analisis Pengaruh Masing-Masing Modul dalam Pipa Pemrosesan Terhadap Akurasi OCR\n\n")
        f.write("### Hasil Perbandingan Komparatif\n\n")
        f.write(summary_df.to_markdown(index=False) + "\n\n")
        
    with open(os.path.join(output_dir, "component_ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[OK] Component ablation reports saved to: {output_dir}/")

if __name__ == "__main__":
    main()
