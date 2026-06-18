"""
=======================================================================
PHASE 1: EDGE PREPROCESSING ABLATION BENCHMARKING
=======================================================================
Trains the identical SuperHybridCNN architecture (1.16M params, 12 topology
features) on 5 different edge-extraction preprocessing methods:
1. SuperHybrid_Raw       - Clean binary image (no edge detection).
2. SuperHybrid_Gradient  - Morphological Gradient (2x2 kernel).
3. SuperHybrid_Canny     - Canny edge detection.
4. SuperHybrid_Sobel     - Sobel magnitude.
5. SuperHybrid_Laplacian - Laplacian of Gaussian.
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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# Import components from the main benchmarking script
from super_hybrid_benchmarking import (
    SEED, CLASS_LIST, NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR, IMAGE_SIZE,
    extract_super_features, load_binary_dataset, SuperHybridCNN,
    count_parameters, verify_model_forward, train_model, evaluate_model,
    train_test_split
)

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False

class PreprocessingAblationDataset(Dataset):
    def __init__(self, X_bin, y, is_training=False, preprocessing_mode="raw"):
        self.X_bin = X_bin
        self.y = y
        self.is_training = is_training
        self.preprocessing_mode = preprocessing_mode.lower()
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        img_bin = self.X_bin[idx].copy()
        
        # Online augmentation (Rotation & Translation)
        if self.is_training:
            img_tensor = torch.tensor(img_bin, dtype=torch.float32).unsqueeze(0)
            
            angle = random.uniform(-10.0, 10.0)
            img_tensor = TF.rotate(img_tensor, angle)
            
            max_dx = int(0.1 * 64)
            max_dy = int(0.1 * 64)
            dx = random.randint(-max_dx, max_dx)
            dy = random.randint(-max_dy, max_dy)
            img_tensor = TF.affine(img_tensor, angle=0, translate=[dx, dy], scale=1.0, shear=0)
            
            img_bin = (img_tensor.squeeze(0).numpy() > 127).astype(np.uint8) * 255
            
        # Extract 12 topological and geometric features
        feats = extract_super_features(img_bin)
        
        # Apply the selected preprocessing mode
        if self.preprocessing_mode == "gradient":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img_out = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel)
        elif self.preprocessing_mode == "canny":
            img_out = cv2.Canny(img_bin, 100, 200)
        elif self.preprocessing_mode == "sobel":
            sobelx = cv2.Sobel(img_bin, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_bin, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobelx**2 + sobely**2)
            img_out = (sobel_mag > 127).astype(np.uint8) * 255
        elif self.preprocessing_mode == "laplacian":
            lap = cv2.Laplacian(img_bin, cv2.CV_64F)
            img_out = (np.abs(lap) > 127).astype(np.uint8) * 255
        else: # raw
            img_out = img_bin
            
        # Normalize to [-1.0, 1.0]
        img_out_norm = (img_out.astype(np.float32) / 255.0 - 0.5) / 0.5
        img_out_tensor = torch.tensor(img_out_norm, dtype=torch.float32).unsqueeze(0)
        
        feats_tensor = torch.tensor(feats, dtype=torch.float32)
        label_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        
        return img_out_tensor, feats_tensor, label_tensor

def main():
    print("=" * 75)
    print("  PHASE 1: EDGE PREPROCESSING ABLATION BENCHMARKING")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    # --- Load Dataset ---
    try:
        X_bin, y = load_binary_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating mock dataset for verification...")
        X_bin = np.random.choice([0, 255], size=(200, 64, 64)).astype(np.uint8)
        y = np.random.randint(0, NUM_CLASSES, size=(200,))
        
    # --- Split Data ---
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=y
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=y[temp_idx]
    )
    
    X_train, y_train = X_bin[train_idx], y[train_idx]
    X_val, y_val = X_bin[val_idx], y[val_idx]
    X_test, y_test = X_bin[test_idx], y[test_idx]
    
    print(f"\nData Split:")
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    
    batch_size = 64
    epochs = int(os.getenv("OCR_EPOCHS", "50"))
    patience = 10
    output_dir = "ocr_evaluation_outputs_ablation_preprocessing"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check dry run mode
    is_dry_run = os.getenv("DRY_RUN", "False").lower() == "true"
    if is_dry_run:
        print("\n[DRY RUN] Restricting to 2 epochs with mini subsets.")
        epochs = 2
        patience = 2
        X_train, y_train = X_train[:128], y_train[:128]
        X_val, y_val = X_val[:64], y_val[:64]
        X_test, y_test = X_test[:64], y_test[:64]
        batch_size = 32

    # Save Preprocessing Comparison Grid
    print("\nSaving preprocessing visualization grid...")
    sample_img = X_train[0]
    
    # Compute variants for visualization
    grad_img = cv2.morphologyEx(sample_img, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    canny_img = cv2.Canny(sample_img, 100, 200)
    
    sobelx = cv2.Sobel(sample_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(sample_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_img = (sobel_mag > 127).astype(np.uint8) * 255
    
    lap = cv2.Laplacian(sample_img, cv2.CV_64F)
    lap_img = (np.abs(lap) > 127).astype(np.uint8) * 255
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    titles = ["Raw (Binary)", "Morph Gradient", "Canny Edge", "Sobel Mag", "Laplacian"]
    images = [sample_img, grad_img, canny_img, sobel_img, lap_img]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "preprocessing_comparison_grid.png"), dpi=150)
    plt.close()
    print(f"  [OK] Preprocessing grid saved to {output_dir}/preprocessing_comparison_grid.png")

    # Benchmarking configurations
    variants = [
        ("SuperHybrid_Raw", "raw"),
        ("SuperHybrid_Gradient", "gradient"),
        ("SuperHybrid_Canny", "canny"),
        ("SuperHybrid_Sobel", "sobel"),
        ("SuperHybrid_Laplacian", "laplacian")
    ]
    
    results = {}
    
    num_workers = int(os.getenv("NUM_WORKERS", "4"))
    
    for model_name, mode in variants:
        print(f"\nTraining configuration: {model_name} (Mode: {mode})")
        
        # Dataloaders
        train_loader = DataLoader(PreprocessingAblationDataset(X_train, y_train, is_training=True, preprocessing_mode=mode),
                                  batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(PreprocessingAblationDataset(X_val, y_val, is_training=False, preprocessing_mode=mode),
                                batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        test_loader = DataLoader(PreprocessingAblationDataset(X_test, y_test, is_training=False, preprocessing_mode=mode),
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        
        model = SuperHybridCNN(num_classes=NUM_CLASSES, feat_dim=12)
        verify_model_forward(model, model_name, device, is_hybrid=True, feat_dim=12)
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            device=device,
            model_name=model_name,
            patience=patience,
            is_hybrid=True
        )
        
        # Plot Curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_title(f"Loss: {model_name}", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history["train_acc"], label="Train Acc", linewidth=2)
        axes[1].plot(history["val_acc"], label="Val Acc", linewidth=2)
        axes[1].set_title(f"Accuracy: {model_name}", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"training_curves_{model_name}.png"), dpi=160, bbox_inches="tight")
        plt.close()
        
        # Evaluate
        eval_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            model_name=model_name,
            is_hybrid=True,
            output_dir=output_dir
        )
        
        results[model_name] = {
            "mode": mode,
            "params": count_parameters(model),
            "strict_accuracy": eval_metrics["strict_accuracy"],
            "tolerant_accuracy": eval_metrics["tolerant_accuracy"],
            "avg_latency_ms": eval_metrics["avg_latency_ms"]
        }

    # =====================================================================
    # TABEL KOMPARASI FINAL ABLASI PREPROCESSING
    # =====================================================================
    print(f"\n\n{'='*95}")
    print("                       HASIL ABLASI PREPROCESSING TEPI")
    print(f"{'='*95}")
    
    summary_data = []
    for m_name, res in results.items():
        summary_data.append({
            "Model": m_name,
            "Mode": res["mode"].upper(),
            "Parameters": f"{res['params']:,}",
            "Strict Acc (%)": f"{res['strict_accuracy']:.2f}",
            "Tolerant Acc (%)": f"{res['tolerant_accuracy']:.2f}",
            "Latency (ms)": f"{res['avg_latency_ms']:.4f}"
        })
        
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(os.path.join(output_dir, "preprocessing_ablation_summary.csv"), index=False)
    
    # Save Markdown Report
    report_md_path = os.path.join(output_dir, "preprocessing_ablation_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Laporan Hasil Ablasi Preprocessing Tepi\n")
        f.write("## Analisis Pengaruh Metode Ekstraksi Kontur Terhadap Kinerja SuperHybridCNN\n\n")
        f.write("### Protokol Eksperimen\n")
        f.write("- **Dataset**: Chars74K (64x64, Grayscale, Preprocessed on-the-fly)\n")
        f.write(f"- **Split**: Train {X_train.shape[0]} | Val {X_val.shape[0]} | Test {X_test.shape[0]} (seed={SEED})\n")
        f.write(f"- **Epochs**: {epochs} (early stopping patience={patience})\n")
        f.write("- **Model**: SuperHybridCNN (1.16M params, 12 topology/Hu features)\n\n")
        f.write("### Hasil Perbandingan Komparatif\n\n")
        f.write(summary_df.to_markdown(index=False) + "\n\n")
        
    with open(os.path.join(output_dir, "preprocessing_ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[OK] Ablation reports saved to: {output_dir}/")

if __name__ == "__main__":
    main()
