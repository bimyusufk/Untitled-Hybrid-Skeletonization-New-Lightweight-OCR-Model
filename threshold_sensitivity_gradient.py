# threshold_sensitivity_gradient.py
import os
import time
import random
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage as ndimage
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from super_hybrid_benchmarking import (
    CLASS_LIST, NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR, IMAGE_SIZE,
    SuperDataset, SuperHybridCNN, count_parameters, verify_model_forward,
    train_model, train_test_split, SEED
)

# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

def preprocess_image_with_threshold(raw_path, hole_threshold):
    img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
        
    img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
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
        
    img_bool = img_bin > 0
    all_filled = ndimage.binary_fill_holes(img_bool)
    only_holes = np.logical_xor(all_filled, img_bool)
    labeled_holes, num_features = ndimage.label(only_holes)
    small_holes_mask = np.zeros_like(img_bool)
    
    for slice_index in range(1, num_features + 1):
        hole_area = np.sum(labeled_holes == slice_index)
        if hole_area <= hole_threshold:
            small_holes_mask = np.logical_or(small_holes_mask, (labeled_holes == slice_index))
            
    img_clean_bin = np.logical_or(img_bool, small_holes_mask).astype(np.uint8) * 255
    return img_clean_bin, None

def load_binary_dataset_with_threshold(hole_threshold, csv_path="datasets/annotations.csv", raw_base_dir="/mnt/c/Users/Unpad-hci/Documents/Untitled-Hybrid-Skeletonization-New-Lightweight-OCR-Model/datasets/raw"):
    print(f"Loading raw dataset from {csv_path} with threshold {hole_threshold}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotation file not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    X_bin_list = []
    y_list = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing raw to clean binary (thr={hole_threshold})"):
        folder_name = row['Folder Name']
        label_char = str(row['Label'])
        
        raw_folder = os.path.join(raw_base_dir, folder_name)
        if not os.path.exists(raw_folder):
            continue
            
        for img_name in sorted(os.listdir(raw_folder)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                raw_path = os.path.join(raw_folder, img_name)
                img_bin, _ = preprocess_image_with_threshold(raw_path, hole_threshold)
                if img_bin is not None:
                    X_bin_list.append(img_bin)
                    y_list.append(CHAR_TO_IDX[label_char])
                    
    X_bin = np.array(X_bin_list, dtype=np.uint8)
    y = np.array(y_list)
    print(f"Dataset loaded: {X_bin.shape[0]} samples.")
    return X_bin, y

def evaluate_model_custom(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, feats, labels in test_loader:
            images = images.to(device)
            feats = feats.to(device)
            outputs = model(images, feats)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    y_true_chars = [IDX_TO_CHAR[idx] for idx in y_true]
    y_pred_chars = [IDX_TO_CHAR[idx] for idx in y_pred]
    
    strict_correct = 0
    case_error_but_char_correct = 0
    total_samples = len(y_true)
    
    for true_char, pred_char in zip(y_true_chars, y_pred_chars):
        if true_char == pred_char:
            strict_correct += 1
        elif true_char.lower() == pred_char.lower():
            case_error_but_char_correct += 1
            
    strict_accuracy = (strict_correct / total_samples) * 100
    tolerant_accuracy = ((strict_correct + case_error_but_char_correct) / total_samples) * 100
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    return strict_accuracy, tolerant_accuracy, macro_f1

def main():
    print("=" * 75)
    print("  HOLE-FILLING THRESHOLD SENSITIVITY ANALYSIS (MORPH GRADIENT MODEL)")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    thresholds = [0, 5, 7, 15, 35, 70]
    results = []
    
    epochs = int(os.getenv("OCR_EPOCHS", "50"))
    patience = 10
    batch_size = 64
    
    is_dry_run = os.getenv("DRY_RUN", "False").lower() == "true"
    if is_dry_run:
        print("\n[DRY RUN] Restricting epochs & thresholds.")
        thresholds = [0, 35]
        epochs = 2
        patience = 2
        
    for thr in thresholds:
        print(f"\n\n====================== RUNNING FOR THRESHOLD: {thr} px ======================")
        set_seed(SEED)
        
        # Load dataset for this threshold
        X_bin, y = load_binary_dataset_with_threshold(thr)
        
        if is_dry_run:
            X_bin = X_bin[:128]
            y = y[:128]
            batch_size = 32
            
        # Split Data
        indices = np.arange(len(y))
        train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=y)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y[temp_idx])
        
        X_train, y_train = X_bin[train_idx], y[train_idx]
        X_val, y_val = X_bin[val_idx], y[val_idx]
        X_test, y_test = X_bin[test_idx], y[test_idx]
        
        # Dataloaders
        train_loader = DataLoader(
            SuperDataset(X_train, y_train, is_training=True, use_gradient=True, is_super_hybrid=True),
            batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            SuperDataset(X_val, y_val, is_training=False, use_gradient=True, is_super_hybrid=True),
            batch_size=batch_size, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            SuperDataset(X_test, y_test, is_training=False, use_gradient=True, is_super_hybrid=True),
            batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        # Instantiate model
        model = SuperHybridCNN(NUM_CLASSES, feat_dim=12)
        
        # Train model
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            device=device,
            model_name=f"SuperHybrid_Grad_Thr_{thr}",
            patience=patience,
            is_hybrid=True
        )
        
        # Evaluate model
        strict_acc, tolerant_acc, macro_f1 = evaluate_model_custom(model, test_loader, device)
        print(f"\nResults for Threshold {thr} px:")
        print(f"  Strict Accuracy:  {strict_acc:.2f}%")
        print(f"  Tolerant Accuracy: {tolerant_acc:.2f}%")
        print(f"  Macro F1-Score:    {macro_f1:.4f}")
        
        results.append({
            "Threshold": thr,
            "Strict Acc (%)": f"{strict_acc:.2f}",
            "Tolerant Acc (%)": f"{tolerant_acc:.2f}",
            "Macro F1": f"{macro_f1:.4f}"
        })
        
    df_results = pd.DataFrame(results)
    print("\n\n" + "=" * 50)
    print("FINAL SENSITIVITY RESULTS:")
    print("=" * 50)
    print(df_results.to_string(index=False))
    
    output_dir = "ocr_evaluation_outputs_super_hybrid"
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(os.path.join(output_dir, "threshold_sensitivity_gradient_results.csv"), index=False)
    print(f"\nResults saved to {output_dir}/threshold_sensitivity_gradient_results.csv")

if __name__ == "__main__":
    main()
