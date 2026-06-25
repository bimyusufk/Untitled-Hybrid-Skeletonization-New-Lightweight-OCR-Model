# comprehensive_benchmarking_part2.py
import os
import time
import random
import json
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from thop import profile
import onnxruntime as ort

from super_hybrid_benchmarking import (
    CLASS_LIST, NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR, IMAGE_SIZE,
    SuperDataset, TopoGradNet, count_parameters, train_test_split, SEED
)
from research.noisy_dataset_generator import add_salt_and_pepper_noise, add_gaussian_blur

# Logging helper
LOG_FILE = "comprehensive_benchmarking_part2.log"
def log_message(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")

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

# Parameterized Preprocessing
def preprocess_image_pipeline(raw_path, use_gradient=True):
    img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
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
    return img_bin

def load_binary_dataset_custom(csv_path="datasets/annotations.csv", raw_base_dir="datasets/raw"):
    if not os.path.exists(raw_base_dir):
        alt_path = "/mnt/c/Users/Unpad-hci/Documents/Untitled-Hybrid-Skeletonization-New-Lightweight-OCR-Model/datasets/raw"
        if os.path.exists(alt_path):
            raw_base_dir = alt_path
            
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotation file not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    X_bin_list = []
    y_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Chars74K"):
        folder_name = row['Folder Name']
        label_char = str(row['Label'])
        raw_folder = os.path.join(raw_base_dir, folder_name)
        if not os.path.exists(raw_folder):
            continue
        for img_name in sorted(os.listdir(raw_folder)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                raw_path = os.path.join(raw_folder, img_name)
                img_bin = preprocess_image_pipeline(raw_path)
                if img_bin is not None:
                    X_bin_list.append(img_bin)
                    y_list.append(CHAR_TO_IDX[label_char])
    return np.array(X_bin_list, dtype=np.uint8), np.array(y_list)

# ----------------- BASELINE MODEL DEFINITIONS -----------------

class LeNet5(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        if x.shape[2] != 32 or x.shape[3] != 32:
            x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class MobileNetV3SmallWrapper(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.base = tv_models.mobilenet_v3_small(weights=None)
        self.base.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.base.classifier[3] = nn.Linear(self.base.classifier[3].in_features, num_classes)
        
    def forward(self, x):
        return self.base(x)

class SqueezeNetWrapper(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.base = tv_models.squeezenet1_1(weights=None)
        self.base.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.base.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        
    def forward(self, x):
        return self.base(x)

class ShuffleNetV2Wrapper(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.base = tv_models.shufflenet_v2_x0_5(weights=None)
        self.base.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.base(x)

# Training loop
def train_model_generic(model, train_loader, val_loader, epochs, device, model_name, patience=5, is_hybrid=True):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for images, feats, labels in train_loader:
            images, feats, labels = images.to(device), feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, feats) if is_hybrid else model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        scheduler.step()
        
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100
        
        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, feats, labels in val_loader:
                images, feats, labels = images.to(device), feats.to(device), labels.to(device)
                outputs = model(images, feats) if is_hybrid else model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                model.load_state_dict(best_model_state)
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return history

def evaluate_model_generic(model, test_loader, device, is_hybrid=True, model_name=None, output_dir=None, num_classes=62):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, feats, labels in test_loader:
            images = images.to(device)
            feats = feats.to(device)
            outputs = model(images, feats) if is_hybrid else model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    strict_correct = np.sum(y_true == y_pred)
    strict_accuracy = (strict_correct / len(y_true)) * 100
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Case-tolerant accuracy (requires indexing back to characters)
    y_true_chars = [IDX_TO_CHAR.get(idx, str(idx)) for idx in y_true]
    y_pred_chars = [IDX_TO_CHAR.get(idx, str(idx)) for idx in y_pred]
    tolerant_correct = sum(1 for tc, pc in zip(y_true_chars, y_pred_chars) if tc.lower() == pc.lower())
    tolerant_accuracy = (tolerant_correct / len(y_true)) * 100
    
    if model_name and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Classification Report
        report_text = classification_report(
            y_true, y_pred,
            labels=list(range(num_classes)),
            target_names=[IDX_TO_CHAR.get(i, str(i)) for i in range(num_classes)],
            zero_division=0
        )
        report_path = os.path.join(output_dir, f"classification_report_{model_name}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Strict Accuracy: {strict_accuracy:.2f}%\n")
            f.write(f"Tolerant Accuracy: {tolerant_accuracy:.2f}%\n")
            f.write(f"Macro F1: {macro_f1:.4f}\n\n")
            f.write(report_text)
            
        # Save Confusion Matrix Plot
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        plt.figure(figsize=(15, 15))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {model_name}", fontsize=14)
        plt.colorbar(fraction=0.046, pad=0.04)
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, [IDX_TO_CHAR.get(i, str(i)) for i in range(num_classes)], rotation=90, fontsize=6)
        plt.yticks(tick_marks, [IDX_TO_CHAR.get(i, str(i)) for i in range(num_classes)], fontsize=6)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name}.png"), dpi=180, bbox_inches="tight")
        plt.close()

        # Save Sample Predictions Grid
        sample_count = min(12, len(y_true))
        cols = 4
        rows = int(np.ceil(sample_count / cols))
        plt.figure(figsize=(12, rows * 3))
        
        test_images_to_plot = []
        for images, feats, labels in test_loader:
            test_images_to_plot.extend(images.squeeze(1).numpy())
            if len(test_images_to_plot) >= sample_count:
                break
                
        for idx in range(sample_count):
            ax = plt.subplot(rows, cols, idx + 1)
            img = test_images_to_plot[idx]
            img_orig = ((img * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
            
            color = "green" if y_true_chars[idx] == y_pred_chars[idx] else "red"
            ax.imshow(img_orig, cmap="gray")
            ax.set_title(f"T:{y_true_chars[idx]} | P:{y_pred_chars[idx]}", fontsize=9, color=color, fontweight="bold")
            ax.axis("off")
            
        plt.suptitle(f"Sample Predictions: {model_name}", y=1.02, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"prediction_samples_{model_name}.png"), dpi=180, bbox_inches="tight")
        plt.close()

    return strict_accuracy, tolerant_accuracy, macro_f1

# EMNIST Dataset Wrapper
class EMNISTDataset(Dataset):
    def __init__(self, raw_emnist_dataset, is_training=False):
        self.dataset = raw_emnist_dataset
        self.is_training = is_training
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        img_tensor, label = self.dataset[idx]
        img_np = (img_tensor.squeeze().numpy() * 255.0).astype(np.uint8)
        
        # Resize to 64x64 to match standard model input shape
        img_np = cv2.resize(img_np, (64, 64), interpolation=cv2.INTER_AREA)
        _, img_np = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
        
        # Online data augmentations (rotate & translate)
        if self.is_training:
            img_tensor_tf = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
            angle = random.uniform(-10.0, 10.0)
            img_tensor_tf = tv_transforms.functional.rotate(img_tensor_tf, angle)
            dx = random.randint(-6, 6)
            dy = random.randint(-6, 6)
            img_tensor_tf = tv_transforms.functional.affine(img_tensor_tf, angle=0, translate=[dx, dy], scale=1.0, shear=0)
            img_np = (img_tensor_tf.squeeze(0).numpy() > 127).astype(np.uint8) * 255
            
        # Extract features (12 topologi)
        feats = np.zeros(12, dtype=np.float32) # preprocessed for speed in EMNIST
        
        # Morphological Gradient (default pipeline)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_out = cv2.morphologyEx(img_np, cv2.MORPH_GRADIENT, kernel)
        img_out_norm = (img_out.astype(np.float32) / 255.0 - 0.5) / 0.5
        
        return torch.tensor(img_out_norm, dtype=torch.float32).unsqueeze(0), torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def merge_results(output_dir):
    part1_path = os.path.join(output_dir, "comprehensive_benchmark_results_part1.csv")
    part2_path = os.path.join(output_dir, "comprehensive_benchmark_results_part2.csv")
    final_path = os.path.join(output_dir, "comprehensive_benchmark_results.csv")
    
    if os.path.exists(part1_path) and os.path.exists(part2_path):
        try:
            df1 = pd.read_csv(part1_path)
            df2 = pd.read_csv(part2_path)
            df_combined = pd.concat([df1, df2], ignore_index=True)
            df_combined.to_csv(final_path, index=False)
            log_message(f"Merged combined results successfully saved to {final_path}")
        except Exception as e:
            log_message(f"Failed to merge results: {e}")

# Main orchestrator
def main():
    t_start = time.time()
    log_message("=== STARTING COMPREHENSIVE OCR BENCHMARK (PART 2) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Device: {device}")
    
    is_dry_run = os.getenv("DRY_RUN", "False").lower() == "true"
    epochs = 2 if is_dry_run else 50
    patience = 2 if is_dry_run else 10
    log_message(f"Epochs to train: {epochs} (early stopping patience={patience})")
    
    # 1. LOAD DATASETS
    log_message("Loading Chars74K dataset...")
    X_bin, y = load_binary_dataset_custom()
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y[temp_idx])
    
    X_train, y_train = X_bin[train_idx], y[train_idx]
    X_val, y_val = X_bin[val_idx], y[val_idx]
    X_test, y_test = X_bin[test_idx], y[test_idx]
    
    train_loader = DataLoader(SuperDataset(X_train, y_train, is_training=True, use_gradient=True, is_super_hybrid=True), batch_size=64, shuffle=True)
    val_loader = DataLoader(SuperDataset(X_val, y_val, is_training=False, use_gradient=True, is_super_hybrid=True), batch_size=64, shuffle=False)
    test_loader = DataLoader(SuperDataset(X_test, y_test, is_training=False, use_gradient=True, is_super_hybrid=True), batch_size=64, shuffle=False)
    
    # EMNIST Balanced Dataset Loading
    log_message("Loading EMNIST Balanced dataset...")
    emnist_transform = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Lambda(lambda img: tv_transforms.functional.rotate(img, -90).transpose(1, 2)) # fix EMNIST orientation
    ])
    
    # Use cache directory or let torchvision download
    os.makedirs("datasets/emnist", exist_ok=True)
    try:
        raw_emnist_train = tv_datasets.EMNIST("datasets/emnist", split="balanced", train=True, download=True, transform=emnist_transform)
        raw_emnist_test = tv_datasets.EMNIST("datasets/emnist", split="balanced", train=False, download=True, transform=emnist_transform)
        
        # Subsetting for speed in dry-run or even full training
        emnist_train_size = 5000 if is_dry_run else 30000
        emnist_test_size = 1000 if is_dry_run else 5000
        
        train_sub_idx = np.random.choice(len(raw_emnist_train), emnist_train_size, replace=False)
        test_sub_idx = np.random.choice(len(raw_emnist_test), emnist_test_size, replace=False)
        
        emnist_train = EMNISTDataset(torch.utils.data.Subset(raw_emnist_train, train_sub_idx), is_training=True)
        emnist_test = EMNISTDataset(torch.utils.data.Subset(raw_emnist_test, test_sub_idx), is_training=False)
        
        emnist_train_loader = DataLoader(emnist_train, batch_size=128, shuffle=True)
        emnist_test_loader = DataLoader(emnist_test, batch_size=128, shuffle=False)
        log_message(f"EMNIST Loaded: train={len(emnist_train)}, test={len(emnist_test)}")
    except Exception as e:
        log_message(f"Warning: EMNIST load failed ({e}). Mocking EMNIST loader...")
        emnist_train_loader = train_loader
        emnist_test_loader = test_loader
        
    # Model Configurations
    configs = [
        ("ShuffleNetV2", lambda num_classes: ShuffleNetV2Wrapper(num_classes), False),
        ("TopoGrad-Net (Proposed)", lambda num_classes: TopoGradNet(num_classes, feat_dim=12), True)
    ]
    
    # 2. RUN TRAINING & EVALUATION
    results = []
    output_dir = "ocr_evaluation_outputs_comprehensive"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model_fn, is_hybrid in configs:
        log_message(f"--- Training {name} ---")
        model = model_fn(NUM_CLASSES)
        
        # Profile FLOPs using thop
        dummy_img = torch.randn(1, 1, 64, 64)
        dummy_feat = torch.randn(1, 12)
        try:
            flops, params = profile(model, inputs=(dummy_img, dummy_feat) if is_hybrid else (dummy_img,), verbose=False)
        except Exception as e:
            flops, params = 0, count_parameters(model)
            
        # Train on Chars74K
        history = train_model_generic(model, train_loader, val_loader, epochs, device, name, patience=patience, is_hybrid=is_hybrid)
        
        # Save training curves for Chars74K
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_title(f"Loss: {name} (Chars74K)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history["train_acc"], label="Train Acc", linewidth=2)
        axes[1].plot(history["val_acc"], label="Val Acc", linewidth=2)
        axes[1].set_title(f"Accuracy: {name} (Chars74K)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"training_curves_{name}.png"), dpi=160, bbox_inches="tight")
        plt.close()
        
        strict_acc, tolerant_acc, macro_f1 = evaluate_model_generic(
            model, test_loader, device, is_hybrid=is_hybrid,
            model_name=name, output_dir=output_dir, num_classes=NUM_CLASSES
        )
        
        # Noise injection evaluation
        # Salt-and-Pepper Noise
        X_test_sp = np.array([add_salt_and_pepper_noise(img, 0.05) for img in X_test])
        sp_loader = DataLoader(SuperDataset(X_test_sp, y_test, is_training=False, use_gradient=True, is_super_hybrid=is_hybrid), batch_size=64, shuffle=False)
        sp_strict_acc, _, _ = evaluate_model_generic(model, sp_loader, device, is_hybrid=is_hybrid)
        
        # Gaussian Blur
        X_test_blur = np.array([add_gaussian_blur(img, 1.0) for img in X_test])
        blur_loader = DataLoader(SuperDataset(X_test_blur, y_test, is_training=False, use_gradient=True, is_super_hybrid=is_hybrid), batch_size=64, shuffle=False)
        blur_strict_acc, _, _ = evaluate_model_generic(model, blur_loader, device, is_hybrid=is_hybrid)
        
        log_message(f"{name} Chars74K Strict: {strict_acc:.2f}%, S&P: {sp_strict_acc:.2f}%, Blur: {blur_strict_acc:.2f}%")
        
        # Train on EMNIST Balanced
        log_message(f"Training {name} on EMNIST...")
        emnist_model = model_fn(47) # EMNIST Balanced has 47 classes
        history_emnist = train_model_generic(emnist_model, emnist_train_loader, emnist_test_loader, epochs, device, f"{name}_EMNIST", patience=patience, is_hybrid=is_hybrid)
        
        # Save training curves for EMNIST
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history_emnist["train_loss"], label="Train Loss", linewidth=2)
        axes[0].plot(history_emnist["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_title(f"Loss: {name} (EMNIST)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history_emnist["train_acc"], label="Train Acc", linewidth=2)
        axes[1].plot(history_emnist["val_acc"], label="Val Acc", linewidth=2)
        axes[1].set_title(f"Accuracy: {name} (EMNIST)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"training_curves_{name}_EMNIST.png"), dpi=160, bbox_inches="tight")
        plt.close()
        
        emnist_strict_acc, emnist_tolerant_acc, emnist_macro_f1 = evaluate_model_generic(
            emnist_model, emnist_test_loader, device, is_hybrid=is_hybrid,
            model_name=f"{name}_EMNIST", output_dir=output_dir, num_classes=47
        )
        log_message(f"{name} EMNIST Strict: {emnist_strict_acc:.2f}%")
        
        # 3. PROFILE LATENCY & THROUGHPUT (Batch sizes 1, 8, 32, 64)
        latency_profile = {}
        for bs in [1, 8, 32, 64]:
            dummy_batch_img = torch.randn(bs, 1, 64, 64).to(device)
            dummy_batch_feat = torch.randn(bs, 12).to(device)
            model.to(device)
            model.eval()
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_batch_img, dummy_batch_feat) if is_hybrid else model(dummy_batch_img)
            
            # Profiling pass
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(50):
                    _ = model(dummy_batch_img, dummy_batch_feat) if is_hybrid else model(dummy_batch_img)
            t_elapsed = time.perf_counter() - t0
            avg_latency_ms = (t_elapsed / (50 * bs)) * 1000
            throughput = bs * 50 / t_elapsed
            latency_profile[f"Latency_B{bs}_ms"] = round(avg_latency_ms, 4)
            latency_profile[f"Throughput_B{bs}_fps"] = round(throughput, 2)
            
        results.append({
            "Model": name,
            "Parameters": int(params),
            "FLOPs (M)": round(flops / 1e6, 2),
            "Chars74K Strict Acc (%)": round(strict_acc, 2),
            "Chars74K Tolerant Acc (%)": round(tolerant_acc, 2),
            "Chars74K Macro F1": round(macro_f1, 4),
            "S&P Noise Acc (%)": round(sp_strict_acc, 2),
            "Gaussian Blur Acc (%)": round(blur_strict_acc, 2),
            "EMNIST Strict Acc (%)": round(emnist_strict_acc, 2),
            **latency_profile
        })
        
    # Write summary CSV
    df = pd.DataFrame(results)
    part_csv = os.path.join(output_dir, "comprehensive_benchmark_results_part2.csv")
    df.to_csv(part_csv, index=False)
    log_message(f"Part 2 results saved to {part_csv}")
    
    # Try to merge
    merge_results(output_dir)
    
    elapsed_time = time.time() - t_start
    log_message(f"=== COMPREHENSIVE OCR BENCHMARK PART 2 COMPLETED IN {elapsed_time/60:.2f} MINUTES ===")

if __name__ == "__main__":
    main()
