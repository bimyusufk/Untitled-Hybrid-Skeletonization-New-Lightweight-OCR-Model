"""
=======================================================================
BENCHMARKING TERKONTROL: KLASIFIKASI KARAKTER TERISOLASI UNTUK EDGE DEVICE
=======================================================================

Model yang dibenchmark (3x2 Matrix, Raw vs. Skeletonized):

1. ResNet-18        (~11.2M) - Standar industri medium-weight SOTA.
2. MobileNetV3-Large (~3.5M) - Standar industri edge/mobile SOTA.
3. Proposed_1M       (~1.07M) - Arsitektur usulan kita: CNN + SE Blocks + Dilated Conv.

Jalur Evaluasi Silang (Cross-Evaluation):
- Jalur A: Dilatih dan diuji pada Gambar Mentah (Raw)
- Jalur B: Dilatih dan diuji pada Gambar Skeletonized (Skeletonized)

Protokol: Semua model dilatih dari scratch dengan optimizer, scheduler,
          data split, dan resolusi input yang identik.
=======================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# =====================================================================
# FASE 1: STANDARDISASI DATASET & LINGKUNGAN
# =====================================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Class mapping: Alphanumeric 62 kelas
CLASS_LIST = (
    [str(i) for i in range(10)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
)
NUM_CLASSES = 62
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CLASS_LIST)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CLASS_LIST)}

# Resolusi masukan wajib
IMAGE_SIZE = (64, 64)

def load_paired_dataset(csv_path="datasets/annotations.csv", raw_base_dir="datasets/raw", skeleton_base_dir="datasets/skeletonize"):
    """
    Memuat dataset berpasangan (Raw dan Skeletonized) secara bersamaan untuk memastikan
    keselarasan indeks sampel 100% sempurna.
    """
    print(f"Loading paired dataset from {csv_path}...")
    print(f"  Raw base dir: {raw_base_dir}")
    print(f"  Skeleton base dir: {skeleton_base_dir}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotation file not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    X_raw_data = []
    X_skel_data = []
    y_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading paired images"):
        folder_name = row['Folder Name']
        label = str(row['Label'])
        
        raw_folder = os.path.join(raw_base_dir, folder_name)
        skel_folder = os.path.join(skeleton_base_dir, folder_name)
        
        if not os.path.exists(raw_folder) or not os.path.exists(skel_folder):
            continue
            
        raw_files = set(os.listdir(raw_folder))
        skel_files = set(os.listdir(skel_folder))
        common_files = sorted(list(raw_files.intersection(skel_files)))
        
        for img_name in common_files:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                raw_path = os.path.join(raw_folder, img_name)
                skel_path = os.path.join(skel_folder, img_name)
                
                img_raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
                img_skel = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
                
                if img_raw is not None and img_skel is not None:
                    if img_raw.shape[:2] != IMAGE_SIZE:
                        img_raw = cv2.resize(img_raw, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                    if img_skel.shape[:2] != IMAGE_SIZE:
                        img_skel = cv2.resize(img_skel, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                        
                    # Normalisasi: mean=[0.5], std=[0.5] -> memetakan [0, 255] ke [-1.0, 1.0]
                    img_raw_normalized = (img_raw.astype(np.float32) / 255.0 - 0.5) / 0.5
                    img_skel_normalized = (img_skel.astype(np.float32) / 255.0 - 0.5) / 0.5
                    
                    X_raw_data.append(img_raw_normalized)
                    X_skel_data.append(img_skel_normalized)
                    y_data.append(CHAR_TO_IDX[label])
                    
    X_raw = np.expand_dims(np.array(X_raw_data), axis=1)  # C=1 (Grayscale) -> shape [N, 1, 64, 64]
    X_skel = np.expand_dims(np.array(X_skel_data), axis=1) # C=1 (Grayscale) -> shape [N, 1, 64, 64]
    y = np.array(y_data)
    
    print(f"Paired dataset loaded: {X_raw.shape[0]} samples across {NUM_CLASSES} classes.")
    return X_raw, X_skel, y

class StandardDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================================================================
# FASE 2: DEFINISI ARSITEKTUR MODEL
# =====================================================================

# --- MODEL 1: MODEL USULAN (Proposed ~1M Model) ---
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (Hu et al., 2018)"""
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // ratio)),
            nn.ReLU(),
            nn.Linear(max(1, channels // ratio), channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale

class Proposed1MModel(nn.Module):
    """
    Model arsitektur usulan (~1.07M parameter).
    CNN dengan SE Blocks + Dilated Convolutions + Strided Downsampling.
    """
    def __init__(self, num_classes=62, in_channels=1):
        super().__init__()
        # Block 1: 64x64 -> 32x32
        self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.se1 = SEBlock(24)
        self.drop1 = nn.Dropout(0.2)

        # Block 2: 32x32 -> 16x16
        self.conv3 = nn.Conv2d(24, 48, kernel_size=3, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(48)
        self.se2 = SEBlock(48)
        self.drop2 = nn.Dropout(0.2)

        # Block 3: 16x16 -> 8x8
        self.conv5 = nn.Conv2d(48, 96, kernel_size=3, padding=2, dilation=2)
        self.bn5 = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(96)
        self.se3 = SEBlock(96)
        self.drop3 = nn.Dropout(0.3)

        # Block 4: 8x8 -> 4x4
        self.conv7 = nn.Conv2d(96, 192, kernel_size=3, padding=2, dilation=2)
        self.bn7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(192)
        self.se4 = SEBlock(192)
        self.drop4 = nn.Dropout(0.3)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Classifier Head
        self.fc1 = nn.Linear(192 * 4 * 4, 128)
        self.bn9 = nn.BatchNorm1d(128)
        self.drop5 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.se1(x); x = self.drop1(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.se2(x); x = self.drop2(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.se3(x); x = self.drop3(x)

        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.se4(x); x = self.drop4(x)

        x = self.flatten(x)
        x = self.relu(self.bn9(self.fc1(x)))
        x = self.drop5(x)
        x = self.fc2(x)
        return x


# --- MODEL 2: ResNet-18 (Standar Industri Medium-weight) ---
def build_resnet18(num_classes=62, in_channels=1):
    """
    ResNet-18 (~11.2M parameter).
    Modifikasi: conv pertama grayscale, FC head 62 kelas.
    """
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# --- MODEL 3: MobileNetV3-Large (Standar Industri Edge/Mobile) ---
def build_mobilenet_v3_large(num_classes=62, in_channels=1):
    """
    MobileNetV3-Large (~3.5M parameter).
    Modifikasi: conv pertama grayscale, classifier head 62 kelas.
    """
    model = models.mobilenet_v3_large(weights=None)
    # Ubah conv pertama: 3 channel RGB -> 1 channel Grayscale
    model.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
    # Ubah classifier head: 960 -> 1280 -> 62 kelas
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


# =====================================================================
# UTILITAS
# =====================================================================

def count_parameters(model):
    """Hitung total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_model_forward(model, model_name, device, in_channels=1):
    """Verifikasi bahwa forward pass berhasil dan dimensi output benar."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(2, in_channels, 64, 64).to(device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, NUM_CLASSES), \
        f"{model_name}: Expected output (2, {NUM_CLASSES}), got {out.shape}"
    print(f"  [OK] {model_name}: forward pass OK, output shape {out.shape}")
    return True


# =====================================================================
# FASE 3: PROTOKOL PELATIHAN TERKONTROL (CONTROLLED TRAINING)
# =====================================================================

def train_model(model, train_loader, val_loader, epochs, device, model_name, patience=10):
    """
    Melatih model dengan protokol standar:
      - Optimizer: AdamW (lr=1e-3, weight_decay=1e-2)
      - Scheduler: CosineAnnealingLR
      - Loss: CrossEntropyLoss
      - Early stopping berdasarkan validation loss
    """
    params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"  Training: {model_name} ({params:,} parameters)")
    print(f"{'='*60}")
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        scheduler.step()
        
        epoch_loss = running_loss / total_train
        epoch_acc = (correct_train / total_train) * 100
        
        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = val_running_loss / total_val
        val_acc = (correct_val / total_val) * 100
        
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"  Epoch [{epoch:3d}/{epochs}] - "
              f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
              
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [WARNING] Early stopping triggered at epoch {epoch}. Restoring best weights.")
                model.load_state_dict(best_model_state)
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return history


# =====================================================================
# FASE 4: EVALUASI & EXPORT METRIK
# =====================================================================

def evaluate_model(model, test_loader, device, model_name, output_dir="ocr_evaluation_outputs_benchmark"):
    """Evaluasi model: strict/tolerant accuracy, latensi, classification report, confusion matrix."""
    print(f"\n--- Evaluating: {model_name} ---")
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    model.eval()
    
    # Warm-up (10 runs)
    dummy_input = torch.randn(1, 1, 64, 64).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Inference + latency measurement
    all_preds = []
    all_targets = []
    total_time = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            t_start = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            total_time += (t_end - t_start)
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    # Konversi ke karakter untuk metrik tolerant
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
    avg_latency_ms = (total_time / total_samples) * 1000
    
    print(f"  Strict Accuracy:  {strict_accuracy:.2f}%")
    print(f"  Tolerant Accuracy: {tolerant_accuracy:.2f}%")
    print(f"  Avg Latency:      {avg_latency_ms:.4f} ms/image")
    
    # Classification Report
    report_text = classification_report(
        y_true, y_pred, 
        target_names=CLASS_LIST, 
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )
    report_path = os.path.join(output_dir, f"classification_report_{model_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Parameters: {count_parameters(model):,}\n")
        f.write(f"Strict Accuracy: {strict_accuracy:.2f}%\n")
        f.write(f"Tolerant Accuracy: {tolerant_accuracy:.2f}%\n")
        f.write(f"Average Latency: {avg_latency_ms:.4f} ms/image\n\n")
        f.write(report_text)
        
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(15, 15))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, CLASS_LIST, rotation=90, fontsize=6)
    plt.yticks(tick_marks, CLASS_LIST, fontsize=6)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name}.png"), 
                dpi=180, bbox_inches="tight")
    plt.close()
    
    # Prediction Samples
    sample_count = min(12, total_samples)
    cols = 4
    rows = int(np.ceil(sample_count / cols))
    plt.figure(figsize=(12, rows * 3))
    
    for idx in range(sample_count):
        ax = plt.subplot(rows, cols, idx + 1)
        test_img = test_loader.dataset.X[idx].squeeze().numpy()
        test_img_orig = (test_img * 0.5 + 0.5) * 255.0
        test_img_orig = np.clip(test_img_orig, 0, 255).astype(np.uint8)
        
        color = "green" if y_true_chars[idx] == y_pred_chars[idx] else "red"
        ax.imshow(test_img_orig, cmap="gray")
        ax.set_title(f"T:{y_true_chars[idx]} | P:{y_pred_chars[idx]}", 
                     fontsize=9, color=color, fontweight="bold")
        ax.axis("off")
        
    plt.suptitle(f"Sample Predictions: {model_name}", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"prediction_samples_{model_name}.png"), 
                dpi=180, bbox_inches="tight")
    plt.close()
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}.pth"))
    
    return {
        "strict_accuracy": strict_accuracy,
        "tolerant_accuracy": tolerant_accuracy,
        "avg_latency_ms": avg_latency_ms,
        "params": count_parameters(model)
    }


# =====================================================================
# MAIN RUNNER
# =====================================================================

def main():
    print("=" * 70)
    print("  BENCHMARKING TERKONTROL: KLASIFIKASI KARAKTER UNTUK EDGE DEVICE")
    print("  Model: ResNet-18 | MobileNetV3-Large | Proposed_1M")
    print("  Tipe Input: Raw vs. Skeletonized (3x2 Matrix)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # --- Load Paired Dataset ---
    try:
        X_raw, X_skel, y = load_paired_dataset()
    except Exception as e:
        print(f"Error loading paired dataset: {e}")
        print("Creating mock dataset for dry-run verification...")
        X_raw = np.random.randn(200, 1, 64, 64).astype(np.float32)
        X_skel = np.random.randn(200, 1, 64, 64).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=(200,))
    
    # --- Data Split berbasis Indeks ---
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=y
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=y[temp_idx]
    )
    
    # Split Raw Dataset
    X_train_raw, y_train_raw = X_raw[train_idx], y[train_idx]
    X_val_raw, y_val_raw = X_raw[val_idx], y[val_idx]
    X_test_raw, y_test_raw = X_raw[test_idx], y[test_idx]
    
    # Split Skeletonized Dataset
    X_train_skel, y_train_skel = X_skel[train_idx], y[train_idx]
    X_val_skel, y_val_skel = X_skel[val_idx], y[val_idx]
    X_test_skel, y_test_skel = X_skel[test_idx], y[test_idx]
    
    print(f"\nData Split (Identik Raw & Skeletonized):")
    print(f"  Train: {X_train_raw.shape[0]} | Val: {X_val_raw.shape[0]} | Test: {X_test_raw.shape[0]}")
    
    # --- Dataloaders ---
    batch_size = 64
    
    # Raw Loaders
    train_loader_raw = DataLoader(StandardDataset(X_train_raw, y_train_raw), batch_size=batch_size, shuffle=True, 
                                 num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader_raw = DataLoader(StandardDataset(X_val_raw, y_val_raw), batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader_raw = DataLoader(StandardDataset(X_test_raw, y_test_raw), batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=(device.type == "cuda"))

    # Skeletonized Loaders
    train_loader_skel = DataLoader(StandardDataset(X_train_skel, y_train_skel), batch_size=batch_size, shuffle=True, 
                                  num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader_skel = DataLoader(StandardDataset(X_val_skel, y_val_skel), batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader_skel = DataLoader(StandardDataset(X_test_skel, y_test_skel), batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=(device.type == "cuda"))
    
    # --- Konfigurasi 3x2 Matrix (Format: Name, Instantiation Lambda, Input Type, loaders) ---
    configs = [
        ("ResNet18_Raw", lambda: build_resnet18(NUM_CLASSES), "Raw", train_loader_raw, val_loader_raw, test_loader_raw),
        ("ResNet18_Skeleton", lambda: build_resnet18(NUM_CLASSES), "Skeleton", train_loader_skel, val_loader_skel, test_loader_skel),
        ("MobileNetV3_Raw", lambda: build_mobilenet_v3_large(NUM_CLASSES), "Raw", train_loader_raw, val_loader_raw, test_loader_raw),
        ("MobileNetV3_Skeleton", lambda: build_mobilenet_v3_large(NUM_CLASSES), "Skeleton", train_loader_skel, val_loader_skel, test_loader_skel),
        ("Proposed_1M_Raw", lambda: Proposed1MModel(num_classes=NUM_CLASSES), "Raw", train_loader_raw, val_loader_raw, test_loader_raw),
        ("Proposed_1M_Skeleton", lambda: Proposed1MModel(num_classes=NUM_CLASSES), "Skeleton", train_loader_skel, val_loader_skel, test_loader_skel),
    ]
    
    epochs = int(os.getenv("OCR_EPOCHS", "50"))
    patience = 10
    output_dir = "ocr_evaluation_outputs_benchmark"
    
    # DRY RUN mode
    if os.getenv("DRY_RUN", "False").lower() == "true":
        print("\n[DRY RUN] Restricting to 2 epochs with mini subsets.")
        epochs = 2
        patience = 2
        
        # Mini loaders Raw
        train_loader_raw = DataLoader(StandardDataset(X_train_raw[:128], y_train_raw[:128]), batch_size=32, shuffle=True)
        val_loader_raw = DataLoader(StandardDataset(X_val_raw[:64], y_val_raw[:64]), batch_size=32, shuffle=False)
        test_loader_raw = DataLoader(StandardDataset(X_test_raw[:64], y_test_raw[:64]), batch_size=32, shuffle=False)

        # Mini loaders Skeletonized
        train_loader_skel = DataLoader(StandardDataset(X_train_skel[:128], y_train_skel[:128]), batch_size=32, shuffle=True)
        val_loader_skel = DataLoader(StandardDataset(X_val_skel[:64], y_val_skel[:64]), batch_size=32, shuffle=False)
        test_loader_skel = DataLoader(StandardDataset(X_test_skel[:64], y_test_skel[:64]), batch_size=32, shuffle=False)
        
        # Override configs with mini loaders
        configs = [
            ("ResNet18_Raw", lambda: build_resnet18(NUM_CLASSES), "Raw", train_loader_raw, val_loader_raw, test_loader_raw),
            ("ResNet18_Skeleton", lambda: build_resnet18(NUM_CLASSES), "Skeleton", train_loader_skel, val_loader_skel, test_loader_skel),
            ("MobileNetV3_Raw", lambda: build_mobilenet_v3_large(NUM_CLASSES), "Raw", train_loader_raw, val_loader_raw, test_loader_raw),
            ("MobileNetV3_Skeleton", lambda: build_mobilenet_v3_large(NUM_CLASSES), "Skeleton", train_loader_skel, val_loader_skel, test_loader_skel),
            ("Proposed_1M_Raw", lambda: Proposed1MModel(num_classes=NUM_CLASSES), "Raw", train_loader_raw, val_loader_raw, test_loader_raw),
            ("Proposed_1M_Skeleton", lambda: Proposed1MModel(num_classes=NUM_CLASSES), "Skeleton", train_loader_skel, val_loader_skel, test_loader_skel),
        ]
    
    # --- Training & Evaluation Loop ---
    results = {}
    
    for config_name, model_fn, input_type, train_ldr, val_ldr, test_ldr in configs:
        model = model_fn()
        params = count_parameters(model)
        
        print(f"\n[Verifikasi] Model: {config_name}")
        verify_model_forward(model, config_name, device)
        print(f"  Trainable params: {params:,}")
        
        # Train
        history = train_model(
            model=model,
            train_loader=train_ldr,
            val_loader=val_ldr,
            epochs=epochs,
            device=device,
            model_name=config_name,
            patience=patience
        )
        
        # Plot training curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_title(f"Loss: {config_name}", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history["train_acc"], label="Train Acc", linewidth=2)
        axes[1].plot(history["val_acc"], label="Val Acc", linewidth=2)
        axes[1].set_title(f"Accuracy: {config_name}", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"training_curves_{config_name}.png"), 
                    dpi=160, bbox_inches="tight")
        plt.close()
        
        # Evaluate
        eval_metrics = evaluate_model(
            model=model,
            test_loader=test_ldr,
            device=device,
            model_name=config_name,
            output_dir=output_dir
        )
        eval_metrics["input_type"] = input_type
        
        results[config_name] = eval_metrics
    
    # =====================================================================
    # TABEL KOMPARASI FINAL
    # =====================================================================
    print(f"\n\n{'='*85}")
    print("                       HASIL BENCHMARK TERKONTROL (3x2 MATRIX)")
    print(f"{'='*85}")
    
    paper_refs = {
        "ResNet18_Raw": "He et al., CVPR 2016",
        "ResNet18_Skeleton": "He et al., CVPR 2016",
        "MobileNetV3_Raw": "Howard et al., ICCV 2019",
        "MobileNetV3_Skeleton": "Howard et al., ICCV 2019",
        "Proposed_1M_Raw": "Custom (SE + Dilated Conv)",
        "Proposed_1M_Skeleton": "Custom (SE + Dilated Conv)",
    }
    
    summary_data = []
    for m_name, res in results.items():
        summary_data.append({
            "Model": m_name.replace("_Raw", "").replace("_Skeleton", ""),
            "Input Type": res["input_type"],
            "Referensi": paper_refs.get(m_name, ""),
            "Parameters": f"{res['params']:,}",
            "Strict Acc (%)": f"{res['strict_accuracy']:.2f}",
            "Tolerant Acc (%)": f"{res['tolerant_accuracy']:.2f}",
            "Latency (ms)": f"{res['avg_latency_ms']:.4f}"
        })
        
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save CSV
    summary_df.to_csv(os.path.join(output_dir, "benchmark_summary.csv"), index=False)
    
    # Save Markdown Report
    report_md_path = os.path.join(output_dir, "benchmark_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Laporan Benchmarking Terkontrol (Matriks 3x2)\n")
        f.write("## Klasifikasi Karakter Terisolasi untuk Edge Device (Raw vs. Skeletonized)\n\n")
        f.write("### Protokol Eksperimen\n")
        f.write("- **Dataset**: Chars74K Paired (Raw & Skeletonized, 64x64, Grayscale)\n")
        f.write(f"- **Split**: Train {X_train_raw.shape[0]} | Val {X_val_raw.shape[0]} | Test {X_test_raw.shape[0]} (seed={SEED})\n")
        f.write(f"- **Epochs**: {epochs} (early stopping patience={patience})\n")
        f.write("- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-2)\n")
        f.write("- **Scheduler**: CosineAnnealingLR\n")
        f.write("- **Loss**: CrossEntropyLoss\n\n")
        f.write("### Hasil Perbandingan Komparatif\n\n")
        f.write(summary_df.to_markdown(index=False) + "\n\n")
        f.write("### Referensi Arsitektur\n\n")
        f.write("| Model | Referensi | Deskripsi Arsitektur |\n")
        f.write("|---|---|---|\n")
        f.write("| **ResNet-18** | He et al., CVPR 2016 | Standar industri mid-weight, residual connection |\n")
        f.write("| **MobileNetV3-Large** | Howard et al., ICCV 2019 | Standar industri edge-optimized dengan MBConv |\n")
        f.write("| **Proposed_1M** | Custom | Model usulan: dilated conv + SE attention blocks |\n")
    
    # Save JSON results
    json_results = {}
    for m_name, res in results.items():
        json_results[m_name] = {
            "model_base": m_name.replace("_Raw", "").replace("_Skeleton", ""),
            "input_type": res["input_type"],
            "params": res["params"],
            "strict_accuracy": round(res["strict_accuracy"], 4),
            "tolerant_accuracy": round(res["tolerant_accuracy"], 4),
            "avg_latency_ms": round(res["avg_latency_ms"], 6),
            "paper_ref": paper_refs.get(m_name, "")
        }
    
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        import json
        json.dump(json_results, f, indent=2)
    
    print(f"\n[OK] Reports saved to: {output_dir}/")
    print(f"  - benchmark_summary.csv")
    print(f"  - benchmark_report.md")
    print(f"  - benchmark_results.json")
    print(f"  - Per-model: classification reports, confusion matrices, training curves")


if __name__ == "__main__":
    main()
