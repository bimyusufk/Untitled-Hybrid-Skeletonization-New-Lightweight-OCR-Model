"""
=======================================================================
SUPER HYBRID BENCHMARKING: MENGALAHKAN MODEL BERAT
=======================================================================

Model dan Skenario yang dibenchmark (4 Skenario):
1. TopoGrad-Net_Binary   - TopoGradNet (~1.16M) pada input Clean Binary (dengan Online Augmentation + 12 Fitur).
2. TopoGrad-Net          - TopoGradNet (~1.16M) pada input Morphological Gradient (dengan Online Augmentation + 12 Fitur).
3. Gradient_CNN_Hybrid  - ShallowCNNHybrid (~556k) pada input Morphological Gradient (Baseline Hybrid 5 Fitur - Tanpa Augmentasi).
4. Proposed_1M_Raw      - Proposed1MModel (~1.07M) pada input Clean Binary (Baseline Dilated CNN).

12 Fitur Topologi & Geometris:
- 5 Fitur Wilayah: Euler Number, Eccentricity, Aspect Ratio, Extent, Solidity.
- 7 Hu Moments: Invarian terhadap Translasi, Rotasi, dan Skala (Skala Logaritma).
=======================================================================
"""

import os
import time
import random
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.measure import label, regionprops
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# =====================================================================
# FASE 1: STANDARDISASI LINGKUNGAN & PREPROCESSING
# =====================================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

CLASS_LIST = (
    [str(i) for i in range(10)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
)
NUM_CLASSES = 62
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CLASS_LIST)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CLASS_LIST)}

IMAGE_SIZE = (64, 64)

def preprocess_image(raw_path):
    """Binarisasi Otsu + Conditional Hole Filling (lubang <= 35 piksel)"""
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
        if hole_area <= 35:
            small_holes_mask = np.logical_or(small_holes_mask, (labeled_holes == slice_index))
            
    img_clean_bin = np.logical_or(img_bool, small_holes_mask).astype(np.uint8) * 255
    return img_clean_bin, None

def extract_super_features(img_bin):
    """
    Ekstraksi 12 Fitur Geometris/Topologi:
    - 5 Properti Wilayah (Euler, Eccentricity, Aspect Ratio, Extent, Solidity)
    - 7 Hu Moments (invariant terhadap translasi, rotasi, skala - Skala Log)
    """
    binary = img_bin > 0
    labeled = label(binary)
    props = regionprops(labeled)
    
    # 5 region props
    if len(props) == 0:
        region_feats = np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    else:
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
        
    # 7 Hu Moments
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
    return np.concatenate([region_feats, hu_feats])

def load_binary_dataset(csv_path="datasets/annotations.csv", raw_base_dir="datasets/raw"):
    """Memuat dan menyimpan representasi biner awal."""
    print(f"Loading raw dataset from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotation file not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    X_bin_list = []
    y_list = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing raw to clean binary"):
        folder_name = row['Folder Name']
        label_char = str(row['Label'])
        
        raw_folder = os.path.join(raw_base_dir, folder_name)
        if not os.path.exists(raw_folder):
            continue
            
        for img_name in sorted(os.listdir(raw_folder)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                raw_path = os.path.join(raw_folder, img_name)
                img_bin, _ = preprocess_image(raw_path)
                if img_bin is not None:
                    X_bin_list.append(img_bin)
                    y_list.append(CHAR_TO_IDX[label_char])
                    
    X_bin = np.array(X_bin_list, dtype=np.uint8) # [N, 64, 64]
    y = np.array(y_list)
    print(f"Dataset biner dimuat: {X_bin.shape[0]} sampel.")
    return X_bin, y

class SuperDataset(Dataset):
    def __init__(self, X_bin, y, is_training=False, use_gradient=False, is_super_hybrid=True):
        self.X_bin = X_bin
        self.y = y
        self.is_training = is_training
        self.use_gradient = use_gradient
        self.is_super_hybrid = is_super_hybrid
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        img_bin = self.X_bin[idx].copy()
        
        # Online augmentation (Rotasi & Translasi acak)
        if self.is_training and self.is_super_hybrid:
            img_tensor = torch.tensor(img_bin, dtype=torch.float32).unsqueeze(0) # [1, 64, 64]
            
            # Rotasi acak: -10 ke +10 derajat
            angle = random.uniform(-10.0, 10.0)
            img_tensor = TF.rotate(img_tensor, angle)
            
            # Translasi acak: maks 10%
            max_dx = int(0.1 * 64)
            max_dy = int(0.1 * 64)
            dx = random.randint(-max_dx, max_dx)
            dy = random.randint(-max_dy, max_dy)
            img_tensor = TF.affine(img_tensor, angle=0, translate=[dx, dy], scale=1.0, shear=0)
            
            img_bin = (img_tensor.squeeze(0).numpy() > 127).astype(np.uint8) * 255
            
        # Ekstraksi fitur topologi berdasarkan input akhir
        if self.is_super_hybrid:
            # 12 Fitur (5 Properti + 7 Hu)
            feats = extract_super_features(img_bin)
        else:
            # 5 Fitur Properti Wilayah Dasar (Euler, Eccentricity, Aspect Ratio, Extent, Solidity)
            binary = img_bin > 0
            labeled = label(binary)
            props = regionprops(labeled)
            if len(props) == 0:
                feats = np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            else:
                props = sorted(props, key=lambda x: x.area, reverse=True)
                main_prop = props[0]
                h = max(1, main_prop.bbox[2] - main_prop.bbox[0])
                w = max(1, main_prop.bbox[3] - main_prop.bbox[1])
                feats = np.array([
                    float(max(-5.0, min(5.0, main_prop.euler_number))),
                    float(main_prop.eccentricity),
                    float(w / h),
                    float(main_prop.extent),
                    float(main_prop.solidity)
                ], dtype=np.float32)
                
        # Morphological Gradient
        if self.use_gradient:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img_out = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel)
        else:
            img_out = img_bin
            
        # Normalisasi ke [-1.0, 1.0]
        img_out_norm = (img_out.astype(np.float32) / 255.0 - 0.5) / 0.5
        img_out_tensor = torch.tensor(img_out_norm, dtype=torch.float32).unsqueeze(0)
        
        feats_tensor = torch.tensor(feats, dtype=torch.float32)
        label_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        
        return img_out_tensor, feats_tensor, label_tensor

# =====================================================================
# FASE 2: DEFINISI ARSITEKTUR MODEL
# =====================================================================

class TopoGradNet(nn.Module):
    """
    TopoGradNet (~1.16M parameter).
    CNN Backbone [32, 64, 128] filter dengan 12 fitur topologi/Hu di classifier head.
    """
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
        
        # Lapisan FC awal untuk visual
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        
        # Classifier Fusion Head
        self.fc2 = nn.Linear(128 + feat_dim, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        
        self.fc_out = nn.Linear(128, num_classes)
        
    def forward(self, x_img, x_feats):
        x = self.pool1(self.relu(self.bn1(self.conv1(x_img))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.drop1(x)
        
        # Gabungkan visual + geometris
        combined = torch.cat([x, x_feats], dim=1)
        
        x_fused = self.relu(self.bn5(self.fc2(combined)))
        x_fused = self.drop2(x_fused)
        
        out = self.fc_out(x_fused)
        return out

class ShallowCNNHybrid(nn.Module):
    """Model baseline hybrid kita sebelumnya (~556k parameter)"""
    def __init__(self, num_classes=62, in_channels=1, feat_dim=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(128 + feat_dim, num_classes)
        
    def forward(self, x_img, x_feats):
        x = self.pool1(self.relu(self.bn1(self.conv1(x_img))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.drop(x)
        
        combined = torch.cat([x, x_feats], dim=1)
        out = self.fc2(combined)
        return out

class SEBlock(nn.Module):
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
    """Model baseline berat kita sebelumnya (~1.07M parameter)"""
    def __init__(self, num_classes=62, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.se1 = SEBlock(24)
        self.drop1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(24, 48, kernel_size=3, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(48)
        self.se2 = SEBlock(48)
        self.drop2 = nn.Dropout(0.2)

        self.conv5 = nn.Conv2d(48, 96, kernel_size=3, padding=2, dilation=2)
        self.bn5 = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(96)
        self.se3 = SEBlock(96)
        self.drop3 = nn.Dropout(0.3)

        self.conv7 = nn.Conv2d(96, 192, kernel_size=3, padding=2, dilation=2)
        self.bn7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(192)
        self.se4 = SEBlock(192)
        self.drop4 = nn.Dropout(0.3)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_model_forward(model, model_name, device, is_hybrid=True, feat_dim=12):
    model = model.to(device)
    model.eval()
    dummy_img = torch.randn(2, 1, 64, 64).to(device)
    dummy_feats = torch.randn(2, feat_dim).to(device)
    with torch.no_grad():
        if is_hybrid:
            out = model(dummy_img, dummy_feats)
        else:
            out = model(dummy_img)
    assert out.shape == (2, NUM_CLASSES), f"{model_name} output shape mismatch"
    print(f"  [OK] {model_name} forward pass OK")

# =====================================================================
# FASE 3: TRAINING & EVALUATION FUNCTIONS
# =====================================================================

def train_model(model, train_loader, val_loader, epochs, device, model_name, patience=10, is_hybrid=True):
    params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"  Training: {model_name} ({params:,} parameters, Hybrid={is_hybrid})")
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
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, feats, labels in train_loader:
            images, feats, labels = images.to(device), feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            if is_hybrid:
                outputs = model(images, feats)
            else:
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
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, feats, labels in val_loader:
                images, feats, labels = images.to(device), feats.to(device), labels.to(device)
                if is_hybrid:
                    outputs = model(images, feats)
                else:
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

def evaluate_model(model, test_loader, device, model_name, is_hybrid=True, output_dir="ocr_evaluation_outputs_super_hybrid"):
    print(f"\n--- Evaluating: {model_name} ---")
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    model.eval()
    
    # Check features dimension from dataset
    dummy_img = torch.randn(1, 1, 64, 64).to(device)
    # Get feat_dim from dummy batch
    _, dummy_feats_batch, _ = next(iter(test_loader))
    feat_dim = dummy_feats_batch.shape[1]
    dummy_feats = torch.randn(1, feat_dim).to(device)
    
    with torch.no_grad():
        for _ in range(10):
            if is_hybrid:
                _ = model(dummy_img, dummy_feats)
            else:
                _ = model(dummy_img)
                
    all_preds = []
    all_targets = []
    total_time = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, feats, labels in test_loader:
            images = images.to(device)
            feats = feats.to(device)
            
            t_start = time.perf_counter()
            if is_hybrid:
                outputs = model(images, feats)
            else:
                outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            total_time += (t_end - t_start)
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            total_samples += labels.size(0)
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    y_true_chars = [IDX_TO_CHAR[idx] for idx in y_true]
    y_pred_chars = [IDX_TO_CHAR[idx] for idx in y_pred]
    
    strict_correct = 0
    case_error_but_char_correct = 0
    
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
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name}.png"), dpi=180, bbox_inches="tight")
    plt.close()
    
    # Prediction Samples
    sample_count = min(12, total_samples)
    cols = 4
    rows = int(np.ceil(sample_count / cols))
    plt.figure(figsize=(12, rows * 3))
    
    for idx in range(sample_count):
        ax = plt.subplot(rows, cols, idx + 1)
        test_img = test_loader.dataset.X_bin[idx].squeeze()
        test_img_orig = np.clip(test_img, 0, 255).astype(np.uint8)
        
        color = "green" if y_true_chars[idx] == y_pred_chars[idx] else "red"
        ax.imshow(test_img_orig, cmap="gray")
        ax.set_title(f"T:{y_true_chars[idx]} | P:{y_pred_chars[idx]}", fontsize=9, color=color, fontweight="bold")
        ax.axis("off")
        
    plt.suptitle(f"Sample Predictions: {model_name}", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"prediction_samples_{model_name}.png"), dpi=180, bbox_inches="tight")
    plt.close()
    
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
    print("=" * 75)
    print("  SUPER HYBRID BENCHMARKING: MENGALAHKAN MODEL BERAT")
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
        print("Creating mock dataset for dry-run verification...")
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
    
    # --- Dataloaders ---
    batch_size = 64
    
    # 1. Super Dataset Loaders (Hu + Online Augment)
    train_loader_super_bin = DataLoader(SuperDataset(X_train, y_train, is_training=True, use_gradient=False, is_super_hybrid=True),
                                        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader_super_bin = DataLoader(SuperDataset(X_val, y_val, is_training=False, use_gradient=False, is_super_hybrid=True),
                                      batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader_super_bin = DataLoader(SuperDataset(X_test, y_test, is_training=False, use_gradient=False, is_super_hybrid=True),
                                       batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
                                       
    train_loader_super_grad = DataLoader(SuperDataset(X_train, y_train, is_training=True, use_gradient=True, is_super_hybrid=True),
                                         batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader_super_grad = DataLoader(SuperDataset(X_val, y_val, is_training=False, use_gradient=True, is_super_hybrid=True),
                                       batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader_super_grad = DataLoader(SuperDataset(X_test, y_test, is_training=False, use_gradient=True, is_super_hybrid=True),
                                        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
                                        
    # 2. Standard Hybrid Loaders (5 region features, no augment)
    train_loader_std_grad = DataLoader(SuperDataset(X_train, y_train, is_training=False, use_gradient=True, is_super_hybrid=False),
                                       batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader_std_grad = DataLoader(SuperDataset(X_val, y_val, is_training=False, use_gradient=True, is_super_hybrid=False),
                                     batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader_std_grad = DataLoader(SuperDataset(X_test, y_test, is_training=False, use_gradient=True, is_super_hybrid=False),
                                      batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
                                      
    # 3. Baseline Dilated CNN Loaders (Clean Binary, no features, no augment)
    train_loader_raw = DataLoader(SuperDataset(X_train, y_train, is_training=False, use_gradient=False, is_super_hybrid=False),
                                  batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader_raw = DataLoader(SuperDataset(X_val, y_val, is_training=False, use_gradient=False, is_super_hybrid=False),
                                batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader_raw = DataLoader(SuperDataset(X_test, y_test, is_training=False, use_gradient=False, is_super_hybrid=False),
                                 batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
                                 
    configs = [
        ("TopoGrad-Net_Binary", lambda: TopoGradNet(NUM_CLASSES, feat_dim=12), "Clean Binary", 
         train_loader_super_bin, val_loader_super_bin, test_loader_super_bin, True, 12),
         
        ("TopoGrad-Net", lambda: TopoGradNet(NUM_CLASSES, feat_dim=12), "Morphological Gradient", 
         train_loader_super_grad, val_loader_super_grad, test_loader_super_grad, True, 12),
         
        ("Gradient_CNN_Hybrid_Baseline", lambda: ShallowCNNHybrid(NUM_CLASSES, feat_dim=5), "Morphological Gradient", 
         train_loader_std_grad, val_loader_std_grad, test_loader_std_grad, True, 5),
         
        ("Proposed_1M_Raw_Baseline", lambda: Proposed1MModel(NUM_CLASSES), "Clean Binary", 
         train_loader_raw, val_loader_raw, test_loader_raw, False, 0)
    ]
    
    epochs = int(os.getenv("OCR_EPOCHS", "50"))
    patience = 10
    output_dir = "ocr_evaluation_outputs_super_hybrid"
    
    if os.getenv("DRY_RUN", "False").lower() == "true":
        print("\n[DRY RUN] Restricting to 2 epochs with mini subsets.")
        epochs = 2
        patience = 2
        
        train_loader_super_bin = DataLoader(SuperDataset(X_train[:128], y_train[:128], is_training=True, use_gradient=False, is_super_hybrid=True), batch_size=32, shuffle=True)
        val_loader_super_bin = DataLoader(SuperDataset(X_val[:64], y_val[:64], is_training=False, use_gradient=False, is_super_hybrid=True), batch_size=32, shuffle=False)
        test_loader_super_bin = DataLoader(SuperDataset(X_test[:64], y_test[:64], is_training=False, use_gradient=False, is_super_hybrid=True), batch_size=32, shuffle=False)
        
        train_loader_super_grad = DataLoader(SuperDataset(X_train[:128], y_train[:128], is_training=True, use_gradient=True, is_super_hybrid=True), batch_size=32, shuffle=True)
        val_loader_super_grad = DataLoader(SuperDataset(X_val[:64], y_val[:64], is_training=False, use_gradient=True, is_super_hybrid=True), batch_size=32, shuffle=False)
        test_loader_super_grad = DataLoader(SuperDataset(X_test[:64], y_test[:64], is_training=False, use_gradient=True, is_super_hybrid=True), batch_size=32, shuffle=False)
        
        train_loader_std_grad = DataLoader(SuperDataset(X_train[:128], y_train[:128], is_training=False, use_gradient=True, is_super_hybrid=False), batch_size=32, shuffle=True)
        val_loader_std_grad = DataLoader(SuperDataset(X_val[:64], y_val[:64], is_training=False, use_gradient=True, is_super_hybrid=False), batch_size=32, shuffle=False)
        test_loader_std_grad = DataLoader(SuperDataset(X_test[:64], y_test[:64], is_training=False, use_gradient=True, is_super_hybrid=False), batch_size=32, shuffle=False)
        
        train_loader_raw = DataLoader(SuperDataset(X_train[:128], y_train[:128], is_training=False, use_gradient=False, is_super_hybrid=False), batch_size=32, shuffle=True)
        val_loader_raw = DataLoader(SuperDataset(X_val[:64], y_val[:64], is_training=False, use_gradient=False, is_super_hybrid=False), batch_size=32, shuffle=False)
        test_loader_raw = DataLoader(SuperDataset(X_test[:64], y_test[:64], is_training=False, use_gradient=False, is_super_hybrid=False), batch_size=32, shuffle=False)
        
        configs = [
            ("TopoGrad-Net_Binary", lambda: TopoGradNet(NUM_CLASSES, feat_dim=12), "Clean Binary", 
             train_loader_super_bin, val_loader_super_bin, test_loader_super_bin, True, 12),
             
            ("TopoGrad-Net", lambda: TopoGradNet(NUM_CLASSES, feat_dim=12), "Morphological Gradient", 
             train_loader_super_grad, val_loader_super_grad, test_loader_super_grad, True, 12),
             
            ("Gradient_CNN_Hybrid_Baseline", lambda: ShallowCNNHybrid(NUM_CLASSES, feat_dim=5), "Morphological Gradient", 
             train_loader_std_grad, val_loader_std_grad, test_loader_std_grad, True, 5),
             
            ("Proposed_1M_Raw_Baseline", lambda: Proposed1MModel(NUM_CLASSES), "Clean Binary", 
             train_loader_raw, val_loader_raw, test_loader_raw, False, 0)
        ]
        
    results = {}
    
    for config_name, model_fn, input_type, train_ldr, val_ldr, test_ldr, is_hybrid, feat_dim in configs:
        model = model_fn()
        params = count_parameters(model)
        
        print(f"\n[Verifikasi] Model: {config_name}")
        verify_model_forward(model, config_name, device, is_hybrid, feat_dim)
        print(f"  Trainable params: {params:,}")
        
        history = train_model(
            model=model,
            train_loader=train_ldr,
            val_loader=val_ldr,
            epochs=epochs,
            device=device,
            model_name=config_name,
            patience=patience,
            is_hybrid=is_hybrid
        )
        
        # Plot curves
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
        plt.savefig(os.path.join(output_dir, f"training_curves_{config_name}.png"), dpi=160, bbox_inches="tight")
        plt.close()
        
        # Evaluate
        eval_metrics = evaluate_model(
            model=model,
            test_loader=test_ldr,
            device=device,
            model_name=config_name,
            is_hybrid=is_hybrid,
            output_dir=output_dir
        )
        eval_metrics["input_type"] = input_type
        eval_metrics["is_hybrid"] = f"YES ({feat_dim} feats)" if is_hybrid else "NO"
        
        results[config_name] = eval_metrics
        
    # =====================================================================
    # TABEL KOMPARASI FINAL
    # =====================================================================
    print(f"\n\n{'='*105}")
    print("                       HASIL BENCHMARK SUPER HYBRID")
    print(f"{'='*105}")
    
    summary_data = []
    for m_name, res in results.items():
        summary_data.append({
            "Model": m_name,
            "Input Type": res["input_type"],
            "Hybrid": res["is_hybrid"],
            "Parameters": f"{res['params']:,}",
            "Strict Acc (%)": f"{res['strict_accuracy']:.2f}",
            "Tolerant Acc (%)": f"{res['tolerant_accuracy']:.2f}",
            "Latency (ms)": f"{res['avg_latency_ms']:.4f}"
        })
        
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(os.path.join(output_dir, "super_hybrid_summary.csv"), index=False)
    
    # Save Markdown Report
    report_md_path = os.path.join(output_dir, "super_hybrid_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Laporan Benchmarking Super Hybrid CNN\n")
        f.write("## Strategi Mengalahkan Model Berat (ResNet-18 & Proposed_1M) dengan Model Ringan Ultra Cepat\n\n")
        f.write("### Protokol Eksperimen\n")
        f.write("- **Dataset**: Chars74K (64x64, Grayscale, Preprocessed on-the-fly)\n")
        f.write(f"- **Split**: Train {X_train.shape[0]} | Val {X_val.shape[0]} | Test {X_test.shape[0]} (seed={SEED})\n")
        f.write(f"- **Epochs**: {epochs} (early stopping patience={patience})\n")
        f.write("- **Online Augmentation**: Random Rotation (+/-10 deg), Random Translation (+/-10%)\n")
        f.write("- **Fitur Geometris/Topologi**: 12 Fitur (5 Properti + 7 Hu Moments)\n\n")
        f.write("### Hasil Perbandingan Komparatif\n\n")
        f.write(summary_df.to_markdown(index=False) + "\n\n")
        f.write("### Kesimpulan & Temuan Utama\n")
        f.write("1. **Akurasi**: Apakah penskalaan lebar saluran (`[32, 64, 128]`) ditambah 12 fitur geometris (termasuk Hu moments) dan augmentasi online berhasil mengalahkan model dilated Proposed_1M dan SOTA ResNet-18?\n")
        f.write("2. **Latensi**: Memverifikasi keunggulan latensi model penskalaan non-dilasi yang diproyeksikan berada di kisaran ~0.02 ms (sub-milidetik).\n")
        
    # Save JSON results
    json_results = {}
    for m_name, res in results.items():
        json_results[m_name] = {
            "input_type": res["input_type"],
            "hybrid": res["is_hybrid"],
            "params": res["params"],
            "strict_accuracy": round(res["strict_accuracy"], 4),
            "tolerant_accuracy": round(res["tolerant_accuracy"], 4),
            "avg_latency_ms": round(res["avg_latency_ms"], 6)
        }
        
    with open(os.path.join(output_dir, "super_hybrid_results.json"), "w") as f:
        import json
        json.dump(json_results, f, indent=2)
        
    print(f"\n[OK] Reports saved to: {output_dir}/")

if __name__ == "__main__":
    main()
