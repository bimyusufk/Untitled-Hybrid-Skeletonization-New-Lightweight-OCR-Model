import os
import time
import yaml
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Force CPU training as requested to focus resources and run efficiently
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import layers, models

from research.data_loader import load_emnist, augment_training_set
from research.models import build_iso_parameter_cnn, compile_keras_model
from research.evaluation import evaluate_keras_model
from ocr_evaluation import save_ocr_evaluation_artifacts

# =====================================================================
# 1. LOAD CONFIGURATION
# =====================================================================
print("Loading config.yaml...")
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

CSV_PATH = "datasets/annotations.csv"
SKELETON_BASE_DIR = "datasets/skeletonize"
IMAGE_SIZE = tuple(CONFIG["datasets"]["chars74k"]["model_input_size"])
SEED = CONFIG["project"]["random_seed"]

tf.random.set_seed(SEED)
np.random.seed(SEED)

# =====================================================================
# 2. LOAD SKELETONIZED DATASET
# =====================================================================
print("Loading skeletonized Chars74K dataset...")
df = pd.read_csv(CSV_PATH)

X_data = []
y_labels = []

for index, row in df.iterrows():
    folder_name = row['Folder Name']
    label = row['Label']
    folder_path = os.path.join(SKELETON_BASE_DIR, folder_name)
    
    if not os.path.exists(folder_path):
        continue
        
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape[:2] != IMAGE_SIZE:
                    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                img_normalized = img.astype(np.float32) / 255.0
                X_data.append(img_normalized)
                y_labels.append(str(label))

X = np.expand_dims(np.array(X_data), axis=-1)
y = np.array(y_labels)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"Loaded {X.shape[0]} skeleton samples across {num_classes} classes.")

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)

# =====================================================================
# 3. APPLY S1 DATA AUGMENTATION (OFFLINE)
# =====================================================================
print("\n--- Applying S1 Data Augmentation (Elastic + Endpoint + Dilation) ---")
# Multiply training dataset by 9x
X_train_aug, y_train_aug = augment_training_set(X_train, y_train, multiplier=9)
print(f"Original Training Size: {X_train.shape[0]}")
print(f"Augmented Training Size: {X_train_aug.shape[0]}")

# =====================================================================
# 4. BUILD & COMPILE MODEL (Small 16-32-64 architecture, no internal spatial aug layers)
# =====================================================================
print("\nBuilding Hybrid Skeleton + S1 Aug model...")
model = build_iso_parameter_cnn(num_classes, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
model = compile_keras_model(model, CONFIG)
model.summary()

# =====================================================================
# 5. TRAINING
# =====================================================================
EPOCHS = int(os.getenv("OCR_EPOCHS", "30"))
BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "64")) # config batch size
OUTPUT_DIR = "ocr_evaluation_outputs_s1"
MODEL_KEY = "hybrid_skeleton_s1"
MODEL_NAME = "Hybrid Skeleton + S1 Aug"

print(f"\nStarting CPU training: epochs={EPOCHS}, batch_size={BATCH_SIZE}...")
t0 = time.time()
history = model.fit(
    X_train_aug, y_train_aug,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=2
)
elapsed_min = (time.time() - t0) / 60.0
print(f"Training completed in {elapsed_min:.2f} minutes.")

# =====================================================================
# 6. EVALUATION
# =====================================================================
print("\n=========================================================")
# Save standard artifacts using the existing evaluation helper
evaluation_results = save_ocr_evaluation_artifacts(
    history=history,
    X_test=X_test,
    y_test=y_test,
    label_encoder=label_encoder,
    model=model,
    output_dir=OUTPUT_DIR,
    model_key=MODEL_KEY,
    model_name=MODEL_NAME,
    batch_size=BATCH_SIZE,
)

metrics = evaluation_results["metrics"]

print(f"Hasil Pengujian S1 Pada {metrics['total_test']} Data Test:")
print(f"-------------------------------------------------")
print(f"1. Benar Mutlak (Strict Accuracy)             : {metrics['strict_correct']} data ({metrics['strict_accuracy']:.2f}%)")
print(f"2. Akurasi Toleran (Case-Insensitive)          : {metrics['tolerant_accuracy']:.2f}%")
print(f"3. Rerata Waktu per Gambar                     : {metrics['avg_inference_time_ms']:.2f} ms / gambar")
print(f"-------------------------------------------------")

# =====================================================================
# 7. CROSS-DATASET VALIDATION ON EMNIST
# =====================================================================
print("\nRunning EMNIST cross-dataset validation...")
try:
    X_emnist_raw, X_emnist_skel, y_emnist, le_emnist = load_emnist(CONFIG)
    emnist_metrics = evaluate_keras_model(
        model, X_emnist_skel, y_emnist, le_emnist, batch_size=BATCH_SIZE
    )
    print(f"EMNIST Cross-Dataset Strict Accuracy: {emnist_metrics['strict_accuracy']:.2f}%")
    print(f"EMNIST Cross-Dataset Tolerant Accuracy: {emnist_metrics['tolerant_accuracy']:.2f}%")
    print(f"EMNIST Cross-Dataset Macro F1: {emnist_metrics['macro_f1']:.4f}")
    
    # Save results to a comparison summary
    summary_path = os.path.join(OUTPUT_DIR, "s1_vs_baseline.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== EXPERIMENT S1 RESULTS SUMMARY ===\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Parameters: {model.count_params():,}\n")
        f.write(f"Chars74K Strict Accuracy: {metrics['strict_accuracy']:.2f}%\n")
        f.write(f"Chars74K Tolerant Accuracy: {metrics['tolerant_accuracy']:.2f}%\n")
        f.write(f"EMNIST Strict Accuracy: {emnist_metrics['strict_accuracy']:.2f}%\n")
        f.write(f"EMNIST Tolerant Accuracy: {emnist_metrics['tolerant_accuracy']:.2f}%\n")
        f.write(f"EMNIST Macro F1: {emnist_metrics['macro_f1']:.4f}\n")
    print(f"Comparison report saved to {summary_path}")
except Exception as e:
    print(f"Could not complete EMNIST evaluation: {e}")
