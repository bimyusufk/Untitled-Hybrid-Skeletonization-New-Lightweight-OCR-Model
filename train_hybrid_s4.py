import os
import time
import yaml
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Force CPU training
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import layers, models

from research.data_loader import augment_training_set
from research.models import compile_keras_model
from ocr_evaluation import save_ocr_evaluation_artifacts

# =====================================================================
# 1. LOAD CONFIGURATION
# =====================================================================
print("Loading config.yaml...")
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

CSV_PATH = "datasets/annotations.csv"
SKELETON_BASE_DIR = "datasets/skeletonize"
SEED = CONFIG["project"]["random_seed"]

tf.random.set_seed(SEED)
np.random.seed(SEED)

# =====================================================================
# DATA LOADING FUNCTION
# =====================================================================
def load_skeleton_dataset(resolution):
    print(f"\nLoading skeletonized Chars74K dataset at {resolution[0]}x{resolution[1]} resolution...")
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
                    if img.shape[:2] != resolution:
                        img = cv2.resize(img, resolution, interpolation=cv2.INTER_AREA)
                    img_normalized = img.astype(np.float32) / 255.0
                    X_data.append(img_normalized)
                    y_labels.append(str(label))

    X = np.expand_dims(np.array(X_data), axis=-1)
    y = np.array(y_labels)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    print(f"Loaded {X.shape[0]} samples. Number of classes: {num_classes}")
    return X, y_encoded, label_encoder

# =====================================================================
# ATTENTION MODULES
# =====================================================================
def se_block(input_tensor, ratio=8):
    """Squeeze-and-Excitation Block"""
    channels = input_tensor.shape[-1]
    bottleneck_channels = max(1, channels // ratio)
    
    # Squeeze
    squeeze = layers.GlobalAveragePooling2D()(input_tensor)
    squeeze = layers.Reshape((1, 1, channels))(squeeze)
    
    # Excitation
    excitation = layers.Dense(bottleneck_channels, activation='relu')(squeeze)
    excitation = layers.Dense(channels, activation='sigmoid')(excitation)
    
    # Scale
    scaled = layers.Multiply()([input_tensor, excitation])
    return scaled

def cbam_block(input_tensor, ratio=8):
    """Convolutional Block Attention Module (CBAM)"""
    channels = input_tensor.shape[-1]
    bottleneck_channels = max(1, channels // ratio)
    
    # --- 1. Channel Attention Module ---
    # Global Avg and Max Pooling
    avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
    avg_pool = layers.Reshape((1, 1, channels))(avg_pool)
    
    max_pool = layers.GlobalMaxPooling2D()(input_tensor)
    max_pool = layers.Reshape((1, 1, channels))(max_pool)
    
    # Shared MLP
    shared_dense_1 = layers.Dense(bottleneck_channels, activation='relu')
    shared_dense_2 = layers.Dense(channels)
    
    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))
    
    channel_attention = layers.Add()([avg_out, max_out])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    scale_channel = layers.Multiply()([input_tensor, channel_attention])
    
    # --- 2. Spatial Attention Module ---
    # Global Avg and Max Pooling across channels
    avg_spatial = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(scale_channel)
    max_spatial = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(scale_channel)
    concat_spatial = layers.Concatenate(axis=-1)([avg_spatial, max_spatial])
    
    spatial_attention = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat_spatial)
    scale_spatial = layers.Multiply()([scale_channel, spatial_attention])
    
    return scale_spatial

# =====================================================================
# MODEL BUILDERS
# =====================================================================
def build_s1_s3_se_model(num_classes, input_shape=(32, 32, 1)):
    """S1 + S3 + SE (32x32 resolution, 3 blocks with Dilated + Strided Conv + SE)"""
    inputs = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', dilation_rate=2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x, ratio=8)  # SE Block
    x = layers.Dropout(0.2)(x)
    
    # Block 2
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x, ratio=8)  # SE Block
    x = layers.Dropout(0.2)(x)
    
    # Block 3
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x, ratio=8)  # SE Block
    x = layers.Dropout(0.3)(x)
    
    # Classifier
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_s1_s3_cbam_model(num_classes, input_shape=(32, 32, 1)):
    """S1 + S3 + CBAM (32x32 resolution, 3 blocks with Dilated + Strided Conv + CBAM)"""
    inputs = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', dilation_rate=2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x, ratio=8)  # CBAM Block
    x = layers.Dropout(0.2)(x)
    
    # Block 2
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x, ratio=8)  # CBAM Block
    x = layers.Dropout(0.2)(x)
    
    # Block 3
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x, ratio=8)  # CBAM Block
    x = layers.Dropout(0.3)(x)
    
    # Classifier
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# =====================================================================
# TRAINING FUNCTION FOR A CONFIGURATION
# =====================================================================
def train_and_evaluate_model(resolution, build_model_fn, output_dir, model_key, model_name, multiplier=9):
    # Load dataset
    X, y, label_encoder = load_skeleton_dataset(resolution)
    num_classes = len(label_encoder.classes_)
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Apply S1 offline augmentation
    X_train_aug, y_train_aug = augment_training_set(X_train, y_train, multiplier=multiplier)
    
    # Build & compile model
    model = build_model_fn(num_classes, input_shape=(resolution[0], resolution[1], 1))
    model = compile_keras_model(model, CONFIG)
    model.summary()
    
    # Early stopping callback
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    EPOCHS = int(os.getenv("OCR_EPOCHS", "30"))
    BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "64"))
    
    print(f"\nStarting training for {model_name}...")
    t0 = time.time()
    history = model.fit(
        X_train_aug, y_train_aug,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=2
    )
    elapsed_min = (time.time() - t0) / 60.0
    print(f"Training of {model_name} completed in {elapsed_min:.2f} minutes.")
    
    # Evaluation
    evaluation_results = save_ocr_evaluation_artifacts(
        history=history,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
        model=model,
        output_dir=output_dir,
        model_key=model_key,
        model_name=model_name,
        batch_size=BATCH_SIZE,
    )
    
    metrics = evaluation_results["metrics"]
    return {
        "model_name": model_name,
        "parameters": model.count_params(),
        "strict_accuracy": metrics["strict_accuracy"],
        "tolerant_accuracy": metrics["tolerant_accuracy"],
        "avg_inference_time_ms": metrics["avg_inference_time_ms"],
        "epochs_trained": len(history.history["loss"]),
        "elapsed_min": elapsed_min
    }

# =====================================================================
# EXECUTION & COMPARISON
# =====================================================================
def main():
    print("=== STARTING EXPERIMENT S4 ===")
    
    # 1. Train Model 1: S1 + S3 + SE (32x32)
    res_se = train_and_evaluate_model(
        resolution=(32, 32),
        build_model_fn=build_s1_s3_se_model,
        output_dir="ocr_evaluation_outputs_s1s3se",
        model_key="hybrid_s1_s3_se",
        model_name="Hybrid Skeleton S1+S3+SE"
    )
    
    # 2. Train Model 2: S1 + S3 + CBAM (32x32)
    res_cbam = train_and_evaluate_model(
        resolution=(32, 32),
        build_model_fn=build_s1_s3_cbam_model,
        output_dir="ocr_evaluation_outputs_s1s3cbam",
        model_key="hybrid_s1_s3_cbam",
        model_name="Hybrid Skeleton S1+S3+CBAM"
    )
    
    # 3. Log Comparison
    print("\n=========================================================")
    print("=== S4 EXPERIMENT RESULTS COMPARISON ===")
    print("=========================================================")
    for r in [res_se, res_cbam]:
        print(f"\nModel: {r['model_name']}")
        print(f"- Parameters: {r['parameters']:,}")
        print(f"- Strict Accuracy: {r['strict_accuracy']:.2f}%")
        print(f"- Tolerant Accuracy: {r['tolerant_accuracy']:.2f}%")
        print(f"- Inference Latency: {r['avg_inference_time_ms']:.4f} ms")
        print(f"- Epochs Trained: {r['epochs_trained']}")
        print(f"- Elapsed Time: {r['elapsed_min']:.2f} min")
    
    summary_path = "s4_comparison_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== S4 EXPERIMENT COMPARISON REPORT ===\n\n")
        for r in [res_se, res_cbam]:
            f.write(f"Model: {r['model_name']}\n")
            f.write(f"- Parameters: {r['parameters']:,}\n")
            f.write(f"- Chars74K Strict Accuracy: {r['strict_accuracy']:.2f}%\n")
            f.write(f"- Chars74K Tolerant Accuracy: {r['tolerant_accuracy']:.2f}%\n")
            f.write(f"- Average Latency: {r['avg_inference_time_ms']:.4f} ms\n")
            f.write(f"- Epochs Trained: {r['epochs_trained']}\n")
            f.write(f"- Training Duration: {r['elapsed_min']:.2f} min\n\n")
            
    print(f"\nComparison report saved to {summary_path}")

if __name__ == "__main__":
    main()
