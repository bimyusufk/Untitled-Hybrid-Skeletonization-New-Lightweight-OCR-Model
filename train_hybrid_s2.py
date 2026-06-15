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

from research.data_loader import augment_training_set, get_label_encoder_chars74k
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
SYNTHETIC_DATA_PATH = "datasets/synthetic_dataset_32x32.npz"
SEED = CONFIG["project"]["random_seed"]

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Standard Label Encoder for Chars74K
LABEL_ENCODER = get_label_encoder_chars74k()
NUM_CLASSES = len(LABEL_ENCODER.classes_)

# =====================================================================
# DATA LOADING FUNCTIONS
# =====================================================================
def load_skeleton_dataset(resolution):
    print(f"\nLoading original skeletonized Chars74K dataset at {resolution[0]}x{resolution[1]} resolution...")
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

    # Encode labels using standard encoder
    y_encoded = LABEL_ENCODER.transform(y)
    
    print(f"Loaded {X.shape[0]} samples. Number of classes: {NUM_CLASSES}")
    return X, y_encoded

def load_synthetic_dataset():
    print(f"\nLoading synthetic skeleton dataset from {SYNTHETIC_DATA_PATH}...")
    if not os.path.exists(SYNTHETIC_DATA_PATH):
        raise FileNotFoundError(f"Synthetic dataset not found at {SYNTHETIC_DATA_PATH}. Run generate_synthetic_data.py first.")
        
    data = np.load(SYNTHETIC_DATA_PATH)
    X = data['X'].astype(np.float32) / 255.0
    y = data['y']
    
    # Encode labels using standard encoder
    y_encoded = LABEL_ENCODER.transform(y.astype(str))
    
    print(f"Loaded {X.shape[0]} synthetic samples. Shape: {X.shape}")
    return X, y_encoded

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
    """S1 + S3 + SE (32x32 resolution)"""
    inputs = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', dilation_rate=2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.2)(x)
    
    # Block 2
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.2)(x)
    
    # Block 3
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x, ratio=8)
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
    """S1 + S3 + CBAM (32x32 resolution)"""
    inputs = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', dilation_rate=2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x, ratio=8)
    x = layers.Dropout(0.2)(x)
    
    # Block 2
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x, ratio=8)
    x = layers.Dropout(0.2)(x)
    
    # Block 3
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x, ratio=8)
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
# PIPELINE STAGES
# =====================================================================
def run_pretraining(X_synth, y_synth, build_model_fn, model_name, weights_path):
    print(f"\n--- [Fase 1: Pre-training] {model_name} ---")
    model = build_model_fn(NUM_CLASSES, input_shape=(32, 32, 1))
    model = compile_keras_model(model, CONFIG)
    
    EPOCHS = 5 # 5 epochs is sufficient for 100K samples
    BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "64"))
    
    print(f"Starting pre-training on {X_synth.shape[0]} synthetic samples...")
    t0 = time.time()
    model.fit(
        X_synth, y_synth,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )
    print(f"Pre-training of {model_name} completed in {(time.time() - t0)/60.0:.2f} minutes.")
    
    # Ensure directory exists and save weights
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    model.save_weights(weights_path)
    print(f"Saved pre-trained weights to {weights_path}")
    
    # Clean up model from memory
    del model
    tf.keras.backend.clear_session()

def run_finetuning(X_train_aug, y_train_aug, X_test, y_test, build_model_fn, model_name, weights_path, output_dir, model_key):
    print(f"\n--- [Fase 2: Fine-tuning] {model_name} ---")
    model = build_model_fn(NUM_CLASSES, input_shape=(32, 32, 1))
    model = compile_keras_model(model, CONFIG)
    
    # Load the pre-trained weights
    print(f"Loading pre-trained weights from {weights_path}...")
    model.load_weights(weights_path)
    
    # Compile and print summary
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
    
    print(f"Starting fine-tuning on Chars74K dataset...")
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
    print(f"Fine-tuning of {model_name} completed in {elapsed_min:.2f} minutes.")
    
    # Evaluation
    evaluation_results = save_ocr_evaluation_artifacts(
        history=history,
        X_test=X_test,
        y_test=y_test,
        label_encoder=LABEL_ENCODER,
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
# MAIN RUN
# =====================================================================
def main():
    print("=== STARTING EXPERIMENT S2 ===")
    
    # 1. Load Synthetic Data for pre-training (Skipped: weights already on disk)
    # X_synth, y_synth = load_synthetic_dataset()
    
    # Weights save paths
    weights_se = "research_outputs/models/pretrained_se.weights.h5"
    weights_cbam = "research_outputs/models/pretrained_cbam.weights.h5"
    
    # 2. Run Pre-training (Skipped: weights already on disk)
    # run_pretraining(X_synth, y_synth, build_s1_s3_se_model, "Hybrid Skeleton S1+S3+SE", weights_se)
    # run_pretraining(X_synth, y_synth, build_s1_s3_cbam_model, "Hybrid Skeleton S1+S3+CBAM", weights_cbam)
    
    # Clean up synthetic data from memory to make space for original dataset fine-tuning
    # del X_synth, y_synth
    # tf.keras.backend.clear_session()
    
    # 3. Load Original Chars74K Data
    X_real, y_real = load_skeleton_dataset((32, 32))
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=SEED, stratify=y_real
    )
    
    # Apply S1 offline augmentation (9x)
    X_train_aug, y_train_aug = augment_training_set(X_train, y_train, multiplier=9)
    
    # 4. Run Fine-tuning
    res_se = run_finetuning(
        X_train_aug, y_train_aug, X_test, y_test,
        build_model_fn=build_s1_s3_se_model,
        model_name="Hybrid Skeleton S1+S3+SE+S2",
        weights_path=weights_se,
        output_dir="ocr_evaluation_outputs_s1s3se_s2",
        model_key="hybrid_s1_s3_se_s2"
    )
    
    res_cbam = run_finetuning(
        X_train_aug, y_train_aug, X_test, y_test,
        build_model_fn=build_s1_s3_cbam_model,
        model_name="Hybrid Skeleton S1+S3+CBAM+S2",
        weights_path=weights_cbam,
        output_dir="ocr_evaluation_outputs_s1s3cbam_s2",
        model_key="hybrid_s1_s3_cbam_s2"
    )
    
    # 5. Output Report
    print("\n=========================================================")
    print("=== S2 EXPERIMENT RESULTS COMPARISON ===")
    print("=========================================================")
    for r in [res_se, res_cbam]:
        print(f"\nModel: {r['model_name']}")
        print(f"- Parameters: {r['parameters']:,}")
        print(f"- Strict Accuracy: {r['strict_accuracy']:.2f}%")
        print(f"- Tolerant Accuracy: {r['tolerant_accuracy']:.2f}%")
        print(f"- Inference Latency: {r['avg_inference_time_ms']:.4f} ms")
        print(f"- Epochs Trained (Fine-tuning): {r['epochs_trained']}")
        print(f"- Fine-tuning Duration: {r['elapsed_min']:.2f} min")
    
    summary_path = "s2_comparison_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== S2 EXPERIMENT COMPARISON REPORT ===\n\n")
        for r in [res_se, res_cbam]:
            f.write(f"Model: {r['model_name']}\n")
            f.write(f"- Parameters: {r['parameters']:,}\n")
            f.write(f"- Chars74K Strict Accuracy: {r['strict_accuracy']:.2f}%\n")
            f.write(f"- Chars74K Tolerant Accuracy: {r['tolerant_accuracy']:.2f}%\n")
            f.write(f"- Average Latency: {r['avg_inference_time_ms']:.4f} ms\n")
            f.write(f"- Epochs Trained (Fine-tuning): {r['epochs_trained']}\n")
            f.write(f"- Fine-tuning Duration: {r['elapsed_min']:.2f} min\n\n")
            
    print(f"\nComparison report saved to {summary_path}")

if __name__ == "__main__":
    main()
