import os
import time
import yaml
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Force CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import layers, models
from research.data_loader import get_label_encoder_chars74k
from research.models import compile_keras_model

# =====================================================================
# CONFIGURATION
# =====================================================================
print("Loading config.yaml...")
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

CSV_PATH = "datasets/annotations.csv"
SKELETON_BASE_DIR = "datasets/skeletonize"
WEIGHTS_PATH_SE = "research_outputs/models/pretrained_se.weights.h5"
MODEL_PATH_DISTILLED = "research_outputs/models/distilled_student.keras"
SEED = CONFIG["project"]["random_seed"]

tf.random.set_seed(SEED)
np.random.seed(SEED)

LABEL_ENCODER = get_label_encoder_chars74k()
NUM_CLASSES = len(LABEL_ENCODER.classes_)

# =====================================================================
# DATA LOADER
# =====================================================================
def load_test_dataset(resolution=(32, 32)):
    print(f"Loading skeletonized Chars74K dataset at {resolution}...")
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
    y_encoded = LABEL_ENCODER.transform(y)
    
    # 80/20 Split
    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
    )
    return X_test, y_test

# =====================================================================
# ATTENTION SE BLOCK & MODEL BUILDER
# =====================================================================
def se_block(input_tensor, ratio=8):
    channels = input_tensor.shape[-1]
    bottleneck_channels = max(1, channels // ratio)
    
    squeeze = layers.GlobalAveragePooling2D()(input_tensor)
    squeeze = layers.Reshape((1, 1, channels))(squeeze)
    
    excitation = layers.Dense(bottleneck_channels, activation='relu')(squeeze)
    excitation = layers.Dense(channels, activation='sigmoid')(excitation)
    
    scaled = layers.Multiply()([input_tensor, excitation])
    return scaled

def build_se_model(num_classes, input_shape=(32, 32, 1)):
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

def benchmark_latency(model, X_test, name):
    num_samples = X_test.shape[0]
    print(f"\n--- Latency Benchmark for {name} ---")
    
    # Warm-up
    print("Performing warm-up runs...")
    for _ in range(10):
        _ = model(X_test[0:1], training=False)
        
    # 1. Batch Inference (Batch Size = 64)
    print(f"[Metode A: Batch Inference - Batch Size = 64]")
    t0 = time.perf_counter()
    _ = model.predict(X_test, batch_size=64, verbose=0)
    dt_batch = time.perf_counter() - t0
    avg_batch_ms = (dt_batch / num_samples) * 1000
    print(f"- Total time for {num_samples} samples: {dt_batch:.4f} seconds")
    print(f"- Average latency per image: {avg_batch_ms:.4f} ms")
    
    # 2. Direct Function Call (Batch Size = 1)
    print(f"[Metode C: Direct Model Call model(x, training=False) - Batch Size = 1]")
    t2 = time.perf_counter()
    for i in range(min(500, num_samples)):
        _ = model(X_test[i:i+1], training=False)
    dt_direct = time.perf_counter() - t2
    avg_direct_ms = (dt_direct / min(500, num_samples)) * 1000
    print(f"- Total time for {min(500, num_samples)} samples: {dt_direct:.4f} seconds")
    print(f"- Average latency per image: {avg_direct_ms:.4f} ms")

def main():
    X_test, y_test = load_test_dataset()
    
    # Model A: Baseline
    print("\nSetting up Baseline Model (S2 Pretrained)...")
    model_baseline = build_se_model(NUM_CLASSES)
    model_baseline = compile_keras_model(model_baseline, CONFIG)
    model_baseline.load_weights(WEIGHTS_PATH_SE)
    benchmark_latency(model_baseline, X_test, "S1+S3+SE+S2 (Baseline)")
    
    # Model B: Distilled
    print("\nSetting up Distilled Model (S6)...")
    if os.path.exists(MODEL_PATH_DISTILLED):
        model_distilled = build_se_model(NUM_CLASSES)
        model_distilled = compile_keras_model(model_distilled, CONFIG)
        model_distilled.load_weights(MODEL_PATH_DISTILLED)
        benchmark_latency(model_distilled, X_test, "S1+S3+SE+S2+S6 (Distilled)")
    else:
        print(f"Error: Distilled model not found at {MODEL_PATH_DISTILLED}")

if __name__ == "__main__":
    main()

