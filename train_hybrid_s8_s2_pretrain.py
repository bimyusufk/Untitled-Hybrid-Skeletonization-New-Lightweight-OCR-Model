import os
import time
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. LOAD CONFIGURATION
# =====================================================================
print("Loading config.yaml...")
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# GPU setup MUST happen before any TF import
import ctypes
import sys
if sys.platform.startswith("linux"):
    try:
        for path in sys.path:
            possible_path = os.path.join(path, "nvidia", "cusolver", "lib", "libcusolver.so.11")
            if os.path.exists(possible_path):
                ctypes.CDLL(possible_path)
                print(f"[GPU] Preloaded libcusolver: {possible_path}")
                break
    except Exception as e:
        print(f"[GPU] Warning: Failed to preload libcusolver: {e}")

if not CONFIG["hardware"].get("use_gpu", True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("[GPU] Disabled by config — using CPU")
else:
    print("[GPU] Enabled by config")

import tensorflow as tf
from tensorflow.keras import layers, models
from research.data_loader import get_label_encoder_chars74k
from research.models import compile_keras_model

if CONFIG["hardware"].get("use_gpu", True):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if CONFIG["hardware"].get("gpu_memory_growth", True):
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        if CONFIG["hardware"].get("mixed_precision", False):
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print(f"[GPU] {len(gpus)} GPU(s) detected: {[g.name for g in gpus]}")
    else:
        print("[GPU] No GPU found — falling back to CPU")

SEED = CONFIG["project"]["random_seed"]
tf.random.set_seed(SEED)
np.random.seed(SEED)

LABEL_ENCODER = get_label_encoder_chars74k()
NUM_CLASSES = len(LABEL_ENCODER.classes_)
INPUT_NPZ = "datasets/synthetic_dataset_64x64.npz"
WEIGHTS_PATH = "research_outputs/models/pretrained_se_64x64.weights.h5"

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

def build_student_model_64x64(num_classes, input_shape=(64, 64, 1)):
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
    
    # Block 4 (Added for 64x64)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
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

def main():
    print("=== STARTING S8 SYNTHETIC PRE-TRAINING (64x64) ===")
    
    if not os.path.exists(INPUT_NPZ):
        raise FileNotFoundError(f"Synthetic dataset not found at {INPUT_NPZ}. Please generate it first.")
        
    print(f"Loading synthetic dataset from {INPUT_NPZ}...")
    data = np.load(INPUT_NPZ)
    X = data['X'].astype(np.float32) / 255.0
    y = data['y']
    
    y_encoded = LABEL_ENCODER.transform(y)
    
    print(f"Loaded {X.shape[0]} samples.")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.1, random_state=SEED, stratify=y_encoded
    )
    
    # Build student
    student = build_student_model_64x64(NUM_CLASSES)
    student = compile_keras_model(student, CONFIG)
    student.summary()
    
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
    
    print(f"Pre-training student model on {X_train.shape[0]} synthetic samples...")
    t0 = time.time()
    student.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=2
    )
    print(f"Pre-training completed in {(time.time() - t0)/60.0:.2f} minutes.")
    
    # Save weights
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    student.save_weights(WEIGHTS_PATH)
    print(f"Saved pre-trained weights to {WEIGHTS_PATH}")

if __name__ == "__main__":
    main()
