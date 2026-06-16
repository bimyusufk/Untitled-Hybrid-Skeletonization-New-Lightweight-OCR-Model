import os
import time
import yaml
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

from tensorflow.keras import layers, models

from research.data_loader import augment_training_set, get_label_encoder_chars74k
from research.models import compile_keras_model
from ocr_evaluation import save_ocr_evaluation_artifacts

CSV_PATH = "datasets/annotations.csv"
SKELETON_BASE_DIR = "datasets/skeletonize"
RAW_BASE_DIR = "datasets/raw"
SEED = CONFIG["project"]["random_seed"]
WEIGHTS_PATH = "research_outputs/models/pretrained_se.weights.h5"

tf.random.set_seed(SEED)
np.random.seed(SEED)

LABEL_ENCODER = get_label_encoder_chars74k()
NUM_CLASSES = len(LABEL_ENCODER.classes_)

# =====================================================================
# DATA LOADER
# =====================================================================
def load_dataset_for_distillation(resolution=(32, 32)):
    """Load both raw and skeleton datasets to match teacher and student inputs"""
    print(f"Loading datasets at {resolution}...")
    df = pd.read_csv(CSV_PATH)
    
    X_raw_data = []
    X_skel_data = []
    y_labels = []

    for index, row in df.iterrows():
        folder_name = row['Folder Name']
        label = row['Label']
        
        skel_folder = os.path.join(SKELETON_BASE_DIR, folder_name)
        raw_folder = os.path.join(RAW_BASE_DIR, folder_name)
        
        if not os.path.exists(skel_folder) or not os.path.exists(raw_folder):
            continue
            
        # Match filenames
        skel_files = set(os.listdir(skel_folder))
        raw_files = set(os.listdir(raw_folder))
        common_files = skel_files.intersection(raw_files)
        
        for img_name in common_files:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                skel_path = os.path.join(skel_folder, img_name)
                raw_path = os.path.join(raw_folder, img_name)
                
                img_skel = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
                img_raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
                
                if img_skel is not None and img_raw is not None:
                    # Preprocess skeleton
                    if img_skel.shape[:2] != resolution:
                        img_skel = cv2.resize(img_skel, resolution, interpolation=cv2.INTER_AREA)
                    X_skel_data.append(img_skel.astype(np.float32) / 255.0)
                    
                    # Preprocess raw (resize raw for teacher model if input shape is 32x32)
                    if img_raw.shape[:2] != resolution:
                        img_raw = cv2.resize(img_raw, resolution, interpolation=cv2.INTER_AREA)
                    X_raw_data.append(img_raw.astype(np.float32) / 255.0)
                    
                    y_labels.append(str(label))

    X_skel = np.expand_dims(np.array(X_skel_data), axis=-1)
    X_raw = np.expand_dims(np.array(X_raw_data), axis=-1)
    y = np.array(y_labels)
    y_encoded = LABEL_ENCODER.transform(y)
    
    print(f"Loaded {X_skel.shape[0]} matching samples.")
    return X_skel, X_raw, y_encoded

# =====================================================================
# DISTILLER CLASS
# =====================================================================
class Distiller(models.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3.0):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        # x details: (x_student, x_teacher)
        x, y = data
        x_student, x_teacher = x
        
        # Forward pass of teacher
        teacher_predictions = self.teacher(x_teacher, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x_student, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=-1),
                tf.nn.softmax(student_predictions / self.temperature, axis=-1)
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss * (self.temperature ** 2)

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss, "student_loss": student_loss, "distillation_loss": distillation_loss})
        return results

    def test_step(self, data):
        # Unpack data
        x, y = data
        x_student, _ = x
        
        # Compute predictions
        y_prediction = self.student(x_student, training=False)

        # Calculate loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update metrices
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": student_loss, "student_loss": student_loss})
        return results

# =====================================================================
# SE STUDENT BLOCK & BUILDERS
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

def build_student_model(num_classes, input_shape=(32, 32, 1)):
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

def build_teacher_model(num_classes, input_shape=(32, 32, 1)):
    """A heavy pretrained model representing the Teacher using MobileNetV2 (with ImageNet weights)"""
    inputs = layers.Input(shape=input_shape)
    
    # Scale grayscale input [0, 1] to [-1, 1] for MobileNetV2 expected range
    x = layers.Lambda(lambda t: (t * 2.0) - 1.0)(inputs)
    
    # Tile grayscale channel to 3 channels
    x = layers.Concatenate(axis=-1)([x, x, x])
    
    # Load MobileNetV2 pretrained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        alpha=1.0,
        include_top=False,
        weights='imagenet',
        pooling="avg"
    )
    
    # Keep the base model trainable to allow fine-tuning on characters
    base_model.trainable = True
    
    x = base_model(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)  # linear activation (logits)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# =====================================================================
# MAIN RUN
# =====================================================================
def main():
    print("=== STARTING EXPERIMENT S6 (KNOWLEDGE DISTILLATION) ===")
    
    # 1. Load Matching Datasets
    X_skel, X_raw, y = load_dataset_for_distillation()
    
    # Split
    indices = np.arange(X_skel.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y)
    
    X_skel_train, X_skel_test = X_skel[train_idx], X_skel[test_idx]
    X_raw_train, X_raw_test = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Apply S1 offline augmentation to student's skeleton training dataset (9x multiplier)
    print("Augmenting Student Skeleton training dataset...")
    X_skel_train_aug, y_train_aug = augment_training_set(X_skel_train, y_train, multiplier=9)
    
    # To match student dataset size, we tile the raw teacher inputs 9 times as well (without skeleton augmentation)
    print("Duplicating Teacher Raw training dataset to match size...")
    X_raw_train_aug = np.concatenate([X_raw_train] * 9, axis=0)
    
    # 2. Build and Train Teacher Model
    print("\n--- [Fase 1: Training Teacher Model] ---")
    teacher = build_teacher_model(NUM_CLASSES)
    # Compile with softmax activation wrapper for training
    teacher_trainable = models.Sequential([
        teacher,
        layers.Activation('softmax')
    ])
    teacher_trainable = compile_keras_model(teacher_trainable, CONFIG)
    teacher_trainable.summary()
    
    teacher_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    EPOCHS_TEACHER = int(os.getenv("OCR_EPOCHS_TEACHER", "30")) # Can configure separate epochs if needed
    BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "64"))
    
    t0 = time.time()
    teacher_history = teacher_trainable.fit(
        X_skel_train_aug, y_train_aug,
        epochs=EPOCHS_TEACHER,
        batch_size=BATCH_SIZE,
        validation_data=(X_skel_test, y_test),
        callbacks=teacher_callbacks,
        verbose=2
    )
    print(f"Teacher training completed in {(time.time() - t0)/60.0:.2f} minutes.")
    
    # Save teacher weights
    teacher_path = "research_outputs/models/teacher_model.keras"
    os.makedirs(os.path.dirname(teacher_path), exist_ok=True)
    teacher.save(teacher_path)
    print(f"Saved Teacher model to {teacher_path}")
    
    # 3. Build Student Model and load S2 pre-trained weights
    print("\n--- [Fase 2: Knowledge Distillation to Student] ---")
    student = build_student_model(NUM_CLASSES)
    print(f"Loading S2 pre-trained weights from {WEIGHTS_PATH}...")
    student.load_weights(WEIGHTS_PATH)
    
    # Temporarily remove activation function from the last layer for logits-based distillation
    print("Temporarily setting student output layer activation to linear (logits)...")
    student.layers[-1].activation = None
    
    # 4. Initialize and compile Distiller
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.5,
        temperature=3.0
    )
    
    student_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Run Distillation
    t1 = time.time()
    distillation_history = distiller.fit(
        (X_skel_train_aug, X_skel_train_aug), y_train_aug,
        epochs=int(os.getenv("OCR_EPOCHS", "30")),
        batch_size=BATCH_SIZE,
        validation_data=((X_skel_test, X_skel_test), y_test),
        callbacks=student_callbacks,
        verbose=2
    )
    elapsed_min = (time.time() - t1) / 60.0
    print(f"Knowledge Distillation completed in {elapsed_min:.2f} minutes.")
    
    # Restore student output layer activation to softmax
    print("Restoring student output layer activation to softmax...")
    student.layers[-1].activation = tf.keras.activations.softmax
    
    # Save distilled student
    distilled_student_path = "research_outputs/models/distilled_student.keras"
    student.save(distilled_student_path)
    print(f"Saved Distilled Student model to {distilled_student_path}")
    
    # 5. Evaluate Distilled Student Model
    print("\n--- [Fase 3: Evaluating Distilled Student] ---")
    evaluation_results = save_ocr_evaluation_artifacts(
        history=distillation_history,
        X_test=X_skel_test,
        y_test=y_test,
        label_encoder=LABEL_ENCODER,
        model=student,
        output_dir="ocr_evaluation_outputs_s1s3se_s2_s6",
        model_key="hybrid_s1_s3_se_s2_s6",
        model_name="Hybrid Skeleton S1+S3+SE+S2+S6 (Distilled)",
        batch_size=BATCH_SIZE,
    )
    
    metrics = evaluation_results["metrics"]
    
    summary_path = "s6_comparison_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== S6 KNOWLEDGE DISTILLATION COMPARISON REPORT ===\n\n")
        f.write("Model: Hybrid Skeleton S1+S3+SE+S2+S6 (Distilled Student)\n")
        f.write(f"- Parameters: {student.count_params():,}\n")
        f.write(f"- Chars74K Strict Accuracy: {metrics['strict_accuracy']:.2f}%\n")
        f.write(f"- Chars74K Tolerant Accuracy: {metrics['tolerant_accuracy']:.2f}%\n")
        f.write(f"- Average Latency: {metrics['avg_inference_time_ms']:.4f} ms\n")
        f.write(f"- Epochs Trained (Distillation): {len(distillation_history.history['accuracy'])}\n")
        f.write(f"- Training Duration: {elapsed_min:.2f} min\n\n")
        
    print(f"\nDistillation comparison report saved to {summary_path}")

if __name__ == "__main__":
    main()
