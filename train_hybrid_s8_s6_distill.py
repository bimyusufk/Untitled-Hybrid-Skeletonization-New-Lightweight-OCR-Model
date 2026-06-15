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
if not CONFIG["hardware"].get("use_gpu", True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("[GPU] Disabled by config — using CPU")
else:
    print("[GPU] Enabled by config")

import tensorflow as tf
from tensorflow.keras import layers, models

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

from research.data_loader import augment_training_set, get_label_encoder_chars74k
from research.models import compile_keras_model
from ocr_evaluation import save_ocr_evaluation_artifacts

CSV_PATH = "datasets/annotations.csv"
SKELETON_BASE_DIR = "datasets/skeletonize"
SEED = CONFIG["project"]["random_seed"]
WEIGHTS_PATH = "research_outputs/models/pretrained_se_64x64.weights.h5"
IMAGE_SIZE = (64, 64)

tf.random.set_seed(SEED)
np.random.seed(SEED)

LABEL_ENCODER = get_label_encoder_chars74k()
NUM_CLASSES = len(LABEL_ENCODER.classes_)

# =====================================================================
# DATA LOADER
# =====================================================================
def load_skeleton_dataset_64x64():
    print(f"Loading skeletonized Chars74K dataset at {IMAGE_SIZE}...")
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
                    X_data.append(img.astype(np.float32) / 255.0)
                    y_labels.append(str(label))

    X = np.expand_dims(np.array(X_data), axis=-1)
    y = np.array(y_labels)
    y_encoded = LABEL_ENCODER.transform(y)
    
    print(f"Loaded {X.shape[0]} samples.")
    return X, y_encoded

# =====================================================================
# DISTILLER CLASS
# =====================================================================
class Distiller(models.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.5, temperature=3.0):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        x_student, x_teacher = x
        
        teacher_predictions = self.teacher(x_teacher, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x_student, training=True)

            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=-1),
                tf.nn.softmax(student_predictions / self.temperature, axis=-1)
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss * (self.temperature ** 2)

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_predictions)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss, "student_loss": student_loss, "distillation_loss": distillation_loss})
        return results

    def test_step(self, data):
        x, y = data
        x_student, _ = x
        
        y_prediction = self.student(x_student, training=False)
        student_loss = self.student_loss_fn(y, y_prediction)
        self.compiled_metrics.update_state(y, y_prediction)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": student_loss, "student_loss": student_loss})
        return results

# =====================================================================
# ATTENTION SE BLOCK & MODEL BUILDERS
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

def build_teacher_model_64x64(num_classes, input_shape=(64, 64, 1)):
    """A heavy pretrained model representing the Teacher using MobileNetV2 (with ImageNet weights)"""
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda t: (t * 2.0) - 1.0)(inputs)
    x = layers.Concatenate(axis=-1)([x, x, x])
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        alpha=1.0,
        include_top=False,
        weights='imagenet',
        pooling="avg"
    )
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
    print("=== STARTING EXPERIMENTS FOR S8 SKELETON OCR (64x64) ===")
    
    # Load dataset
    X, y = load_skeleton_dataset_64x64()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Apply S1 offline augmentation
    print("Augmenting Chars74K training dataset...")
    X_train_aug, y_train_aug = augment_training_set(X_train, y_train, multiplier=9)
    
    EPOCHS = int(os.getenv("OCR_EPOCHS", "30"))
    BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "64"))
    
    # -----------------------------------------------------------------
    # EXPERIMENT 1: S1 + S8 + S3 + SE + S2 (Baseline 64x64 Pretrained)
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### EXPERIMENT 1: S1 + S8 + S3 + SE + S2 (Baseline 64x64)     ###")
    print("#################################################################")
    
    model_exp1 = build_student_model_64x64(NUM_CLASSES)
    model_exp1 = compile_keras_model(model_exp1, CONFIG)
    
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading pre-trained synthetic weights from {WEIGHTS_PATH}...")
        model_exp1.load_weights(WEIGHTS_PATH)
    else:
        print("[WARNING] Pre-trained weights not found. Running from random initialization.")
        
    callbacks_exp1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    t0 = time.time()
    history_exp1 = model_exp1.fit(
        X_train_aug, y_train_aug,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks_exp1,
        verbose=2
    )
    duration_exp1 = (time.time() - t0) / 60.0
    print(f"Experiment 1 training completed in {duration_exp1:.2f} minutes.")
    
    # Save Model
    model_exp1_path = "research_outputs/models/baseline_student_64x64.keras"
    model_exp1.save(model_exp1_path)
    
    # Evaluate
    print("Evaluating Experiment 1...")
    eval_exp1 = save_ocr_evaluation_artifacts(
        history=history_exp1,
        X_test=X_test,
        y_test=y_test,
        label_encoder=LABEL_ENCODER,
        model=model_exp1,
        output_dir="ocr_evaluation_outputs_s1s8s3se_s2",
        model_key="hybrid_s1_s8_s3_se_s2",
        model_name="Hybrid Skeleton S1+S8+S3+SE+S2",
        batch_size=BATCH_SIZE,
    )
    metrics_exp1 = eval_exp1["metrics"]
    
    # -----------------------------------------------------------------
    # EXPERIMENT 2: S1 + S8 + S3 + SE + S2 + S6_v3 (Distillation)
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### EXPERIMENT 2: S1 + S8 + S3 + SE + S2 + S6_v3 (Distillation)###")
    print("#################################################################")
    
    # 2a. Train Teacher Model on 64x64 skeleton dataset
    print("\n--- Training Teacher Model (MobileNetV2 ImageNet) ---")
    teacher = build_teacher_model_64x64(NUM_CLASSES)
    teacher_trainable = models.Sequential([
        teacher,
        layers.Activation('softmax')
    ])
    teacher_trainable = compile_keras_model(teacher_trainable, CONFIG)
    
    callbacks_teacher = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    t_teacher = time.time()
    teacher_trainable.fit(
        X_train_aug, y_train_aug,
        epochs=int(os.getenv("OCR_EPOCHS_TEACHER", "30")),
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks_teacher,
        verbose=2
    )
    print(f"Teacher training completed in {(time.time() - t_teacher)/60.0:.2f} minutes.")
    
    # Save Teacher Model
    teacher_path = "research_outputs/models/teacher_model_64x64.keras"
    teacher.save(teacher_path)
    
    # 2b. Knowledge Distillation to Student
    print("\n--- Running Knowledge Distillation ---")
    student = build_student_model_64x64(NUM_CLASSES)
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading pre-trained synthetic weights from {WEIGHTS_PATH}...")
        student.load_weights(WEIGHTS_PATH)
        
    print("Temporarily setting student output layer activation to linear (logits)...")
    student.layers[-1].activation = None
    
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.5,
        temperature=3.0
    )
    
    callbacks_student = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    t_distill = time.time()
    history_distill = distiller.fit(
        (X_train_aug, X_train_aug), y_train_aug,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=((X_test, X_test), y_test),
        callbacks=callbacks_student,
        verbose=2
    )
    duration_distill = (time.time() - t_distill) / 60.0
    print(f"Knowledge Distillation completed in {duration_distill:.2f} minutes.")
    
    print("Restoring student output layer activation to softmax...")
    student.layers[-1].activation = tf.keras.activations.softmax
    
    # Save Distilled Student
    model_distilled_path = "research_outputs/models/distilled_student_64x64.keras"
    student.save(model_distilled_path)
    
    # Evaluate
    print("Evaluating Experiment 2...")
    eval_exp2 = save_ocr_evaluation_artifacts(
        history=history_distill,
        X_test=X_test,
        y_test=y_test,
        label_encoder=LABEL_ENCODER,
        model=student,
        output_dir="ocr_evaluation_outputs_s1s8s3se_s2_s6",
        model_key="hybrid_s1_s8_s3_se_s2_s6",
        model_name="Hybrid Skeleton S1+S8+S3+SE+S2+S6 (Distilled)",
        batch_size=BATCH_SIZE,
    )
    metrics_exp2 = eval_exp2["metrics"]
    
    # -----------------------------------------------------------------
    # COMPARISON REPORT
    # -----------------------------------------------------------------
    summary_path = "s8_s6_comparison_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== S8 + S6 RESOLUTION 64x64 EXPERIMENTS REPORT ===\n\n")
        f.write("Model A: Hybrid Skeleton S1+S8+S3+SE+S2 (Baseline 64x64)\n")
        f.write(f"- Parameters: {model_exp1.count_params():,}\n")
        f.write(f"- Chars74K Strict Accuracy: {metrics_exp1['strict_accuracy']:.2f}%\n")
        f.write(f"- Chars74K Tolerant Accuracy: {metrics_exp1['tolerant_accuracy']:.2f}%\n")
        f.write(f"- Average Latency: {metrics_exp1['avg_inference_time_ms']:.4f} ms/image\n")
        f.write(f"- Training Duration: {duration_exp1:.2f} min\n\n")
        
        f.write("Model B: Hybrid Skeleton S1+S8+S3+SE+S2+S6_v3 (Distilled 64x64)\n")
        f.write(f"- Parameters: {student.count_params():,}\n")
        f.write(f"- Chars74K Strict Accuracy: {metrics_exp2['strict_accuracy']:.2f}%\n")
        f.write(f"- Chars74K Tolerant Accuracy: {metrics_exp2['tolerant_accuracy']:.2f}%\n")
        f.write(f"- Average Latency: {metrics_exp2['avg_inference_time_ms']:.4f} ms/image\n")
        f.write(f"- Training Duration (Distillation): {duration_distill:.2f} min\n\n")
        
    print(f"\nDistillation comparison report saved to {summary_path}")

if __name__ == "__main__":
    main()
