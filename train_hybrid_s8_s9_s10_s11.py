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
# DATA LOADER WITH S9 METADATA EXTRACTION
# =====================================================================
def load_skeleton_dataset_with_metadata():
    print(f"Loading skeletonized Chars74K dataset at {IMAGE_SIZE} with metadata extraction (S9)...")
    df = pd.read_csv(CSV_PATH)
    X_data = []
    X_meta = []
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
                    # S9 Feature Extraction: Aspect Ratio & Pixel Density
                    h_orig, w_orig = img.shape[:2]
                    aspect_ratio = float(w_orig) / float(h_orig) if h_orig > 0 else 1.0
                    num_pixels = np.sum(img > 0)
                    density = float(num_pixels) / float(w_orig * h_orig) if (w_orig * h_orig) > 0 else 0.0
                    
                    if img.shape[:2] != IMAGE_SIZE:
                        img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                    else:
                        img_resized = img.copy()
                        
                    X_data.append(img_resized.astype(np.float32) / 255.0)
                    X_meta.append([aspect_ratio, density])
                    y_labels.append(str(label))

    X = np.expand_dims(np.array(X_data), axis=-1)
    X_m = np.array(X_meta, dtype=np.float32)
    y = np.array(y_labels)
    y_encoded = LABEL_ENCODER.transform(y)
    
    print(f"Loaded {X.shape[0]} samples with metadata.")
    return X, X_m, y_encoded

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
        # x details: (x_student, x_teacher) where x_student can be (image, meta)
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
# ATTENTION SE BLOCK
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

# =====================================================================
# MODEL BUILDERS
# =====================================================================
def build_backbone(inputs):
    """CNN Backbone (S8+S3+SE) shared across configurations"""
    # Block 1
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.2, name='drop_1')(x)
    
    # Block 2
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.2, name='drop_2')(x)
    
    # Block 3
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_5')(x)
    x = layers.BatchNormalization(name='bn_5')(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_6')(x)
    x = layers.BatchNormalization(name='bn_6')(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.3, name='drop_3')(x)
    
    # Block 4
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_7')(x)
    x = layers.BatchNormalization(name='bn_7')(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_8')(x)
    x = layers.BatchNormalization(name='bn_8')(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.3, name='drop_4')(x)
    
    x = layers.Flatten(name='flat')(x)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_9')(x)
    return x

# S2 Reference Model (to load weights topologically)
def build_student_model_64x64_baseline(num_classes, input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    x = build_backbone(inputs)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)

# S9 Model: Feature Fusion
def build_student_s9(num_classes, input_shape=(64, 64, 1)):
    input_img = layers.Input(shape=input_shape, name="input_img")
    input_meta = layers.Input(shape=(2,), name="input_meta")
    
    x = build_backbone(input_img)
    
    # Fusion (S9)
    x = layers.Concatenate(name="fusion")([x, input_meta])
    
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output_s9")(x)
    
    return models.Model(inputs=[input_img, input_meta], outputs=outputs)

# S10 Model: Hierarchical Multi-Task Output Head
class JointReconstruction(layers.Layer):
    def call(self, inputs):
        p_char, p_case = inputs
        # Char: 36 units (0-9, A-Z), Case: 3 units (digit, upper, lower)
        p_digits = p_char[:, 0:10] * p_case[:, 0:1]
        p_upper = p_char[:, 10:36] * p_case[:, 1:2]
        p_lower = p_char[:, 10:36] * p_case[:, 2:3]
        return layers.Concatenate(axis=-1)([p_digits, p_upper, p_lower])

def build_student_s10(num_classes, input_shape=(64, 64, 1)):
    input_img = layers.Input(shape=input_shape, name="input_img")
    x = build_backbone(input_img)
    x = layers.Dropout(0.4)(x)
    
    # Multi-task Heads (S10)
    p_char = layers.Dense(36, activation='softmax', name="head_char")(x)
    p_case = layers.Dense(3, activation='softmax', name="head_case")(x)
    
    # Probabilistic Join
    outputs = JointReconstruction(name="hierarchical_reconstruction")([p_char, p_case])
    
    return models.Model(inputs=input_img, outputs=outputs)

# S11 Model: Fusion + Hierarchical Head (S9 + S10 Combined)
def build_student_s11(num_classes, input_shape=(64, 64, 1)):
    input_img = layers.Input(shape=input_shape, name="input_img")
    input_meta = layers.Input(shape=(2,), name="input_meta")
    
    x = build_backbone(input_img)
    x = layers.Concatenate(name="fusion")([x, input_meta])
    x = layers.Dropout(0.4)(x)
    
    p_char = layers.Dense(36, activation='softmax', name="head_char")(x)
    p_case = layers.Dense(3, activation='softmax', name="head_case")(x)
    
    outputs = JointReconstruction(name="hierarchical_reconstruction")([p_char, p_case])
    
    return models.Model(inputs=[input_img, input_meta], outputs=outputs)

# Teacher model
def build_teacher_model_64x64(num_classes, input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda t: (t * 2.0) - 1.0)(inputs)
    x = layers.Concatenate(axis=-1)([x, x, x])
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        alpha=1.0, include_top=False, weights='imagenet', pooling="avg"
    )
    base_model.trainable = True
    x = base_model(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)
    return models.Model(inputs=inputs, outputs=outputs)

# Helper to transfer weights from 64x64 baseline pre-trained model
def load_backbone_weights(target_model):
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[WARNING] Weights path {WEIGHTS_PATH} not found. Skipping weight loading.")
        return
        
    print(f"Initializing student backbone weights from {WEIGHTS_PATH}...")
    baseline = build_student_model_64x64_baseline(NUM_CLASSES)
    baseline.load_weights(WEIGHTS_PATH)
    
    transferred = 0
    for layer in baseline.layers:
        try:
            target_layer = target_model.get_layer(layer.name)
            target_layer.set_weights(layer.get_weights())
            transferred += 1
        except Exception:
            continue
    print(f"Successfully transferred weights for {transferred} layers.")

# =====================================================================
# MAIN RUN
# =====================================================================
def main():
    print("=== STARTING S9, S10 & S11 EXPERIMENTS (64x64) ===")
    
    # Load dataset
    X, X_m, y = load_skeleton_dataset_with_metadata()
    
    # Split
    indices = np.arange(X.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y)
    
    X_train, X_test = X[train_idx], X[test_idx]
    X_m_train, X_m_test = X_m[train_idx], X_m[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Apply S1 offline augmentation
    print("Augmenting Chars74K training dataset (Student skeleton)...")
    X_train_aug, y_train_aug = augment_training_set(X_train, y_train, multiplier=9)
    # Augment metadata (it just duplicates the same metadata 9x corresponding to the augmented images)
    X_m_train_aug = np.concatenate([X_m_train] * 9, axis=0)
    
    EPOCHS = int(os.getenv("OCR_EPOCHS", "30"))
    BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "64"))
    
    # 2. Build and Fine-Tune SOTA Teacher model (MobileNetV2 ImageNet)
    print("\n--- [Fase 1: Training Teacher Model on 64x64 skeleton] ---")
    teacher = build_teacher_model_64x64(NUM_CLASSES)
    teacher_trainable = models.Sequential([
        teacher,
        layers.Activation('softmax')
    ])
    teacher_trainable = compile_keras_model(teacher_trainable, CONFIG)
    
    callbacks_teacher = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1
        )
    ]
    
    t_teach = time.time()
    teacher_trainable.fit(
        X_train_aug, y_train_aug,
        epochs=int(os.getenv("OCR_EPOCHS_TEACHER", "30")),
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks_teacher,
        verbose=2
    )
    print(f"Teacher training completed in {(time.time() - t_teach)/60.0:.2f} minutes.")
    
    # -----------------------------------------------------------------
    # EXPERIMENT S9: S1+S8+S3+SE+S2+S6 + S9 (Feature Fusion)
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### EXPERIMENT S9: Distilled 64x64 + S9 (Feature Fusion)      ###")
    print("#################################################################")
    
    student_s9 = build_student_s9(NUM_CLASSES)
    load_backbone_weights(student_s9)
    
    print("Temporarily setting student output layer activation to linear...")
    student_s9.get_layer("output_s9").activation = None
    
    distiller_s9 = Distiller(student=student_s9, teacher=teacher)
    distiller_s9.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.5, temperature=3.0
    )
    
    t0 = time.time()
    history_s9 = distiller_s9.fit(
        ((X_train_aug, X_m_train_aug), X_train_aug), y_train_aug,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(((X_test, X_m_test), X_test), y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)],
        verbose=2
    )
    duration_s9 = (time.time() - t0) / 60.0
    
    student_s9.get_layer("output_s9").activation = tf.keras.activations.softmax
    student_s9.save("research_outputs/models/distilled_student_s9_64x64.keras")
    
    eval_s9 = save_ocr_evaluation_artifacts(
        history=history_s9, X_test=[X_test, X_m_test], y_test=y_test,
        label_encoder=LABEL_ENCODER, model=student_s9,
        output_dir="ocr_evaluation_outputs_s1s8s3se_s2_s6_s9",
        model_key="hybrid_s1_s8_s3_se_s2_s6_s9",
        model_name="Hybrid Skeleton 64x64 + S6 + S9 (Feature Fusion)",
        batch_size=BATCH_SIZE
    )
    metrics_s9 = eval_s9["metrics"]
    
    # -----------------------------------------------------------------
    # EXPERIMENT S10: S1+S8+S3+SE+S2+S6 + S10 (Hierarchical Head)
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### EXPERIMENT S10: Distilled 64x64 + S10 (Hierarchical Head) ###")
    print("#################################################################")
    
    student_s10 = build_student_s10(NUM_CLASSES)
    load_backbone_weights(student_s10)
    
    # Distillation with hierarchical probabilities (it multiplies softmaxes directly, so it outputs probabilities)
    distiller_s10 = Distiller(student=student_s10, teacher=teacher)
    distiller_s10.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.5, temperature=3.0
    )
    
    t1 = time.time()
    history_s10 = distiller_s10.fit(
        (X_train_aug, X_train_aug), y_train_aug,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=((X_test, X_test), y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)],
        verbose=2
    )
    duration_s10 = (time.time() - t1) / 60.0
    student_s10.save("research_outputs/models/distilled_student_s10_64x64.keras")
    
    eval_s10 = save_ocr_evaluation_artifacts(
        history=history_s10, X_test=X_test, y_test=y_test,
        label_encoder=LABEL_ENCODER, model=student_s10,
        output_dir="ocr_evaluation_outputs_s1s8s3se_s2_s6_s10",
        model_key="hybrid_s1_s8_s3_se_s2_s6_s10",
        model_name="Hybrid Skeleton 64x64 + S6 + S10 (Hierarchical Head)",
        batch_size=BATCH_SIZE
    )
    metrics_s10 = eval_s10["metrics"]
    
    # -----------------------------------------------------------------
    # EXPERIMENT S11: S1+S8+S3+SE+S2+S6 + S11 (Fusion + Hierarchical)
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### EXPERIMENT S11: Distilled 64x64 + S11 (Fusion + Head)     ###")
    print("#################################################################")
    
    student_s11 = build_student_s11(NUM_CLASSES)
    load_backbone_weights(student_s11)
    
    distiller_s11 = Distiller(student=student_s11, teacher=teacher)
    distiller_s11.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.5, temperature=3.0
    )
    
    t2 = time.time()
    history_s11 = distiller_s11.fit(
        ((X_train_aug, X_m_train_aug), X_train_aug), y_train_aug,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(((X_test, X_m_test), X_test), y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)],
        verbose=2
    )
    duration_s11 = (time.time() - t2) / 60.0
    student_s11.save("research_outputs/models/distilled_student_s11_64x64.keras")
    
    eval_s11 = save_ocr_evaluation_artifacts(
        history=history_s11, X_test=[X_test, X_m_test], y_test=y_test,
        label_encoder=LABEL_ENCODER, model=student_s11,
        output_dir="ocr_evaluation_outputs_s1s8s3se_s2_s6_s11",
        model_key="hybrid_s1_s8_s3_se_s2_s6_s11",
        model_name="Hybrid Skeleton 64x64 + S6 + S11 (Fusion + Hierarchical)",
        batch_size=BATCH_SIZE
    )
    metrics_s11 = eval_s11["metrics"]
    
    # -----------------------------------------------------------------
    # COMPARISON REPORT
    # -----------------------------------------------------------------
    summary_path = "s9_s10_s11_comparison_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== S9, S10 & S11 ACCURACY OPTIMIZATION REPORT (64x64) ===\n\n")
        
        f.write("Model S9: Hybrid Skeleton + S6 + S9 (Feature Fusion)\n")
        f.write(f"- Parameters: {student_s9.count_params():,}\n")
        f.write(f"- Chars74K Strict Accuracy: {metrics_s9['strict_accuracy']:.2f}%\n")
        f.write(f"- Chars74K Tolerant Accuracy: {metrics_s9['tolerant_accuracy']:.2f}%\n")
        f.write(f"- Average Latency: {metrics_s9['avg_inference_time_ms']:.4f} ms/image\n")
        f.write(f"- Training Duration: {duration_s9:.2f} min\n\n")
        
        f.write("Model S10: Hybrid Skeleton + S6 + S10 (Hierarchical Head)\n")
        f.write(f"- Parameters: {student_s10.count_params():,}\n")
        f.write(f"- Chars74K Strict Accuracy: {metrics_s10['strict_accuracy']:.2f}%\n")
        f.write(f"- Chars74K Tolerant Accuracy: {metrics_s10['tolerant_accuracy']:.2f}%\n")
        f.write(f"- Average Latency: {metrics_s10['avg_inference_time_ms']:.4f} ms/image\n")
        f.write(f"- Training Duration: {duration_s10:.2f} min\n\n")
        
        f.write("Model S11: Hybrid Skeleton + S6 + S11 (Fusion + Hierarchical)\n")
        f.write(f"- Parameters: {student_s11.count_params():,}\n")
        f.write(f"- Chars74K Strict Accuracy: {metrics_s11['strict_accuracy']:.2f}%\n")
        f.write(f"- Chars74K Tolerant Accuracy: {metrics_s11['tolerant_accuracy']:.2f}%\n")
        f.write(f"- Average Latency: {metrics_s11['avg_inference_time_ms']:.4f} ms/image\n")
        f.write(f"- Training Duration: {duration_s11:.2f} min\n\n")
        
    print(f"\nOptimization comparison report saved to {summary_path}")

if __name__ == "__main__":
    main()
