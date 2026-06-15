import os
import time
import yaml
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

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
IMAGE_SIZE = (64, 64)

# Weight paths
WEIGHTS_PATH_WIDTH16 = "research_outputs/models/pretrained_se_64x64.weights.h5"
WEIGHTS_PATH_WIDTH24 = "research_outputs/models/pretrained_se_64x64_width24.weights.h5"

tf.random.set_seed(SEED)
np.random.seed(SEED)

LABEL_ENCODER = get_label_encoder_chars74k()
NUM_CLASSES = len(LABEL_ENCODER.classes_)

# =====================================================================
# DATA LOADER
# =====================================================================
def load_skeleton_dataset():
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
                        img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                    else:
                        img_resized = img.copy()
                        
                    X_data.append(img_resized.astype(np.float32) / 255.0)
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
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)

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
        y_prediction = self.student(x, training=False)
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

# 1. Backbone width 16 (for 800K model)
def build_backbone_width16(inputs, dense_units=128):
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.2, name='drop_1')(x)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.2, name='drop_2')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_5')(x)
    x = layers.BatchNormalization(name='bn_5')(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_6')(x)
    x = layers.BatchNormalization(name='bn_6')(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.3, name='drop_3')(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_7')(x)
    x = layers.BatchNormalization(name='bn_7')(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_8')(x)
    x = layers.BatchNormalization(name='bn_8')(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.3, name='drop_4')(x)
    
    x = layers.Flatten(name='flat')(x)
    x = layers.Dense(dense_units, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_9')(x)
    return x

# 2. Backbone width 24 (for 1.0M & 1.2M models)
def build_backbone_width24(inputs, dense_units=128):
    x = layers.Conv2D(24, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Conv2D(24, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.2, name='drop_1')(x)
    
    x = layers.Conv2D(48, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Conv2D(48, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.2, name='drop_2')(x)
    
    x = layers.Conv2D(96, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_5')(x)
    x = layers.BatchNormalization(name='bn_5')(x)
    x = layers.Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_6')(x)
    x = layers.BatchNormalization(name='bn_6')(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.3, name='drop_3')(x)
    
    x = layers.Conv2D(192, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_7')(x)
    x = layers.BatchNormalization(name='bn_7')(x)
    x = layers.Conv2D(192, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_8')(x)
    x = layers.BatchNormalization(name='bn_8')(x)
    x = se_block(x, ratio=8)
    x = layers.Dropout(0.3, name='drop_4')(x)
    
    x = layers.Flatten(name='flat')(x)
    x = layers.Dense(dense_units, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_9')(x)
    return x

# Student Builders
def build_student_800k(num_classes, input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    x = build_backbone_width16(inputs, dense_units=256)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, name='output')(x)
    return models.Model(inputs=inputs, outputs=outputs)

def build_student_1m(num_classes, input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    x = build_backbone_width24(inputs, dense_units=128)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, name='output')(x)
    return models.Model(inputs=inputs, outputs=outputs)

def build_student_1_2m(num_classes, input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    x = build_backbone_width24(inputs, dense_units=192)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, name='output')(x)
    return models.Model(inputs=inputs, outputs=outputs)

# Pretraining baseline model builders (for topological weight loading)
def build_baseline_width16(num_classes):
    inputs = layers.Input(shape=(64, 64, 1))
    x = build_backbone_width16(inputs, dense_units=128)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)

def build_baseline_width24(num_classes):
    inputs = layers.Input(shape=(64, 64, 1))
    x = build_backbone_width24(inputs, dense_units=128)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)

# Teacher Builder
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

# Weight Loader Helper
def load_backbone_weights(target_model, weights_path, baseline_builder):
    if not os.path.exists(weights_path):
        print(f"[WARNING] Weights path {weights_path} not found. Skipping weight loading.")
        return
        
    print(f"Initializing student backbone weights from {weights_path}...")
    baseline = baseline_builder(NUM_CLASSES)
    baseline.load_weights(weights_path)
    
    transferred = 0
    for layer in baseline.layers:
        try:
            target_layer = target_model.get_layer(layer.name)
            if len(layer.get_weights()) == len(target_layer.get_weights()):
                shape_match = True
                for w_b, w_t in zip(layer.get_weights(), target_layer.get_weights()):
                    if w_b.shape != w_t.shape:
                        shape_match = False
                        break
                if shape_match:
                    target_layer.set_weights(layer.get_weights())
                    transferred += 1
        except Exception:
            continue
    print(f"Successfully transferred weights for {transferred} layers.")

# =====================================================================
# MAIN RUN
# =====================================================================
def main():
    print("=== STARTING SCALED MODELS EXPERIMENTS (64x64) ===")
    
    # Load dataset
    X, y = load_skeleton_dataset()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Apply S1 offline augmentation
    print("Augmenting Chars74K training dataset (Student skeleton)...")
    X_train_aug, y_train_aug = augment_training_set(X_train, y_train, multiplier=9)
    
    EPOCHS = int(os.getenv("OCR_EPOCHS", "30"))
    BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "64"))
    
    # 1. Train Teacher Model (MobileNetV2 ImageNet)
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
    
    results_dict = {}
    
    # -----------------------------------------------------------------
    # MODEL 1: Scaled 800K Model
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### TRAINING MODEL 1: SCALED 800K (Backbone W16 + Dense 256)    ###")
    print("#################################################################")
    student_800k = build_student_800k(NUM_CLASSES)
    load_backbone_weights(student_800k, WEIGHTS_PATH_WIDTH16, build_baseline_width16)
    
    distiller_800k = Distiller(student=student_800k, teacher=teacher)
    distiller_800k.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.5, temperature=3.0
    )
    
    t0 = time.time()
    history_800k = distiller_800k.fit(
        X_train_aug, y_train_aug,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)],
        verbose=2
    )
    duration_800k = (time.time() - t0) / 60.0
    
    # Fix activation to softmax for save/inference
    student_800k.get_layer("output").activation = tf.keras.activations.softmax
    student_800k.save("research_outputs/models/distilled_student_scaled_800k.keras")
    
    eval_800k = save_ocr_evaluation_artifacts(
        history=history_800k, X_test=X_test, y_test=y_test,
        label_encoder=LABEL_ENCODER, model=student_800k,
        output_dir="ocr_evaluation_outputs_scaled_800k",
        model_key="hybrid_scaled_800k",
        model_name="Hybrid Skeleton Scaled 800K (Dense 256)",
        batch_size=BATCH_SIZE
    )
    results_dict["800k"] = {
        "metrics": eval_800k["metrics"],
        "params": student_800k.count_params(),
        "duration": duration_800k
    }

    # -----------------------------------------------------------------
    # MODEL 2: Scaled 1.0M Model
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### TRAINING MODEL 2: SCALED 1.0M (Backbone W24 + Dense 128)    ###")
    print("#################################################################")
    student_1m = build_student_1m(NUM_CLASSES)
    load_backbone_weights(student_1m, WEIGHTS_PATH_WIDTH24, build_baseline_width24)
    
    distiller_1m = Distiller(student=student_1m, teacher=teacher)
    distiller_1m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.5, temperature=3.0
    )
    
    t1 = time.time()
    history_1m = distiller_1m.fit(
        X_train_aug, y_train_aug,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)],
        verbose=2
    )
    duration_1m = (time.time() - t1) / 60.0
    
    student_1m.get_layer("output").activation = tf.keras.activations.softmax
    student_1m.save("research_outputs/models/distilled_student_scaled_1m.keras")
    
    eval_1m = save_ocr_evaluation_artifacts(
        history=history_1m, X_test=X_test, y_test=y_test,
        label_encoder=LABEL_ENCODER, model=student_1m,
        output_dir="ocr_evaluation_outputs_scaled_1m",
        model_key="hybrid_scaled_1m",
        model_name="Hybrid Skeleton Scaled 1.0M (Backbone W24)",
        batch_size=BATCH_SIZE
    )
    results_dict["1m"] = {
        "metrics": eval_1m["metrics"],
        "params": student_1m.count_params(),
        "duration": duration_1m
    }

    # -----------------------------------------------------------------
    # MODEL 3: Scaled 1.2M Model
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### TRAINING MODEL 3: SCALED 1.2M (Backbone W24 + Dense 192)    ###")
    print("#################################################################")
    student_1_2m = build_student_1_2m(NUM_CLASSES)
    load_backbone_weights(student_1_2m, WEIGHTS_PATH_WIDTH24, build_baseline_width24)
    
    distiller_1_2m = Distiller(student=student_1_2m, teacher=teacher)
    distiller_1_2m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.5, temperature=3.0
    )
    
    t2 = time.time()
    history_1_2m = distiller_1_2m.fit(
        X_train_aug, y_train_aug,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)],
        verbose=2
    )
    duration_1_2m = (time.time() - t2) / 60.0
    
    student_1_2m.get_layer("output").activation = tf.keras.activations.softmax
    student_1_2m.save("research_outputs/models/distilled_student_scaled_1_2m.keras")
    
    eval_1_2m = save_ocr_evaluation_artifacts(
        history=history_1_2m, X_test=X_test, y_test=y_test,
        label_encoder=LABEL_ENCODER, model=student_1_2m,
        output_dir="ocr_evaluation_outputs_scaled_1.2m",
        model_key="hybrid_scaled_1_2m",
        model_name="Hybrid Skeleton Scaled 1.2M (Backbone W24)",
        batch_size=BATCH_SIZE
    )
    results_dict["1.2m"] = {
        "metrics": eval_1_2m["metrics"],
        "params": student_1_2m.count_params(),
        "duration": duration_1_2m
    }

    # -----------------------------------------------------------------
    # COMPARISON REPORT
    # -----------------------------------------------------------------
    summary_path = "scaled_models_comparison_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== SCALED MODELS ACCURACY COMPARISON REPORT (64x64) ===\n\n")
        for key in ["800k", "1m", "1.2m"]:
            res = results_dict[key]
            f.write(f"Model: {key.upper()}\n")
            f.write(f"- Parameter Count: {res['params']:,}\n")
            f.write(f"- Chars74K Strict Accuracy: {res['metrics']['strict_accuracy']:.2f}%\n")
            f.write(f"- Chars74K Tolerant Accuracy: {res['metrics']['tolerant_accuracy']:.2f}%\n")
            f.write(f"- Average Inference Time: {res['metrics']['avg_inference_time_ms']:.4f} ms/image\n")
            f.write(f"- Training Duration: {res['duration']:.2f} min\n\n")
            
    print(f"\nScaled models comparison report saved to {summary_path}")

if __name__ == "__main__":
    main()
