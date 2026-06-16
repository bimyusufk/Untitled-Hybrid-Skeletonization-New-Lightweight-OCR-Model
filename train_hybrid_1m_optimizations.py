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
# CUSTOM LAYERS
# =====================================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class CoordinateAttention(layers.Layer):
    def __init__(self, channels, reduction=8, **kwargs):
        super(CoordinateAttention, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.reduced_channels = max(8, channels // reduction)

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.reduced_channels, (1, 1), use_bias=False, name="conv1")
        self.bn1 = layers.BatchNormalization(name="bn1")
        self.conv_h = layers.Conv2D(self.channels, (1, 1), activation="sigmoid", use_bias=False, name="conv_h")
        self.conv_w = layers.Conv2D(self.channels, (1, 1), activation="sigmoid", use_bias=False, name="conv_w")
        super(CoordinateAttention, self).build(input_shape)

    def call(self, inputs):
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        
        # X pool (average along width) -> shape (batch, h, 1, channels)
        x_pool = tf.reduce_mean(inputs, axis=2, keepdims=True)
        # Y pool (average along height) -> shape (batch, 1, w, channels)
        y_pool = tf.reduce_mean(inputs, axis=1, keepdims=True)
        
        # Transpose Y pool -> (batch, w, 1, channels)
        y_pool_t = tf.transpose(y_pool, perm=[0, 2, 1, 3])
        
        # Concatenate along height -> (batch, h+w, 1, channels)
        concat = tf.concat([x_pool, y_pool_t], axis=1)
        
        # Shared 1x1 conv + bn + relu
        x = self.conv1(concat)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        
        # Split back
        x_h = x[:, :h, :, :]
        x_w_t = x[:, h:, :, :]
        
        # Transpose back
        x_w = tf.transpose(x_w_t, perm=[0, 2, 1, 3])
        
        # Sigmoid gates
        s_h = self.conv_h(x_h)
        s_w = self.conv_w(x_w)
        
        return inputs * s_h * s_w
        
    def get_config(self):
        config = super(CoordinateAttention, self).get_config()
        config.update({
            "channels": self.channels,
            "reduction": self.reduction
        })
        return config

# =====================================================================
# DISTILLER CLASSES
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

class HintDistiller(models.Model):
    def __init__(self, student, teacher, student_proj_layer):
        super(HintDistiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.student_proj_layer = student_proj_layer

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, hint_loss_fn, alpha=0.5, beta=1.0, temperature=3.0):
        super(HintDistiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.hint_loss_fn = hint_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        teacher_predictions, teacher_features = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_predictions, student_features = self.student(x, training=True)
            student_proj_features = self.student_proj_layer(student_features, training=True)

            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=-1),
                tf.nn.softmax(student_predictions / self.temperature, axis=-1)
            )
            hint_loss = self.hint_loss_fn(teacher_features, student_proj_features)
            
            loss = (self.alpha * student_loss + 
                    (1 - self.alpha) * distillation_loss * (self.temperature ** 2) + 
                    self.beta * hint_loss)

        trainable_vars = self.student.trainable_variables + self.student_proj_layer.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_predictions)

        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "loss": loss, 
            "student_loss": student_loss, 
            "distillation_loss": distillation_loss,
            "hint_loss": hint_loss
        })
        return results

    def test_step(self, data):
        x, y = data
        y_prediction, _ = self.student(x, training=False)
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
# 1. Base backbone width 24 (used by S12, S14, and Baseline)
def build_backbone_width24_se(inputs):
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
    return x

# 2. S13 backbone using CoordAttention instead of SE
def build_backbone_width24_coord(inputs):
    x = layers.Conv2D(24, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Conv2D(24, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = CoordinateAttention(24, reduction=8, name='coord_attn_1')(x)
    x = layers.Dropout(0.2, name='drop_1')(x)
    
    x = layers.Conv2D(48, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Conv2D(48, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = CoordinateAttention(48, reduction=8, name='coord_attn_2')(x)
    x = layers.Dropout(0.2, name='drop_2')(x)
    
    x = layers.Conv2D(96, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_5')(x)
    x = layers.BatchNormalization(name='bn_5')(x)
    x = layers.Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_6')(x)
    x = layers.BatchNormalization(name='bn_6')(x)
    x = CoordinateAttention(96, reduction=8, name='coord_attn_3')(x)
    x = layers.Dropout(0.3, name='drop_3')(x)
    
    x = layers.Conv2D(192, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_7')(x)
    x = layers.BatchNormalization(name='bn_7')(x)
    x = layers.Conv2D(192, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_8')(x)
    x = layers.BatchNormalization(name='bn_8')(x)
    x = CoordinateAttention(192, reduction=8, name='coord_attn_4')(x)
    x = layers.Dropout(0.3, name='drop_4')(x)
    return x

# 3. Backbone with middle output (for S15 Multi-Scale Skip Connection)
def build_backbone_width24_features_block3(inputs):
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
    x_block3 = se_block(x, ratio=8)
    x = layers.Dropout(0.3, name='drop_3')(x_block3)
    
    x = layers.Conv2D(192, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv2d_7')(x)
    x = layers.BatchNormalization(name='bn_7')(x)
    x = layers.Conv2D(192, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv2d_8')(x)
    x = layers.BatchNormalization(name='bn_8')(x)
    x_block4 = se_block(x, ratio=8)
    x_block4 = layers.Dropout(0.3, name='drop_4')(x_block4)
    
    return x_block3, x_block4

# Student builders
def build_student_1m_baseline(num_classes, input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    x = build_backbone_width24_se(inputs)
    x = layers.Flatten(name='flat')(x)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_9')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, name='output')(x)
    return models.Model(inputs=inputs, outputs=outputs)

def build_student_1m_s13(num_classes, input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    x = build_backbone_width24_coord(inputs)
    x = layers.Flatten(name='flat')(x)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_9')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, name='output')(x)
    return models.Model(inputs=inputs, outputs=outputs)

def build_student_1m_s14(num_classes, input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    x_features = build_backbone_width24_se(inputs)
    x = layers.Flatten(name='flat')(x_features)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_9')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, name='output')(x)
    return models.Model(inputs=inputs, outputs=[outputs, x_features])

def build_student_1m_s15(num_classes, input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    x_block3, x_block4 = build_backbone_width24_features_block3(inputs)
    
    # Downsample Block 3 to match Block 4 shape (4, 4)
    x_block3_down = layers.MaxPooling2D((2,2), name='skip_pool')(x_block3)
    
    # Concatenate Block 3 and Block 4 features along channels axis
    x_concat = layers.Concatenate(axis=-1, name='skip_concat')([x_block4, x_block3_down])
    
    x = layers.Flatten(name='flat')(x_concat)
    x = layers.Dense(256, activation='relu', name='dense_1')(x) # Dense 256 for parameter alignment (~960K params)
    x = layers.BatchNormalization(name='bn_9')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, name='output')(x)
    return models.Model(inputs=inputs, outputs=outputs)

# Baseline Builder (for loading pre-trained weights)
def build_baseline_width24(num_classes):
    inputs = layers.Input(shape=(64, 64, 1))
    x = build_backbone_width24_se(inputs)
    x = layers.Flatten(name='flat')(x)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_9')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)

# Teacher Builders
def build_teacher_mobilenetv2(num_classes, input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda t: (t * 2.0) - 1.0)(inputs)
    x = layers.Concatenate(axis=-1)([x, x, x])
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        alpha=1.0, include_top=False, weights='imagenet', pooling=None
    )
    base_model.trainable = True
    
    features = base_model(x) # shape: (batch, 2, 2, 1280)
    x_pool = layers.GlobalAveragePooling2D()(features)
    
    dense = layers.Dense(256, activation='relu', name='teacher_dense')(x_pool)
    bn = layers.BatchNormalization(name='teacher_bn')(dense)
    drop = layers.Dropout(0.3)(bn)
    logits = layers.Dense(num_classes, name='teacher_output')(drop)
    
    train_model = models.Model(inputs=inputs, outputs=logits)
    distill_model = models.Model(inputs=inputs, outputs=[logits, features])
    return train_model, distill_model

def build_teacher_efficientnetb0(num_classes, input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda t: t * 255.0)(inputs)
    x = layers.Concatenate(axis=-1)([x, x, x])
    
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False, weights='imagenet', pooling='avg'
    )
    base_model.trainable = True
    x = base_model(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

# Helper to transfer weights topologically
def load_backbone_weights(target_model, weights_path):
    if not os.path.exists(weights_path):
        print(f"[WARNING] Weights path {weights_path} not found. Skipping weight loading.")
        return
        
    print(f"Initializing student backbone weights from {weights_path}...")
    baseline = build_baseline_width24(NUM_CLASSES)
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
    print("=== STARTING 1M OPTIMIZATIONS EXPERIMENTS (64x64) ===")
    
    # Load dataset
    X, y = load_skeleton_dataset()
    
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
    # TEACHER TRAINING (MobileNetV2)
    # -----------------------------------------------------------------
    print("\n--- Training MobileNetV2 Teacher Model ---")
    teacher_mobilenet_train, teacher_mobilenet_distill = build_teacher_mobilenetv2(NUM_CLASSES)
    
    teacher_trainable = models.Sequential([
        teacher_mobilenet_train,
        layers.Activation('softmax')
    ])
    teacher_trainable = compile_keras_model(teacher_trainable, CONFIG)
    
    callbacks_teacher = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1
        )
    ]
    
    teacher_trainable.fit(
        X_train_aug, y_train_aug,
        epochs=int(os.getenv("OCR_EPOCHS_TEACHER", "30")),
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks_teacher,
        verbose=2
    )
    
    # -----------------------------------------------------------------
    # TEACHER TRAINING (EfficientNetB0 for S12)
    # -----------------------------------------------------------------
    print("\n--- Training EfficientNetB0 Teacher Model (S12) ---")
    teacher_effnet = build_teacher_efficientnetb0(NUM_CLASSES)
    
    teacher_effnet_trainable = models.Sequential([
        teacher_effnet,
        layers.Activation('softmax')
    ])
    teacher_effnet_trainable = compile_keras_model(teacher_effnet_trainable, CONFIG)
    
    teacher_effnet_trainable.fit(
        X_train_aug, y_train_aug,
        epochs=int(os.getenv("OCR_EPOCHS_TEACHER", "30")),
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks_teacher,
        verbose=2
    )
    
    results_dict = {}
    
    # -----------------------------------------------------------------
    # CONFIGURATION 1: Baseline 1M
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### RUNNING 1M CONFIGURATION 1: BASELINE 1M (SE + MobNet Teacher)###")
    print("#################################################################")
    student_base = build_student_1m_baseline(NUM_CLASSES)
    load_backbone_weights(student_base, WEIGHTS_PATH_WIDTH24)
    
    distiller_base = Distiller(student=student_base, teacher=teacher_mobilenet_train)
    distiller_base.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.5, temperature=3.0
    )
    
    t_start = time.time()
    history_base = distiller_base.fit(
        X_train_aug, y_train_aug,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)],
        verbose=2
    )
    duration_base = (time.time() - t_start) / 60.0
    student_base.get_layer("output").activation = tf.keras.activations.softmax
    
    eval_base = save_ocr_evaluation_artifacts(
        history=history_base, X_test=X_test, y_test=y_test,
        label_encoder=LABEL_ENCODER, model=student_base,
        output_dir="ocr_evaluation_outputs_1m_baseline",
        model_key="hybrid_1m_baseline",
        model_name="Hybrid Skeleton 1M (Baseline)",
        batch_size=BATCH_SIZE
    )
    results_dict["baseline"] = {
        "metrics": eval_base["metrics"],
        "params": student_base.count_params(),
        "duration": duration_base
    }

    # -----------------------------------------------------------------
    # CONFIGURATION 2: S12 (EfficientNetB0 Teacher)
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### RUNNING 1M CONFIGURATION 2: S12 (EfficientNetB0 Teacher)    ###")
    print("#################################################################")
    student_s12 = build_student_1m_baseline(NUM_CLASSES)
    load_backbone_weights(student_s12, WEIGHTS_PATH_WIDTH24)
    
    distiller_s12 = Distiller(student=student_s12, teacher=teacher_effnet)
    distiller_s12.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.5, temperature=3.0
    )
    
    t_start = time.time()
    history_s12 = distiller_s12.fit(
        X_train_aug, y_train_aug,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)],
        verbose=2
    )
    duration_s12 = (time.time() - t_start) / 60.0
    student_s12.get_layer("output").activation = tf.keras.activations.softmax
    
    eval_s12 = save_ocr_evaluation_artifacts(
        history=history_s12, X_test=X_test, y_test=y_test,
        label_encoder=LABEL_ENCODER, model=student_s12,
        output_dir="ocr_evaluation_outputs_1m_s12",
        model_key="hybrid_1m_s12",
        model_name="Hybrid Skeleton 1M (S12: EffNet Teacher)",
        batch_size=BATCH_SIZE
    )
    results_dict["s12"] = {
        "metrics": eval_s12["metrics"],
        "params": student_s12.count_params(),
        "duration": duration_s12
    }

    # -----------------------------------------------------------------
    # CONFIGURATION 3: S13 (Coordinate Attention)
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### RUNNING 1M CONFIGURATION 3: S13 (Coordinate Attention)      ###")
    print("#################################################################")
    student_s13 = build_student_1m_s13(NUM_CLASSES)
    # Load weights (SE blocks will be skipped safely by load_backbone_weights because of name/shape mismatch, but all conv/bn weights load perfectly!)
    load_backbone_weights(student_s13, WEIGHTS_PATH_WIDTH24)
    
    distiller_s13 = Distiller(student=student_s13, teacher=teacher_mobilenet_train)
    distiller_s13.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.5, temperature=3.0
    )
    
    t_start = time.time()
    history_s13 = distiller_s13.fit(
        X_train_aug, y_train_aug,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)],
        verbose=2
    )
    duration_s13 = (time.time() - t_start) / 60.0
    student_s13.get_layer("output").activation = tf.keras.activations.softmax
    
    eval_s13 = save_ocr_evaluation_artifacts(
        history=history_s13, X_test=X_test, y_test=y_test,
        label_encoder=LABEL_ENCODER, model=student_s13,
        output_dir="ocr_evaluation_outputs_1m_s13",
        model_key="hybrid_1m_s13",
        model_name="Hybrid Skeleton 1M (S13: CoordAttention)",
        batch_size=BATCH_SIZE
    )
    results_dict["s13"] = {
        "metrics": eval_s13["metrics"],
        "params": student_s13.count_params(),
        "duration": duration_s13
    }

    # -----------------------------------------------------------------
    # CONFIGURATION 4: S14 (Hint Loss / Feature Mimicking)
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### RUNNING 1M CONFIGURATION 4: S14 (Hint Loss / Mimicking)     ###")
    print("#################################################################")
    student_s14 = build_student_1m_s14(NUM_CLASSES)
    load_backbone_weights(student_s14, WEIGHTS_PATH_WIDTH24)
    
    # 1x1 projection Conv2D to map student Block 4 features (4, 4, 192) to teacher features (2, 2, 1280)
    student_proj_layer = models.Sequential([
        layers.Conv2D(1280, (3, 3), strides=(2, 2), padding='same', name='hint_projection')
    ])
    
    distiller_s14 = HintDistiller(student=student_s14, teacher=teacher_mobilenet_distill, student_proj_layer=student_proj_layer)
    distiller_s14.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        hint_loss_fn=tf.keras.losses.MeanSquaredError(),
        alpha=0.5, beta=1.0, temperature=3.0
    )
    
    t_start = time.time()
    history_s14 = distiller_s14.fit(
        X_train_aug, y_train_aug,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)],
        verbose=2
    )
    duration_s14 = (time.time() - t_start) / 60.0
    
    # Build a standalone student model for deployment (without feature outputs)
    student_s14_deploy = build_student_1m_baseline(NUM_CLASSES)
    # Copy weights
    for layer in student_s14.layers:
        try:
            student_s14_deploy.get_layer(layer.name).set_weights(layer.get_weights())
        except Exception:
            continue
    student_s14_deploy.get_layer("output").activation = tf.keras.activations.softmax
    student_s14_deploy.save("research_outputs/models/distilled_student_scaled_1m_s14.keras")
    
    eval_s14 = save_ocr_evaluation_artifacts(
        history=history_s14, X_test=X_test, y_test=y_test,
        label_encoder=LABEL_ENCODER, model=student_s14_deploy,
        output_dir="ocr_evaluation_outputs_1m_s14",
        model_key="hybrid_1m_s14",
        model_name="Hybrid Skeleton 1M (S14: Hint Loss)",
        batch_size=BATCH_SIZE
    )
    results_dict["s14"] = {
        "metrics": eval_s14["metrics"],
        "params": student_s14_deploy.count_params() + student_proj_layer.count_params(),
        "duration": duration_s14
    }

    # -----------------------------------------------------------------
    # CONFIGURATION 5: S15 (Multi-Scale Aggregation)
    # -----------------------------------------------------------------
    print("\n\n#################################################################")
    print("### RUNNING 1M CONFIGURATION 5: S15 (Multi-Scale Aggregation)   ###")
    print("#################################################################")
    student_s15 = build_student_1m_s15(NUM_CLASSES)
    load_backbone_weights(student_s15, WEIGHTS_PATH_WIDTH24)
    
    distiller_s15 = Distiller(student=student_s15, teacher=teacher_mobilenet_train)
    distiller_s15.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.5, temperature=3.0
    )
    
    t_start = time.time()
    history_s15 = distiller_s15.fit(
        X_train_aug, y_train_aug,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)],
        verbose=2
    )
    duration_s15 = (time.time() - t_start) / 60.0
    student_s15.get_layer("output").activation = tf.keras.activations.softmax
    student_s15.save("research_outputs/models/distilled_student_scaled_1m_s15.keras")
    
    eval_s15 = save_ocr_evaluation_artifacts(
        history=history_s15, X_test=X_test, y_test=y_test,
        label_encoder=LABEL_ENCODER, model=student_s15,
        output_dir="ocr_evaluation_outputs_1m_s15",
        model_key="hybrid_1m_s15",
        model_name="Hybrid Skeleton 1M (S15: Multi-Scale Skip)",
        batch_size=BATCH_SIZE
    )
    results_dict["s15"] = {
        "metrics": eval_s15["metrics"],
        "params": student_s15.count_params(),
        "duration": duration_s15
    }

    # -----------------------------------------------------------------
    # COMPARISON REPORT
    # -----------------------------------------------------------------
    summary_path = "optimizations_1m_comparison_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== 1M OPTIMIZATIONS COMPARISON REPORT (64x64) ===\n\n")
        for key in ["baseline", "s12", "s13", "s14", "s15"]:
            res = results_dict[key]
            f.write(f"Model: {key.upper()}\n")
            f.write(f"- Parameter Count: {res['params']:,}\n")
            f.write(f"- Chars74K Strict Accuracy: {res['metrics']['strict_accuracy']:.2f}%\n")
            f.write(f"- Chars74K Tolerant Accuracy: {res['metrics']['tolerant_accuracy']:.2f}%\n")
            f.write(f"- Average Inference Time: {res['metrics']['avg_inference_time_ms']:.4f} ms/image\n")
            f.write(f"- Training Duration: {res['duration']:.2f} min\n\n")
            
    print(f"\n1M Optimizations comparison report saved to {summary_path}")

if __name__ == "__main__":
    main()
