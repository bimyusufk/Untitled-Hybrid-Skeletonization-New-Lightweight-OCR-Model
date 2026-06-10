import os
import pandas as pd
import numpy as np
import cv2
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from ocr_evaluation import save_ocr_evaluation_artifacts

# =====================================================================
# 1. CONFIGURATION & CONFIG PATHS
# =====================================================================
CSV_PATH = "datasets/annotations.csv"
RAW_BASE_DIR = "datasets/raw" # Menggunakan data MURNI/RAW, bukan skeletonize
IMAGE_SIZE = (32, 32)

# =====================================================================
# 2. LOAD DATASET ASLI (RAW IMAGES)
# =====================================================================
print("Memuat data gambar ASLI (Raw) berdasarkan annotation.csv untuk model standar...")
df = pd.read_csv(CSV_PATH)

X_data = []
y_labels = []

for index, row in df.iterrows():
    folder_name = row['Folder Name']
    label = row['Label']
    
    folder_path = os.path.join(RAW_BASE_DIR, folder_name)
    
    if not os.path.exists(folder_path):
        continue
        
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder_path, img_name)
            
            # Load gambar biasa (Grayscale)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize ke ukuran standar 32x32
                img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                # Normalisasi piksel (0 s.d 1)
                img_normalized = img_resized.astype(np.float32) / 255.0
                X_data.append(img_normalized)
                y_labels.append(str(label))

X = np.array(X_data)
y = np.array(y_labels)

# Tambahkan dimensi channel di akhir -> (N, 32, 32, 1)
X = np.expand_dims(X, axis=-1)

# Encode label string menjadi index angka
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"Total kelas terdeteksi: {num_classes}")
print(f"Total data siap training: {X.shape[0]} sampel")

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# =====================================================================
# 3. MEMBANGUN ARSITEKTUR CNN STANDAR INDUSTRI (LEBIH DALAM)
# =====================================================================
# Model standar industri harus lebih dalam karena harus mengekstrak 
# fitur tekstur, ketebalan font, dan variasi kontras yang ada pada gambar asli.
model_standard = models.Sequential([
    layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
    
    # Blok 1: Menggunakan filter lebih banyak (32) untuk menangkap ketebalan font
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Blok 2: Lebih dalam (64 filter) untuk mengekstrak fitur lengkungan makro
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Blok 3: Lapisan ekstra untuk ekstraksi tekstur tepi yang tebal
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'), # Dense layer lebih besar (256 neuron)
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

model_standard.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

model_standard.summary()

# =====================================================================
# 4. TRAINING PROSES
# =====================================================================
print("\nMemulai proses training model CNN Standar...")
EPOCHS = int(os.getenv("OCR_EPOCHS", "15"))
BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "32"))
OUTPUT_DIR = "ocr_evaluation_outputs"
MODEL_KEY = "standard_raw"
MODEL_NAME = "Standard Raw OCR"

history = model_standard.fit(X_train, y_train, 
                             epochs=EPOCHS, 
                             batch_size=BATCH_SIZE, 
                             validation_data=(X_test, y_test))

# =====================================================================
# 5. EVALUASI LENGKAP MODEL OCR
# =====================================================================
print("\n=========================================================")
print("MENJALANKAN EVALUASI LENGKAP MODEL OCR STANDAR")
print("=========================================================\n")

evaluation_results = save_ocr_evaluation_artifacts(
    history=history,
    X_test=X_test,
    y_test=y_test,
    label_encoder=label_encoder,
    model=model_standard,
    output_dir=OUTPUT_DIR,
    model_key=MODEL_KEY,
    model_name=MODEL_NAME,
    batch_size=BATCH_SIZE,
)

metrics = evaluation_results["metrics"]

print(f"Hasil Pengujian Pada {metrics['total_test']} Data Test:")
print(f"-------------------------------------------------")
print(f"1. Benar Mutlak (Strict Accuracy)             : {metrics['strict_correct']} data ({metrics['strict_accuracy']:.2f}%)")
print(f"2. Salah Case (Huruf Benar tapi Kapital Salah) : {metrics['case_error_but_char_correct']} data")
print(f"3. Benar-Benar Salah Karakter                  : {metrics['total_wrong']} data")
print(f"4. Akurasi Toleran (Case-Insensitive)          : {metrics['tolerant_accuracy']:.2f}%")
print(f"5. Total Waktu Inferensi                       : {metrics['total_inference_time_sec']:.4f} detik")
print(f"6. Rerata Waktu per Gambar                     : {metrics['avg_inference_time_ms']:.2f} ms / gambar")
print(f"-------------------------------------------------")
print("Artefak evaluasi tersimpan di:")
print(f"- Training curves   : {evaluation_results['history_path']}")
print(f"- Confusion matrix  : {evaluation_results['confusion_path']}")
print(f"- Prediction samples: {evaluation_results['samples_path']}")
print(f"- Inference chart   : {evaluation_results['inference_path']}")
print(f"- Classification report: {evaluation_results['report_path']}")
print(f"- Summary CSV       : {evaluation_results['summary_path']}")
print(f"- Comparison chart  : {evaluation_results['comparison_path']}")