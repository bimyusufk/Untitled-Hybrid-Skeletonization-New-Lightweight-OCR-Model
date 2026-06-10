import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from ocr_evaluation import save_ocr_evaluation_artifacts

# =====================================================================
# 1. CONFIGURATION & CONFIG PATHS
# =====================================================================
CSV_PATH = "datasets/annotations.csv"
SKELETON_BASE_DIR = "datasets/skeletonize"
IMAGE_SIZE = (32, 32)

# =====================================================================
# 2. LOAD DATASET SKELETONIZE & MATRICES
# =====================================================================
# =====================================================================
print("Memuat data hasil skeletonize berdasarkan annotation.csv...")
df = pd.read_csv(CSV_PATH)

X_data = []
y_labels = []

for index, row in df.iterrows():
    folder_name = row['Folder Name']
    label = row['Label']
    
    # Path ke folder skeletonize sesuai subfolder labelnya
    folder_path = os.path.join(SKELETON_BASE_DIR, folder_name)
    
    if not os.path.exists(folder_path):
        continue
        
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder_path, img_name)
            
            # Load citra biner skeletonize (0 dan 255)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape[:2] != IMAGE_SIZE:
                    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

                # Normalisasi ke skala 0.0 - 1.0
                img_normalized = img.astype(np.float32) / 255.0
                X_data.append(img_normalized)
                y_labels.append(str(label)) # Pastikan label berupa string

X = np.array(X_data)
y = np.array(y_labels)

# Tambahkan dimensi channel di akhir untuk CNN -> (N, 32, 32, 1)
X = np.expand_dims(X, axis=-1)

# Encode label string (0-9, A-Z, a-z) menjadi angka index (0 s.d Jumlah_Kelas-1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"Total kelas terdeteksi: {num_classes} ({''.join(label_encoder.classes_)})")
print(f"Total data siap training: {X.shape[0]} sampel")

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# =====================================================================
# 3. MEMBANGUN ARSITEKTUR MINI-CNN PENDEKATAN A (EDGE-READY)
# =====================================================================
model = models.Sequential([
    layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
    
    # Augmentasi Data Spasial Mikro (Hanya aktif saat training)
    # Sangat ringan agar tidak mengaburkan garis tipis 1px skeleton
    layers.RandomRotation(0.02, fill_mode='constant', fill_value=0.0),
    layers.RandomTranslation(height_factor=0.04, width_factor=0.04, fill_mode='constant', fill_value=0.0),
    
    # Blok Konvolusi 1: Deteksi Fitur Tepi Dasar (Menggunakan Conv2D ringan)
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)), # MaxPooling lebih baik untuk mempertahankan aktivasi garis tipis 1px
    layers.Dropout(0.2),
    
    # Blok Konvolusi 2: Kombinasi Lengkungan Makro
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    # Blok Konvolusi 3: Tekstur Geometri Lebih Tinggi
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Lapisan Klasifikasi Padat (Dense)
    layers.Flatten(),
    layers.Dense(128, activation='relu'), # Kapasitas dikurangi menjadi 128 untuk penghematan parameter
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# =====================================================================
# 4. TRAINING PROSES
# =====================================================================
print("\nMemulai proses training...")
EPOCHS = int(os.getenv("OCR_EPOCHS", "30"))
BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "32"))
OUTPUT_DIR = "ocr_evaluation_outputs"
MODEL_KEY = "hybrid_skeletonized"
MODEL_NAME = "Hybrid Skeletonized OCR"

history = model.fit(X_train, y_train, 
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    validation_data=(X_test, y_test))

# =====================================================================
# 5. EVALUASI LENGKAP MODEL OCR
# =====================================================================
print("\n=========================================================")
print("MENJALANKAN EVALUASI LENGKAP MODEL OCR HYBRID")
print("=========================================================\n")

evaluation_results = save_ocr_evaluation_artifacts(
    history=history,
    X_test=X_test,
    y_test=y_test,
    label_encoder=label_encoder,
    model=model,
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