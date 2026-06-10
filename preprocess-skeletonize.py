import os
import pandas as pd
import numpy as np
import cv2
import scipy.ndimage as ndimage
from skimage.morphology import skeletonize
from sklearn.model_selection import train_test_split

# =====================================================================
# CONFIGURATION & OUTPUT DIRECTORIES
# =====================================================================
CSV_PATH = "datasets/annotations.csv"
DATASET_DIR = "datasets"
IMAGE_SIZE = (64, 64)  # Ukuran standard untuk input Mini-CNN kamu

# Folder induk untuk menyimpan hasil visualisasi tahapan preprocessing
RAW_DIR = os.path.join(DATASET_DIR, "raw")
OTSU_BASE_DIR = os.path.join(DATASET_DIR, "otsu-thresholding")
HOLE_BASE_DIR = os.path.join(DATASET_DIR, "hole-filling")
SKELETON_BASE_DIR = os.path.join(DATASET_DIR, "skeletonize")

def preprocess_and_save_stages(image_path, img_name, folder_name):
    """
    Memproses gambar dengan Deteksi Polaritas Otomatis, Otsu Thresholding,
    Conditional Hole-Filling (hanya mengisi lubang berukuran <= 35 piksel),
    dan diakhiri dengan Skeletonization.
    """
    # 1. Load gambar dalam format Grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # 2. Resize gambar ke ukuran standar (32x32)
    img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # -----------------------------------------------------------------
    # DETEKSI POLARITAS BERBASIS PIKSEL BINGKAI (BORDER)
    # -----------------------------------------------------------------
    top_border = img_resized[0, :]
    bottom_border = img_resized[-1, :]
    left_border = img_resized[:, 0]
    right_border = img_resized[:, -1]
    
    all_border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border])
    avg_border_intensity = np.mean(all_border_pixels)
    
    # 3. Otsu Thresholding Adaptif
    if avg_border_intensity > 127:
        _, img_biner = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, img_biner = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # -----------------------------------------------------------------
    
    # Simpan Tahap 1: Otsu Thresholding
    otsu_target_dir = os.path.join(OTSU_BASE_DIR, folder_name)
    os.makedirs(otsu_target_dir, exist_ok=True)
    cv2.imwrite(os.path.join(otsu_target_dir, img_name), img_biner)
    
    # Konversi ke matriks boolean untuk pengolahan morfologi
    img_bool = img_biner > 0
    
    # -----------------------------------------------------------------
    # KONDISIONAL HOLE-FILLING (Hanya lubang <= 7 piksel)
    # -----------------------------------------------------------------
    # Ambil semua lubang yang ada pada gambar (tanpa memandang ukuran)
    all_filled = ndimage.binary_fill_holes(img_bool)
    
    # Cari posisi lubang murni dengan mengurangkan hasil fill dengan gambar asli (XOR)
    only_holes = np.logical_xor(all_filled, img_bool)
    
    # Labeli setiap lubang yang terisolasi untuk dihitung luas pikselnya
    labeled_holes, num_features = ndimage.label(only_holes)
    
    # Buat kanvas kosong untuk menampung lubang kecil yang lolos seleksi
    small_holes_mask = np.zeros_like(img_bool)
    
    # Saring setiap lubang satu per satu
    for slice_index in range(1, num_features + 1):
        hole_area = np.sum(labeled_holes == slice_index)
        
        # JIKA ukuran lubang kurang dari atau sama dengan 35 piksel, tandai untuk diisi
        if hole_area <= 35:
            small_holes_mask = np.logical_or(small_holes_mask, (labeled_holes == slice_index))
            
    # Gabungkan gambar asli dengan masker lubang kecil (Proses pengisian selektif)
    img_conditioned = np.logical_or(img_bool, small_holes_mask)
    # -----------------------------------------------------------------
    
    # Simpan Tahap 1.5: Conditional Hole Filling
    hole_target_dir = os.path.join(HOLE_BASE_DIR, folder_name)
    os.makedirs(hole_target_dir, exist_ok=True)
    img_conditioned_v = (img_conditioned * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(hole_target_dir, img_name), img_conditioned_v)
    
    # 4. Skeletonization dari hasil gambar yang sudah dibersihkan lubang kecilnya
    img_skeleton = skeletonize(img_conditioned)
    
    # Simpan Tahap 2: Skeletonize
    skeleton_target_dir = os.path.join(SKELETON_BASE_DIR, folder_name)
    os.makedirs(skeleton_target_dir, exist_ok=True)
    img_skeleton_v = (img_skeleton * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(skeleton_target_dir, img_name), img_skeleton_v)
    
    return img_skeleton.astype(np.float32)

# =====================================================================
# MAIN PIPELINE LOADING DATASET
# =====================================================================

print("Menyisir file annotation.csv...")
df = pd.read_csv(CSV_PATH)

X_data = []
y_labels = []
failed_loads = 0

for index, row in df.iterrows():
    folder_name = row['Folder Name']
    label = row['Label']
    
    folder_path = os.path.join(RAW_DIR, folder_name)
    
    if not os.path.exists(folder_path):
        continue
        
    print(f"Memproses folder: {folder_name} (Label: {label})...")
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder_path, img_name)
            
            # Jalankan fungsi pre-processing, penyimpanan berstruktur, dan ambil hasil skeletonnya
            skeleton_img = preprocess_and_save_stages(img_path, img_name, folder_name)
            
            if skeleton_img is not None:
                X_data.append(skeleton_img)
                y_labels.append(label)
            else:
                failed_loads += 1

# Ubah list menjadi numpy array standar Machine Learning
X = np.array(X_data)
y = np.array(y_labels)

# Tambahkan dimensi channel di akhir -> (N, 32, 32, 1)
X = np.expand_dims(X, axis=-1)

print(f"\n--- LOADING & PREPROCESSING SELESAI ---")
print(f"Struktur subfolder 'Sample*' berhasil dipertahankan di dalam folder pemrosesan baru.")
print(f"Total gambar diproses: {X.shape[0]}")
print(f"Total gambar gagal: {failed_loads}")
print(f"Dimensi matriks fitur X: {X.shape}")
print(f"Dimensi matriks label  y: {y.shape}")

# =====================================================================
# SPLITTING DATASET UNTUK TRAIN & TEST
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nData siap digunakan untuk training CNN!")
print(f"Ukuran X_train: {X_train.shape} | Ukuran y_train: {y_train.shape}")
print(f"Ukuran X_test : {X_test.shape}  | Ukuran y_test : {y_test.shape}")