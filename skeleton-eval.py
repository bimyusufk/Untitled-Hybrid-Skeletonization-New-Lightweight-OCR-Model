import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import euler_number

# =====================================================================
# CONFIGURATION
# =====================================================================
SKELETON_BASE_DIR = "datasets/skeletonize"
CSV_PATH = "datasets/annotations.csv"

print("--- MEMULAI ANALISIS TOPOLOGI SKELETON MENDALAM ---\n")

df = pd.read_csv(CSV_PATH)
unique_labels = sorted(df['Label'].unique())

# Tempat menampung data statistik per label
stats_summary = []

for label_char in unique_labels:
    sample_folders = df[df['Label'] == label_char]['Folder Name'].values
    if len(sample_folders) == 0: continue
    
    folder_path = os.path.join(SKELETON_BASE_DIR, sample_folders[0])
    if not os.path.exists(folder_path): continue
    
    images = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    if len(images) == 0: continue
    
    # List pengumpul untuk satu label grup
    list_length = []
    list_endpoints = []
    list_junctions = []
    list_loops = []
    
    for img_name in images:
        img = cv2.imread(os.path.join(folder_path, img_name), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Konversi ke biner 0 dan 1
        bin_img = (img == 255).astype(np.uint8)
        
        # 1. Hitung Panjang Rangka (Total Piksel)
        total_pixels = np.sum(bin_img)
        list_length.append(total_pixels)
        
        # 2. Deteksi End Points & Junctions menggunakan hitungan tetangga 3x3 via filter2D cepat
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        neighbor_sum = cv2.filter2D(bin_img, cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)
        # Hanya hitung piksel yang merupakan bagian dari rangka (bin_img == 1)
        neighbor_map = neighbor_sum * bin_img
        
        endpoints = np.sum(neighbor_map == 1)   # Hanya 1 tetangga = Ujung
        junctions = np.sum(neighbor_map >= 3)   # 3 atau lebih tetangga = Persimpangan
        
        list_endpoints.append(endpoints)
        list_junctions.append(junctions)
        
        # 3. Hitung Jumlah Lubang (Loops) berdasarkan Euler Number
        # Euler Number = Jumlah Objek - Jumlah Lubang. Karena objek selalu 1 (rangka menyatu),
        # maka Jumlah Lubang = 1 - Euler Number
        e_num = euler_number(bin_img, connectivity=2)
        loops = max(0, 1 - e_num)
        list_loops.append(loops)
        
    # Ambil nilai rata-rata (mean) untuk label ini
    stats_summary.append({
        'Label': label_char,
        'Avg_Length': np.mean(list_length),
        'Avg_Endpoints': np.mean(list_endpoints),
        'Avg_Junctions': np.mean(list_junctions),
        'Avg_Loops': np.mean(list_loops)
    })

# Ubah ke Dataframe untuk visualisasi tabel yang rapi
df_stats = pd.DataFrame(stats_summary)
print(df_stats.to_string(index=False, formatters={
    'Avg_Length': '{:,.1f}'.format,
    'Avg_Endpoints': '{:,.1f}'.format,
    'Avg_Junctions': '{:,.1f}'.format,
    'Avg_Loops': '{:,.1f}'.format
}))