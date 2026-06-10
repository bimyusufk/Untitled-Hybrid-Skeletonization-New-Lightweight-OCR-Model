import matplotlib
matplotlib.use("TkAgg")  # Paksa pakai Tkinter, bukan Qt

import matplotlib.pyplot as plt
import mahotas
import scipy.ndimage as ndimage
from skimage.morphology import skeletonize 
import numpy as np

# 1. Loading image asli lewat mahotas
img = mahotas.imread("horse2.png")

# 2. Filtering & Otsu thresholding
img = img.max(2)
T_otsu = mahotas.otsu(img)

# Menggunakan kurang dari (<) agar kudanya yang berwarna putih (True)
img_biner = img < T_otsu 

# =====================================================================
# PROSES PEMBERSIHAN (Hole Filling & Smoothing)
# =====================================================================
# Tampal lubang di dalam badan
img_filled = ndimage.binary_fill_holes(img_biner)

# Smoothing menggunakan Closing & Opening
se = np.ones((7, 7), dtype=bool)
img_closed = mahotas.close(img_filled, Bc=se)
img_perfectly_clean = mahotas.open(img_closed, Bc=se)

# =====================================================================
# 3. PROSES SKELETONIZATION
# =====================================================================
kuda_kurus = skeletonize(img_perfectly_clean)

# =====================================================================
# DISPLAY MENGGUNAKAN WINDOW PLOT AXES
# =====================================================================
# Membuat susunan plot 1 baris x 3 kolom
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Hasil Thresholding Awal (Kasar & Berlubang)
axes[0].imshow(img_biner, cmap="gray")
axes[0].set_title("1. Thresholding Awal (Kasar)")
axes[0].axis("off")  # Menghilangkan angka koordinat/grid pixel

# Plot 2: Hasil Setelah Pembersihan Total
axes[1].imshow(img_perfectly_clean, cmap="gray")
axes[1].set_title("2. Siluet Bersih (Padat)")
axes[1].axis("off")

# Plot 3: Hasil Akhir Skeletonize 1 Pixel
axes[2].imshow(kuda_kurus, cmap="gray")
axes[2].set_title("3. Skeletonization (1 Pixel)")
axes[2].axis("off")

# Mengatur tata letak agar tidak saling bertumpukan
plt.tight_layout()

# Menampilkan window plot ke layar
plt.show()