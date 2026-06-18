import os
import numpy as np
import cv2
import scipy.ndimage as ndimage
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
RAW_DIR = "../datasets/raw"
OUTPUT_PATH = "../ocr_evaluation_outputs_super_hybrid/preprocessing_comparison.png"
IMAGE_SIZE = (64, 64)

# Samples to display (Folder Name, File Name, Label)
samples = [
    ("Sample014", "img014-00121.png", "D"),
    ("Sample016", "img016-00052.png", "F"),
    ("Sample001", "img001-00068.png", "0"),
    ("Sample008", "img008-00005.png", "7"),
    ("Sample010", "img010-00025.png", "9")
]

fig, axes = plt.subplots(5, 4, figsize=(10, 15))

for row_idx, (folder_name, img_name, label) in enumerate(samples):
    img_path = os.path.join(RAW_DIR, folder_name, img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join("datasets/raw", folder_name, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join("../datasets/raw", folder_name, img_name)
            
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_color is None:
        print(f"Failed to load image: {img_path}")
        continue
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_rgb_resized = cv2.resize(img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_raw_resized = cv2.resize(img_gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # 2. Polarity & Otsu Thresholding
    top_border = img_raw_resized[0, :]
    bottom_border = img_raw_resized[-1, :]
    left_border = img_raw_resized[:, 0]
    right_border = img_raw_resized[:, -1]
    all_border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border])
    avg_border_intensity = np.mean(all_border_pixels)
    
    if avg_border_intensity > 127:
        _, img_biner = cv2.threshold(img_raw_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, img_biner = cv2.threshold(img_raw_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    # 3. Conditional Hole-Filling (<= 35 px)
    img_bool = img_biner > 0
    all_filled = ndimage.binary_fill_holes(img_bool)
    only_holes = np.logical_xor(all_filled, img_bool)
    labeled_holes, num_features = ndimage.label(only_holes)
    small_holes_mask = np.zeros_like(img_bool)
    
    for slice_index in range(1, num_features + 1):
        hole_area = np.sum(labeled_holes == slice_index)
        if hole_area <= 35:
            small_holes_mask = np.logical_or(small_holes_mask, (labeled_holes == slice_index))
            
    img_conditioned = np.logical_or(img_bool, small_holes_mask)
    img_conditioned_v = (img_conditioned * 255).astype(np.uint8)
    
    # 4. Skeletonization
    img_skeleton = skeletonize(img_conditioned)
    img_skeleton_v = (img_skeleton * 255).astype(np.uint8)
    
    # 5. Morphological Gradient (Kontur)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_grad = cv2.morphologyEx(img_conditioned_v, cv2.MORPH_GRADIENT, kernel)
    
    # Display in row
    # Col 1: Raw
    axes[row_idx, 0].imshow(img_rgb_resized)
    axes[row_idx, 0].axis('off')
    if row_idx == 0:
        axes[row_idx, 0].set_title("1. Raw Image", fontsize=12, pad=10)
    axes[row_idx, 0].text(-15, 32, f"Label: {label}", fontsize=11, fontweight='bold', va='center', ha='right')
    
    # Col 2: Clean Binary
    axes[row_idx, 1].imshow(img_conditioned_v, cmap='gray')
    axes[row_idx, 1].axis('off')
    if row_idx == 0:
        axes[row_idx, 1].set_title("2. Clean Binary", fontsize=12, pad=10)
        
    # Col 3: Skeletonized
    axes[row_idx, 2].imshow(img_skeleton_v, cmap='gray')
    axes[row_idx, 2].axis('off')
    if row_idx == 0:
        axes[row_idx, 2].set_title("3. Skeletonized (1px)", fontsize=12, pad=10)
        
    # Col 4: Morphological Gradient
    axes[row_idx, 3].imshow(img_grad, cmap='gray')
    axes[row_idx, 3].axis('off')
    if row_idx == 0:
        axes[row_idx, 3].set_title("4. Morph. Gradient", fontsize=12, pad=10)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=250, bbox_inches='tight')
plt.close()
print(f"Preprocessing comparison grid saved successfully at {OUTPUT_PATH}")
