import os
import cv2
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def generate_figure():
    raw_dir = "datasets/raw"
    if not os.path.exists(raw_dir):
        alt_path = r"C:\Users\Unpad-hci\Documents\Untitled-Hybrid-Skeletonization-New-Lightweight-OCR-Model\datasets\raw"
        if os.path.exists(alt_path):
            raw_dir = alt_path
        else:
            raise FileNotFoundError(f"Raw dataset folder not found at {raw_dir} or {alt_path}")
            
    sample_folder = os.path.join(raw_dir, "Sample012") # 'B' or 'A'
    if not os.path.exists(sample_folder):
        # find first available sample
        samples = [d for d in os.listdir(raw_dir) if d.startswith("Sample")]
        if not samples:
            raise FileNotFoundError(f"No sample folders found in {raw_dir}")
        sample_folder = os.path.join(raw_dir, samples[10]) # something not 0
        
    img_name = [f for f in os.listdir(sample_folder) if f.endswith('.png')][0]
    raw_path = os.path.join(sample_folder, img_name)
    
    img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    
    # 1. Raw Image
    raw_img = img_resized.copy()
    
    # 2. Otsu Threshold (Clean Binary basis)
    top_border = img_resized[0, :]
    bottom_border = img_resized[-1, :]
    left_border = img_resized[:, 0]
    right_border = img_resized[:, -1]
    avg_border = np.mean(np.concatenate([top_border, bottom_border, left_border, right_border]))
    
    if avg_border > 127:
        _, img_bin = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, img_bin = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    # 3. Hole-Filling
    img_bool = img_bin > 0
    all_filled = ndimage.binary_fill_holes(img_bool)
    only_holes = np.logical_xor(all_filled, img_bool)
    labeled_holes, num_features = ndimage.label(only_holes)
    small_holes_mask = np.zeros_like(img_bool)
    
    for slice_index in range(1, num_features + 1):
        hole_area = np.sum(labeled_holes == slice_index)
        if hole_area <= 35:
            small_holes_mask = np.logical_or(small_holes_mask, (labeled_holes == slice_index))
            
    img_clean_bin = np.logical_or(img_bool, small_holes_mask).astype(np.uint8) * 255
    
    # 4. Morphological Gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_grad = cv2.morphologyEx(img_clean_bin, cv2.MORPH_GRADIENT, kernel)
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    
    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title('1. Raw Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_bin, cmap='gray')
    axes[1].set_title('2. Clean Binary (Otsu)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(img_clean_bin, cmap='gray')
    axes[2].set_title('3. Hole-Filling', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(img_grad, cmap='gray')
    axes[3].set_title('4. Morphological Gradient', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    out_dir = "ocr_evaluation_outputs_super_hybrid"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "preprocessing_comparison.png"), dpi=180, bbox_inches="tight")
    print(f"Figure saved to {os.path.join(out_dir, 'preprocessing_comparison.png')}")

if __name__ == "__main__":
    generate_figure()
