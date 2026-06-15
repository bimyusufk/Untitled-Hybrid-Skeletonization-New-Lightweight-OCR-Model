import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from research.preprocessing import preprocess_single, preprocess_to_model_input

# =====================================================================
# CONFIGURATION
# =====================================================================
FONT_DIR = "C:\\Windows\\Fonts"
OUTPUT_NPZ = "datasets/synthetic_dataset_32x32.npz"
SAMPLES_PER_CLASS = 1700
PREPROCESS_SIZE = (64, 64)
MODEL_INPUT_SIZE = (32, 32)
THRESHOLD_HOLE_FILLING = 35

# Alphanumeric character list (62 classes)
DIGITS = [str(i) for i in range(10)]
UPPERCASE = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
LOWERCASE = [chr(c) for c in range(ord('a'), ord('z') + 1)]
CLASSES = DIGITS + UPPERCASE + LOWERCASE

random.seed(42)
np.random.seed(42)

# =====================================================================
# SCAN FOR SYSTEM FONTS
# =====================================================================
def get_available_fonts():
    print(f"Scanning for TTF/OTF fonts in {FONT_DIR}...")
    fonts = []
    if not os.path.exists(FONT_DIR):
        print(f"Font directory {FONT_DIR} not found. Fallback to default...")
        return fonts
        
    for f in os.listdir(FONT_DIR):
        if f.lower().endswith(('.ttf', '.otf')):
            fonts.append(os.path.join(FONT_DIR, f))
            
    print(f"Found {len(fonts)} available fonts.")
    return fonts

def get_text_size(draw, text, font):
    """Robust helper to get text bounding box width and height across Pillow versions"""
    if hasattr(draw, 'textbbox'):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    if hasattr(draw, 'textsize'):
        return draw.textsize(text, font=font)
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top

# =====================================================================
# DATA GENERATION
# =====================================================================
def generate_dataset():
    available_fonts = get_available_fonts()
    if not available_fonts:
        raise RuntimeError("No system fonts found. Cannot generate synthetic data.")
        
    # We will compile results here
    X_list = []
    y_list = []
    
    total_samples = len(CLASSES) * SAMPLES_PER_CLASS
    print(f"Generating {total_samples} synthetic skeleton samples...")
    
    pbar = tqdm(total=total_samples, desc="Rendering Fonts")
    
    for cls_idx, char_str in enumerate(CLASSES):
        success_count = 0
        attempts = 0
        max_attempts = SAMPLES_PER_CLASS * 10
        
        while success_count < SAMPLES_PER_CLASS and attempts < max_attempts:
            attempts += 1
            font_path = random.choice(available_fonts)
            font_size = random.randint(24, 44)
            
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                continue # Skip corrupt font files
                
            # Create a blank image
            img = Image.new('L', PREPROCESS_SIZE, 0)
            draw = ImageDraw.Draw(img)
            
            try:
                w, h = get_text_size(draw, char_str, font)
            except Exception:
                continue
                
            if w <= 0 or h <= 0 or w >= 60 or h >= 60:
                continue # Skip empty, corrupt, or oversized glyphs
                
            # Center coordinates with random translations/offsets
            x_pos = (PREPROCESS_SIZE[0] - w) // 2 + random.randint(-4, 4)
            y_pos = (PREPROCESS_SIZE[1] - h) // 2 + random.randint(-4, 4)
            
            # Draw character
            draw.text((x_pos, y_pos), char_str, fill=255, font=font)
            
            # Apply random rotation
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
            
            # Convert to numpy array and binarize
            img_np = np.array(img, dtype=np.uint8)
            
            # Binarize to remove antialiasing/interpolation gray pixels
            _, img_bin = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
            
            # If the image is completely empty, skip
            if np.sum(img_bin) == 0:
                continue
                
            # Apply same skeletonization pipeline
            try:
                skeleton, _ = preprocess_single(img_bin, threshold=THRESHOLD_HOLE_FILLING, preprocess_size=PREPROCESS_SIZE)
                skel_input = preprocess_to_model_input(skeleton, MODEL_INPUT_SIZE)
                
                # Convert back to uint8 [0, 255] to compress npz storage size
                skel_uint8 = (skel_input * 255).astype(np.uint8)
                
                X_list.append(skel_uint8)
                y_list.append(char_str)
                success_count += 1
                pbar.update(1)
            except Exception as e:
                # Occasional failure in skeletonization or edge cases
                continue
                
        if success_count < SAMPLES_PER_CLASS:
            print(f"\n[Warning] Only generated {success_count}/{SAMPLES_PER_CLASS} for class '{char_str}' after {attempts} attempts.")
            
    pbar.close()
    
    # Save to disk
    X = np.expand_dims(np.array(X_list, dtype=np.uint8), axis=-1)
    y = np.array(y_list)
    
    print(f"Generated dataset shapes: X={X.shape}, y={y.shape}")
    print(f"Saving dataset to {OUTPUT_NPZ}...")
    np.savez_compressed(OUTPUT_NPZ, X=X, y=y)
    print("Dataset saved successfully!")

if __name__ == "__main__":
    generate_dataset()
