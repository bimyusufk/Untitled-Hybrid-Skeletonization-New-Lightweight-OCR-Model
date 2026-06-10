"""
Preprocessing pipeline: Polarity Detection → Otsu → Conditional Hole-Filling → Skeletonization.
Threshold diparameterisasi untuk analisis sensitivitas.
"""

import cv2
import numpy as np
import scipy.ndimage as ndimage
from skimage.morphology import skeletonize
from skimage.measure import euler_number
import logging

logger = logging.getLogger(__name__)


# =====================================================================
# CORE PREPROCESSING
# =====================================================================

def preprocess_single(img_gray, threshold, preprocess_size=(64, 64)):
    """
    Full preprocessing pipeline untuk satu gambar grayscale.

    Parameters
    ----------
    img_gray : np.ndarray
        Gambar grayscale (uint8).
    threshold : int
        Ukuran maksimum lubang yang akan diisi (piksel).
        0 = tanpa hole-filling.
    preprocess_size : tuple
        Ukuran resize sebelum preprocessing.

    Returns
    -------
    skeleton : np.ndarray float32, shape=preprocess_size, values {0.0, 1.0}
    binarized : np.ndarray uint8, shape=preprocess_size, values {0, 255}
    """
    img_resized = cv2.resize(img_gray, preprocess_size, interpolation=cv2.INTER_AREA)

    # --- Polarity detection ---
    top = img_resized[0, :]
    bottom = img_resized[-1, :]
    left = img_resized[:, 0]
    right = img_resized[:, -1]
    avg_border = np.mean(np.concatenate([top, bottom, left, right]))

    if avg_border > 127:
        _, img_bin = cv2.threshold(img_resized, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, img_bin = cv2.threshold(img_resized, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_bool = img_bin > 0

    # --- Conditional hole-filling ---
    if threshold > 0:
        all_filled = ndimage.binary_fill_holes(img_bool)
        only_holes = np.logical_xor(all_filled, img_bool)
        labeled_holes, num_features = ndimage.label(only_holes)

        small_holes_mask = np.zeros_like(img_bool)
        for i in range(1, num_features + 1):
            if np.sum(labeled_holes == i) <= threshold:
                small_holes_mask = np.logical_or(small_holes_mask,
                                                  labeled_holes == i)
        img_bool = np.logical_or(img_bool, small_holes_mask)

    # --- Skeletonization ---
    img_skeleton = skeletonize(img_bool)

    return img_skeleton.astype(np.float32), img_bin


def preprocess_to_model_input(skeleton, model_input_size=(32, 32)):
    """Resize skeleton ke ukuran input model dan normalisasi."""
    skel_uint8 = (skeleton * 255).astype(np.uint8)
    resized = cv2.resize(skel_uint8, model_input_size,
                         interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def preprocess_raw_to_model_input(img_gray, model_input_size=(32, 32)):
    """Resize raw grayscale image ke ukuran input model dan normalisasi."""
    resized = cv2.resize(img_gray, model_input_size,
                         interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# =====================================================================
# SKELETON QUALITY METRICS
# =====================================================================

def compute_skeleton_quality(skeleton_bool):
    """
    Hitung metrik kualitas skeleton untuk satu gambar.

    Parameters
    ----------
    skeleton_bool : np.ndarray, dtype bool atau uint8 {0,1}

    Returns
    -------
    dict dengan kunci:
        total_pixels, endpoints, junctions, loops, connectivity_score
    """
    bin_img = skeleton_bool.astype(np.uint8)
    total_pixels = int(np.sum(bin_img))

    if total_pixels == 0:
        return {
            "total_pixels": 0, "endpoints": 0, "junctions": 0,
            "loops": 0, "connectivity_score": 0.0,
            "branch_point_ratio": 0.0,
        }

    # Neighbor count menggunakan filter2D cepat
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_sum = cv2.filter2D(bin_img, cv2.CV_16U, kernel,
                                borderType=cv2.BORDER_CONSTANT)
    neighbor_map = neighbor_sum * bin_img

    endpoints = int(np.sum(neighbor_map == 1))
    junctions = int(np.sum(neighbor_map >= 3))

    # Loops via Euler number
    e_num = euler_number(bin_img, connectivity=2)
    loops = max(0, 1 - e_num)

    # Connectivity: apakah skeleton 1 komponen terhubung?
    labeled, num_components = ndimage.label(bin_img)
    connectivity_score = 1.0 / num_components if num_components > 0 else 0.0

    branch_point_ratio = junctions / total_pixels if total_pixels > 0 else 0.0

    return {
        "total_pixels": total_pixels,
        "endpoints": endpoints,
        "junctions": junctions,
        "loops": int(loops),
        "connectivity_score": round(connectivity_score, 4),
        "branch_point_ratio": round(branch_point_ratio, 4),
        "num_components": int(num_components),
    }
