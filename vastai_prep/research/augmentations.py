"""
Skeleton-aware data augmentations for S1 experiment.
Includes micro elastic deformation, skeleton dilation, and endpoint noise.
"""

import random
import cv2
import numpy as np
import scipy.ndimage as ndimage

def elastic_transform(image, alpha=0.5, sigma=0.5):
    """
    Random elastic deformation with small magnitude (sigma < 1px).
    Input `image` is a 2D numpy array of shape (32, 32) with values in [0.0, 1.0].
    """
    shape = image.shape
    # Generate random displacement fields
    dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0.0) * alpha
    dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0.0) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    distorted = ndimage.map_coordinates(image, indices, order=1, mode='constant', cval=0.0).reshape(shape)
    # Threshold at 0.3 to keep lines crisp and binary-like
    return (distorted > 0.3).astype(np.float32)

def get_endpoints(bin_img):
    """Detect endpoint pixels in a binary skeleton image."""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_sum = cv2.filter2D(bin_img.astype(np.uint8), cv2.CV_16U, kernel, borderType=cv2.BORDER_CONSTANT)
    neighbor_map = neighbor_sum * bin_img
    return (neighbor_map == 1)

def prune_endpoints(bin_img, pixels_to_remove=1):
    """Shorten the skeleton by removing pixels at the endpoints."""
    img = bin_img.copy()
    for _ in range(pixels_to_remove):
        endpoints = get_endpoints(img)
        # Ensure we don't completely wipe out the skeleton (keep at least 5 pixels)
        if np.sum(img) - np.sum(endpoints) > 5:
            img[endpoints] = 0
        else:
            break
    return img

def extend_endpoints(bin_img, pixels_to_add=1):
    """Lengthen the skeleton lines by extending from endpoints in their current direction."""
    img = bin_img.copy()
    h, w = img.shape
    for _ in range(pixels_to_add):
        endpoints = get_endpoints(img)
        endpoints_y, endpoints_x = np.where(endpoints)
        new_pixels = []
        for ey, ex in zip(endpoints_y, endpoints_x):
            # Find the single neighbor pixel to determine direction
            neighbor_found = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = ey + dy, ex + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if img[ny, nx] == 1:
                            # Extension pixel is in the opposite direction of the neighbor
                            ey_new, ex_new = ey - dy, ex - dx
                            if 0 <= ey_new < h and 0 <= ex_new < w:
                                new_pixels.append((ey_new, ex_new))
                                neighbor_found = True
                                break
                if neighbor_found:
                    break
        for ny, nx in new_pixels:
            img[ny, nx] = 1
    return img

def random_endpoint_noise(image):
    """Randomly add or remove 1-2 pixels at the endpoints of the skeleton."""
    bin_img = (image > 0.5).astype(np.uint8)
    action = np.random.choice(["none", "add1", "add2", "remove1", "remove2"])
    
    if action == "add1":
        res = extend_endpoints(bin_img, 1)
    elif action == "add2":
        res = extend_endpoints(bin_img, 2)
    elif action == "remove1":
        res = prune_endpoints(bin_img, 1)
    elif action == "remove2":
        res = prune_endpoints(bin_img, 2)
    else:
        res = bin_img
        
    return res.astype(np.float32)

def skeleton_dilation(image):
    """
    Randomly dilate the skeleton image by 1-2 pixels to simulate stroke width variations,
    or leave as a 1-pixel line (0px dilation).
    """
    r = np.random.choice([0, 1, 2])
    if r == 0:
        return image
    
    img_uint8 = (image * 255).astype(np.uint8)
    ksize = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(img_uint8, kernel, iterations=1)
    
    return dilated.astype(np.float32) / 255.0

def apply_skeleton_augmentations(image):
    """Apply the three skeleton-aware augmentations sequentially."""
    # 1. Random micro-elastic deformation
    img = elastic_transform(image, alpha=np.random.uniform(0.2, 0.6), sigma=np.random.uniform(0.4, 0.8))
    # 2. Random endpoint noise (add/remove pixels)
    img = random_endpoint_noise(img)
    # 3. Random dilation
    img = skeleton_dilation(img)
    return img
