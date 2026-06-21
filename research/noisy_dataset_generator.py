# research/noisy_dataset_generator.py
import cv2
import numpy as np

def add_salt_and_pepper_noise(img, density=0.05):
    """
    Suntik derau Salt-and-Pepper pada gambar (numpy array [H, W]).
    """
    noisy_img = img.copy()
    # Salt
    num_salt = np.ceil(density * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[tuple(coords)] = 255
    # Pepper
    num_pepper = np.ceil(density * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[tuple(coords)] = 0
    return noisy_img

def add_gaussian_blur(img, sigma=1.0):
    """
    Terapkan Gaussian Blur pada gambar (numpy array [H, W]).
    """
    # Kernel size must be odd
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)
