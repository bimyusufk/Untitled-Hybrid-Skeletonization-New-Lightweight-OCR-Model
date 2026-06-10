"""
Data loader untuk Chars74K dan EMNIST datasets.
Mendukung loading raw images, on-the-fly skeletonization, dan cross-dataset.
"""

import os
import logging

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .preprocessing import (preprocess_single, preprocess_to_model_input,
                             preprocess_raw_to_model_input)

logger = logging.getLogger(__name__)

# Label mapping: EMNIST ByClass index → character
_EMNIST_INDEX_TO_CHAR = (
    [str(i) for i in range(10)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
)


# =====================================================================
# CHARS74K LOADING
# =====================================================================

def _iter_chars74k_images(csv_path, base_dir):
    """Yield (img_gray, label_str) dari Chars74K."""
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        folder_name = row["Folder Name"]
        label = str(row["Label"])
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for img_name in sorted(os.listdir(folder_path)):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                yield img, label


def load_chars74k_raw(config):
    """
    Load Chars74K raw images, resize ke model_input_size, normalisasi.

    Returns: X (N,H,W,1), y_encoded, label_encoder
    """
    ds = config["datasets"]["chars74k"]
    size = tuple(ds["model_input_size"])

    X, labels = [], []
    for img, label in tqdm(_iter_chars74k_images(ds["csv_path"], ds["raw_dir"]),
                           desc="Loading Chars74K raw"):
        x = preprocess_raw_to_model_input(img, size)
        X.append(x)
        labels.append(label)

    X = np.expand_dims(np.array(X, dtype=np.float32), axis=-1)
    le = LabelEncoder()
    y = le.fit_transform(np.array(labels))
    logger.info(f"Chars74K raw loaded: {X.shape[0]} samples, {len(le.classes_)} classes")
    return X, y, le


def load_chars74k_skeleton(config, threshold=None):
    """
    Load Chars74K images dan apply preprocessing+skeletonization on-the-fly.

    Parameters
    ----------
    threshold : int or None
        Hole-filling threshold. None → gunakan config default.

    Returns: X (N,H,W,1), y_encoded, label_encoder
    """
    ds = config["datasets"]["chars74k"]
    pp_size = tuple(ds["preprocessing_size"])
    m_size = tuple(ds["model_input_size"])
    if threshold is None:
        threshold = config["preprocessing"]["default_threshold"]

    X, labels = [], []
    for img, label in tqdm(
        _iter_chars74k_images(ds["csv_path"], ds["raw_dir"]),
        desc=f"Preprocessing skeleton (thr={threshold})",
    ):
        skel, _ = preprocess_single(img, threshold, pp_size)
        x = preprocess_to_model_input(skel, m_size)
        X.append(x)
        labels.append(label)

    X = np.expand_dims(np.array(X, dtype=np.float32), axis=-1)
    le = LabelEncoder()
    y = le.fit_transform(np.array(labels))
    logger.info(f"Chars74K skeleton (thr={threshold}): {X.shape[0]} samples")
    return X, y, le


def load_chars74k_existing_skeleton(config):
    """
    Load skeleton images yang sudah tersimpan di disk (hasil preprocessing sebelumnya).
    Lebih cepat dari on-the-fly preprocessing.
    """
    ds = config["datasets"]["chars74k"]
    skel_dir = ds.get("skeleton_dir", "datasets/skeletonize")
    m_size = tuple(ds["model_input_size"])

    X, labels = [], []
    for img, label in tqdm(
        _iter_chars74k_images(ds["csv_path"], skel_dir),
        desc="Loading existing skeletons",
    ):
        x = preprocess_raw_to_model_input(img, m_size)
        X.append(x)
        labels.append(label)

    X = np.expand_dims(np.array(X, dtype=np.float32), axis=-1)
    le = LabelEncoder()
    y = le.fit_transform(np.array(labels))
    logger.info(f"Chars74K existing skeletons: {X.shape[0]} samples")
    return X, y, le


# =====================================================================
# EMNIST LOADING
# =====================================================================

def load_emnist(config):
    """
    Download dan load EMNIST ByClass test set, mapped ke Chars74K labels.

    Returns: X_raw (N,H,W,1), X_skeleton (N,H,W,1), y_encoded, label_encoder
    """
    import tensorflow_datasets as tfds

    ds_cfg = config["datasets"]["emnist"]
    m_size = tuple(config["datasets"]["chars74k"]["model_input_size"])
    pp_size = tuple(config["datasets"]["chars74k"]["preprocessing_size"])
    threshold = config["preprocessing"]["default_threshold"]
    max_samples = ds_cfg.get("max_test_samples", 3000)

    logger.info(f"Downloading EMNIST {ds_cfg['subset']}...")

    ds = tfds.load(
        f"emnist/{ds_cfg['subset']}",
        split="test",
        as_supervised=True,
        data_dir=ds_cfg.get("download_dir", "datasets/emnist"),
    )

    # Stratified sampling: ambil max_samples secara merata per kelas
    per_class = max_samples // 62
    class_counts = {i: 0 for i in range(62)}

    raw_list, skel_list, label_list = [], [], []

    for image, label_idx in tqdm(tfds.as_numpy(ds), desc="Processing EMNIST"):
        li = int(label_idx)
        if class_counts[li] >= per_class:
            continue
        class_counts[li] += 1

        # EMNIST images perlu di-transpose (known issue)
        img = np.squeeze(image)
        img = np.transpose(img)
        img = np.flip(img, axis=1)
        img = img.astype(np.uint8)

        # Raw version
        raw = preprocess_raw_to_model_input(img, m_size)
        raw_list.append(raw)

        # Skeleton version
        skel, _ = preprocess_single(img, threshold, pp_size)
        skel_m = preprocess_to_model_input(skel, m_size)
        skel_list.append(skel_m)

        # Label → karakter → encoded
        label_list.append(_EMNIST_INDEX_TO_CHAR[li])

        if sum(class_counts.values()) >= max_samples:
            break

    X_raw = np.expand_dims(np.array(raw_list, dtype=np.float32), axis=-1)
    X_skel = np.expand_dims(np.array(skel_list, dtype=np.float32), axis=-1)

    le = LabelEncoder()
    le.fit(_EMNIST_INDEX_TO_CHAR)  # Fit pada semua 62 kelas
    y = le.transform(np.array(label_list))

    logger.info(f"EMNIST loaded: {X_raw.shape[0]} samples (max {per_class}/class)")
    return X_raw, X_skel, y, le


# =====================================================================
# UTILITIES
# =====================================================================

def split_dataset(X, y, test_split=0.2, random_state=42):
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_split,
        random_state=random_state, stratify=y,
    )


def get_class_distribution(y_encoded, label_encoder):
    """Hitung distribusi kelas → DataFrame."""
    classes, counts = np.unique(y_encoded, return_counts=True)
    records = []
    for cls_idx, count in zip(classes, counts):
        char = label_encoder.inverse_transform([cls_idx])[0]
        cat = "digit" if char.isdigit() else ("uppercase" if char.isupper() else "lowercase")
        records.append({"class": char, "category": cat, "count": int(count)})
    return pd.DataFrame(records)


def get_label_encoder_chars74k():
    """Buat LabelEncoder standar untuk 62 kelas alphanumeric."""
    le = LabelEncoder()
    all_labels = (
        [str(i) for i in range(10)]
        + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    )
    le.fit(all_labels)
    return le
