import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from edge_preprocessing_ablation import PreprocessingAblationDataset, load_binary_dataset
from super_hybrid_benchmarking import extract_super_features

print("Loading dataset...")
X_bin, y = load_binary_dataset()

# Let's test speed of extract_super_features
t0 = time.time()
for i in range(100):
    _ = extract_super_features(X_bin[i])
t1 = time.time()
print(f"100 feature extractions: {t1 - t0:.4f} seconds ({(t1 - t0)/100:.6f} s/image)")

# Let's test speed of Torchvision augmentations
import torchvision.transforms.functional as TF
import random
t0 = time.time()
for i in range(100):
    img_bin = X_bin[i]
    img_tensor = torch.tensor(img_bin, dtype=torch.float32).unsqueeze(0)
    angle = random.uniform(-10.0, 10.0)
    img_tensor = TF.rotate(img_tensor, angle)
    max_dx = int(0.1 * 64)
    max_dy = int(0.1 * 64)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    img_tensor = TF.affine(img_tensor, angle=0, translate=[dx, dy], scale=1.0, shear=0)
    img_bin = (img_tensor.squeeze(0).numpy() > 127).astype(np.uint8) * 255
t1 = time.time()
print(f"100 augmentations: {t1 - t0:.4f} seconds ({(t1 - t0)/100:.6f} s/image)")

# Let's measure next(iter(loader)) on clean CPU
ds = PreprocessingAblationDataset(X_bin[:1000], y[:1000], is_training=True, preprocessing_mode='raw')
loader = DataLoader(ds, batch_size=64, num_workers=0)
t0 = time.time()
batch = next(iter(loader))
t1 = time.time()
print(f"Single batch load: {t1 - t0:.4f} seconds ({(t1 - t0)/64:.6f} s/image)")


