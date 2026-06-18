"""
=======================================================================
PHASE 5B: FEATURE MAP VISUALIZATION ANALYSIS
=======================================================================
This script extracts and visualizes intermediate feature maps from the
Conv1, Conv2, and Conv3 blocks of the SuperHybridCNN model.
It visually demonstrates how morphological gradient contours propagate
through pooling layers.
=======================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from super_hybrid_benchmarking import (
    SuperHybridCNN, load_binary_dataset, NUM_CLASSES
)

def get_feature_maps(model, x_img, x_feats):
    """
    Manually run the forward pass up to the Conv blocks to extract activation maps
    """
    model.eval()
    activations = {}
    
    with torch.no_grad():
        # Block 1
        x1 = model.conv1(x_img)
        x1_bn = model.bn1(x1)
        x1_relu = model.relu(x1_bn)
        x1_pool = model.pool1(x1_relu)
        activations["conv1"] = x1_pool
        
        # Block 2
        x2 = model.conv2(x1_pool)
        x2_bn = model.bn2(x2)
        x2_relu = model.relu(x2_bn)
        x2_pool = model.pool2(x2_relu)
        activations["conv2"] = x2_pool
        
        # Block 3
        x3 = model.conv3(x2_pool)
        x3_bn = model.bn3(x3)
        x3_relu = model.relu(x3_bn)
        x3_pool = model.pool3(x3_relu)
        activations["conv3"] = x3_pool
        
    return activations

def plot_activation_grid(activations, output_path):
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    
    blocks = ["conv1", "conv2", "conv3"]
    for row_idx, block in enumerate(blocks):
        act_map = activations[block].cpu().numpy()[0] # [C, H, W]
        num_channels = act_map.shape[0]
        
        for col_idx in range(8):
            ax = axes[row_idx, col_idx]
            if col_idx < num_channels:
                # Average/normalize map for rendering
                channel_map = act_map[col_idx]
                ax.imshow(channel_map, cmap="viridis")
            ax.axis("off")
            if col_idx == 0:
                ax.text(-10, act_map.shape[1]/2, f"{block}\n({act_map.shape[1]}x{act_map.shape[2]})", 
                        fontsize=12, fontweight="bold", ha="right", va="center")
                
    plt.suptitle("SuperHybridCNN Feature Map Activation Grid", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

def main():
    print("=" * 75)
    print("  PHASE 5B: FEATURE MAP VISUALIZATION ANALYSIS")
    print("=" * 75)
    
    output_dir = "ocr_evaluation_outputs_super_hybrid"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample image
    try:
        X_bin, y = load_binary_dataset()
        sample_img = X_bin[0]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic character image...")
        sample_img = np.zeros((64, 64), dtype=np.uint8)
        # Draw a synthetic character 'A'
        cv2.putText(sample_img, "A", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 6)
        
    # Preprocess with Morphological Gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gradient_img = cv2.morphologyEx(sample_img, cv2.MORPH_GRADIENT, kernel)
    
    # Save inputs for side-by-side comparison in the paper
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(sample_img, cmap="gray")
    axes[0].set_title("Input Raw (Biner)", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(gradient_img, cmap="gray")
    axes[1].set_title("Input Gradien Morfologi", fontsize=10, fontweight="bold")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_map_input_comparison.png"), dpi=150)
    plt.close()
    
    # Run through Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperHybridCNN(num_classes=NUM_CLASSES, feat_dim=12).to(device)
    
    # Load trained weights if available
    weights_path = os.path.join(output_dir, "SuperHybrid_Gradient.pth")
    if os.path.exists(weights_path):
        print(f"Loading trained weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Trained weights not found. Running with initialized weights.")
        
    # Prepare batch
    norm_img = (gradient_img.astype(np.float32) / 255.0 - 0.5) / 0.5
    img_tensor = torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, 64, 64]
    feats_tensor = torch.zeros((1, 12)).to(device) # Dummy geometric features
    
    # Extract feature maps
    activations = get_feature_maps(model, img_tensor, feats_tensor)
    
    # Plot grids
    grid_path = os.path.join(output_dir, "feature_map_visualization.png")
    plot_activation_grid(activations, grid_path)
    
    print(f"\n[OK] Feature map visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()
