import os
import argparse
import cv2
import numpy as np
import torch
from super_hybrid_benchmarking import (
    TopoGradNet,
    preprocess_image,
    extract_super_features,
    CLASS_LIST,
)

def run_inference(image_path, model_path="ocr_evaluation_outputs_super_hybrid/TopoGrad-Net.pth"):
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Check if model weights exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
        
    # 3. Load model
    model = TopoGradNet(num_classes=62, feat_dim=12)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 4. Read raw image
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        raise FileNotFoundError(f"Cannot read image at: {image_path}")
        
    # 5. Preprocess (Binarization + Hole Filling)
    img_bin, _ = preprocess_image(image_path)
    
    # 6. Morphological Gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_grad = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel)
    
    # 7. Extract topology features
    feats = extract_super_features(img_bin)
    
    # 8. Normalize and convert to tensors
    img_norm = (img_grad.astype(np.float32) / 255.0 - 0.5) / 0.5
    img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).to(device)
    feats_tensor = torch.tensor(feats).unsqueeze(0).to(device)
    
    # 9. Predict
    with torch.no_grad():
        output = model(img_tensor, feats_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
    karakter = CLASS_LIST[predicted.item()]
    return {
        "prediction": karakter,
        "confidence": confidence.item(),
        "raw_image": raw_img,
        "binary_image": img_bin,
        "gradient_image": img_grad,
        "features": feats
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for TopoGrad-Net")
    parser.add_argument("--image", type=str, required=True, help="Path to input character image")
    parser.add_argument("--model", type=str, default="ocr_evaluation_outputs_super_hybrid/TopoGrad-Net.pth", help="Path to model weights")
    args = parser.parse_args()
    
    try:
        res = run_inference(args.image, args.model)
        print("\n=========================================")
        print("          TOPOGRAD-NET INFERENCE         ")
        print("=========================================")
        print(f"Input Image : {args.image}")
        print(f"Prediction  : '{res['prediction']}'")
        print(f"Confidence  : {res['confidence'] * 100:.2f}%")
        print("=========================================")
    except Exception as e:
        print(f"Error: {e}")
