import os
import yaml
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from research.data_loader import get_label_encoder_chars74k
from ocr_evaluation import save_ocr_evaluation_artifacts

# Load configuration
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

SEED = CONFIG["project"]["random_seed"]
IMAGE_SIZE = (64, 64)
CSV_PATH = "datasets/annotations.csv"
SKELETON_BASE_DIR = "datasets/skeletonize"

LABEL_ENCODER = get_label_encoder_chars74k()
NUM_CLASSES = len(LABEL_ENCODER.classes_)

def load_skeleton_dataset_64x64():
    print(f"Loading skeletonized Chars74K dataset at {IMAGE_SIZE}...")
    df = pd.read_csv(CSV_PATH)
    X_data = []
    y_labels = []

    for index, row in df.iterrows():
        folder_name = row['Folder Name']
        label = row['Label']
        folder_path = os.path.join(SKELETON_BASE_DIR, folder_name)
        if not os.path.exists(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if img.shape[:2] != IMAGE_SIZE:
                        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                    X_data.append(img.astype(np.float32) / 255.0)
                    y_labels.append(str(label))

    X = np.expand_dims(np.array(X_data), axis=-1)
    y = np.array(y_labels)
    y_encoded = LABEL_ENCODER.transform(y)
    print(f"Loaded {X.shape[0]} samples.")
    return X, y_encoded

class DummyHistory:
    def __init__(self):
        self.history = {}

def main():
    # Load dataset
    X, y = load_skeleton_dataset_64x64()
    
    # Split to get the exact same test split used during training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    model_path = "research_outputs/models/distilled_student_64x64.keras"
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Rerunning evaluation and saving artifacts...")
    eval_res = save_ocr_evaluation_artifacts(
        history=DummyHistory(),
        X_test=X_test,
        y_test=y_test,
        label_encoder=LABEL_ENCODER,
        model=model,
        output_dir="ocr_evaluation_outputs_s1s8s3se_s2_s6",
        model_key="hybrid_s1_s8_s3_se_s2_s6",
        model_name="Hybrid Skeleton S1+S8+S3+SE+S2+S6 (Distilled)",
        batch_size=64
    )
    
    metrics = eval_res["metrics"]
    print("\n=== EVALUATION RESULTS ===")
    print(f"Strict Accuracy: {metrics['strict_accuracy']:.2f}%")
    print(f"Tolerant Accuracy: {metrics['tolerant_accuracy']:.2f}%")
    print(f"Average Inference Latency: {metrics['avg_inference_time_ms']:.4f} ms/image")
    print(f"Artifacts successfully saved to: ocr_evaluation_outputs_s1s8s3se_s2_s6/")

if __name__ == "__main__":
    main()
