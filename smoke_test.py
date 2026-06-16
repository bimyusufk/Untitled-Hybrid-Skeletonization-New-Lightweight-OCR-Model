"""
Smoke test: jalankan pipeline dengan subset kecil untuk validasi end-to-end.
Menggunakan config override: epochs=2, max_samples=100, limited experiments.
"""
import os
import sys
import yaml
import copy

# Load and override config for smoke test
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Minimal overrides
CONFIG["project"]["output_dir"] = "smoke_test_output"
CONFIG["training"]["epochs"] = 2
CONFIG["training"]["batch_size"] = 16
CONFIG["experiments"]["controlled_experiment"]["enabled"] = False
CONFIG["experiments"]["threshold_sensitivity"]["enabled"] = False
CONFIG["experiments"]["cross_dataset"]["enabled"] = False
CONFIG["experiments"]["latency_benchmark"]["warmup_iterations"] = 5
CONFIG["experiments"]["latency_benchmark"]["benchmark_iterations"] = 20
CONFIG["experiments"]["skeleton_quality"]["enabled"] = False

# Only enable a few baselines for speed
CONFIG["experiments"]["baselines"]["models"]["mobilenetv2"]["enabled"] = False
CONFIG["experiments"]["baselines"]["models"]["hog_svm"]["enabled"] = False

# GPU setup
import ctypes
import sys
if sys.platform.startswith("linux"):
    try:
        for path in sys.path:
            possible_path = os.path.join(path, "nvidia", "cusolver", "lib", "libcusolver.so.11")
            if os.path.exists(possible_path):
                ctypes.CDLL(possible_path)
                print(f"[GPU] Preloaded libcusolver: {possible_path}")
                break
    except Exception as e:
        print(f"[GPU] Warning: Failed to preload libcusolver: {e}")

if not CONFIG["hardware"]["use_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
if CONFIG["hardware"]["use_gpu"]:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[GPU] {len(gpus)} GPU(s)")

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test")

from research.data_loader import load_chars74k_raw, load_chars74k_skeleton, split_dataset, get_class_distribution
from research.models import build_standard_cnn, build_hybrid_cnn, build_iso_parameter_cnn, compile_keras_model, count_params
from research.evaluation import evaluate_keras_model, benchmark_latency, save_training_curves
from research.report_generator import generate_report

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
OUTPUT_DIR = CONFIG["project"]["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

ALL_RESULTS = {}

# --- Test 1: Load raw data ---
logger.info("=== TEST 1: Load Chars74K Raw ===")
X_raw, y_raw, le = load_chars74k_raw(CONFIG)
logger.info(f"Loaded: {X_raw.shape}, classes: {len(le.classes_)}")
assert X_raw.shape[0] > 0, "No data loaded!"
assert len(le.classes_) == 62, f"Expected 62 classes, got {len(le.classes_)}"

# Class distribution
dist = get_class_distribution(y_raw, le)
logger.info(f"Distribution:\n{dist.head(10)}")

X_tr, X_te, y_tr, y_te = split_dataset(X_raw, y_raw, 0.2, SEED)

# --- Test 2: Train Iso-Param CNN (minimal) ---
logger.info("=== TEST 2: Train Iso-Param CNN (2 epochs) ===")
m = build_iso_parameter_cnn(62)
m = compile_keras_model(m, CONFIG)
logger.info(f"Model params: {count_params(m):,}")
h = m.fit(X_tr, y_tr, epochs=2, batch_size=16, validation_data=(X_te, y_te), verbose=2)
metrics = evaluate_keras_model(m, X_te, y_te, le, batch_size=16)
logger.info(f"Iso-Param: strict={metrics['strict_accuracy']:.2f}%, "
            f"F1={metrics['macro_f1']:.4f}")

# --- Test 3: Load skeleton data ---
logger.info("=== TEST 3: Load Chars74K Skeleton (thr=35) ===")
X_skel, y_skel, le_s = load_chars74k_skeleton(CONFIG, threshold=35)
logger.info(f"Skeleton loaded: {X_skel.shape}")
assert X_skel.shape[0] > 0

Xs_tr, Xs_te, ys_tr, ys_te = split_dataset(X_skel, y_skel, 0.2, SEED)

# --- Test 4: Train Hybrid CNN (minimal) ---
logger.info("=== TEST 4: Train Hybrid CNN (2 epochs) ===")
aug_cfg = CONFIG["training"]["augmentation"]
m_h = build_hybrid_cnn(62, aug_config=aug_cfg)
m_h = compile_keras_model(m_h, CONFIG)
logger.info(f"Hybrid params: {count_params(m_h):,}")
h_h = m_h.fit(Xs_tr, ys_tr, epochs=2, batch_size=16,
              validation_data=(Xs_te, ys_te), verbose=2)
metrics_h = evaluate_keras_model(m_h, Xs_te, ys_te, le_s, batch_size=16)
logger.info(f"Hybrid: strict={metrics_h['strict_accuracy']:.2f}%, "
            f"F1={metrics_h['macro_f1']:.4f}, "
            f"cluster={metrics_h['cluster_f1']}")

# --- Test 5: Latency Benchmark (minimal) ---
logger.info("=== TEST 5: Latency Benchmark (20 runs) ===")
lat = benchmark_latency(m_h, X_skel[:1], n_warmup=5, n_runs=20)
logger.info(f"Latency: mean={lat['mean_ms']:.3f}ms ± {lat['std_ms']:.3f}")

# --- Test 6: Report Generation ---
logger.info("=== TEST 6: Report Generation ===")
ALL_RESULTS["dataset_stats"] = {
    "total_samples": X_raw.shape[0],
    "num_classes": 62,
    "train_size": len(y_tr),
    "test_size": len(y_te),
    "distribution": {
        "digit": {"n_classes": 10, "min": 5, "max": 25, "mean": 12.0, "std": 5.0},
        "uppercase": {"n_classes": 26, "min": 5, "max": 115, "mean": 45.0, "std": 25.0},
        "lowercase": {"n_classes": 26, "min": 5, "max": 50, "mean": 15.0, "std": 10.0},
    },
}
ALL_RESULTS["baselines"] = [
    {
        "model_name": "Iso-Param CNN (raw)",
        "strict_accuracy": metrics["strict_accuracy"],
        "tolerant_accuracy": metrics["tolerant_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "cluster_f1": metrics["cluster_f1"],
        "avg_inference_time_ms": metrics["avg_inference_time_ms"],
        "params_k": count_params(m) / 1000,
    },
    {
        "model_name": "Hybrid Skeleton CNN",
        "strict_accuracy": metrics_h["strict_accuracy"],
        "tolerant_accuracy": metrics_h["tolerant_accuracy"],
        "macro_f1": metrics_h["macro_f1"],
        "cluster_f1": metrics_h["cluster_f1"],
        "avg_inference_time_ms": metrics_h["avg_inference_time_ms"],
        "params_k": count_params(m_h) / 1000,
    },
]

report_path = generate_report(ALL_RESULTS, CONFIG, OUTPUT_DIR)
logger.info(f"Report: {report_path}")

print("\n" + "=" * 50)
print("  [OK] SMOKE TEST PASSED!")
print(f"  Report: {report_path}")
print("=" * 50)
