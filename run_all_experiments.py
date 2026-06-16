#!/usr/bin/env python3
"""
==========================================================================
MASTER ORCHESTRATOR — Hybrid Skeletonization OCR Research Pipeline
==========================================================================
Menjalankan SELURUH eksperimen penelitian dalam satu kali eksekusi.
Semua konfigurasi diambil dari config.yaml.

Usage:
    python run_all_experiments.py                  # default config.yaml
    python run_all_experiments.py --config my.yaml # custom config

Output:
    research_outputs/
        LAPORAN_PENELITIAN.md   ← Laporan lengkap
        all_results.json        ← Data mentah
        figures/                ← Semua plot & confusion matrix
        models/                 ← Model weights tersimpan
==========================================================================
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# =====================================================================
# STEP 0: Parse args & load config BEFORE importing TF
# =====================================================================
parser = argparse.ArgumentParser(description="Run all OCR research experiments")
parser.add_argument("--config", default="config.yaml", help="Path to config file")
args = parser.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# GPU setup MUST happen before any TF import
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
        if CONFIG["hardware"].get("gpu_memory_growth", True):
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        if CONFIG["hardware"].get("mixed_precision", False):
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print(f"[GPU] {len(gpus)} GPU(s) detected: {[g.name for g in gpus]}")
    else:
        print("[GPU] No GPU found — falling back to CPU")
else:
    print("[GPU] Disabled by config — using CPU")

# Now safe to import everything
import numpy as np
import pandas as pd
from tqdm import tqdm

from research.data_loader import (
    load_chars74k_raw, load_chars74k_skeleton, load_emnist,
    split_dataset, get_class_distribution, get_label_encoder_chars74k,
)
from research.models import (
    build_cnn, build_standard_cnn, build_hybrid_cnn,
    build_iso_parameter_cnn, build_mobilenetv2, build_hog_svm,
    compile_keras_model, count_params,
)
from research.evaluation import (
    evaluate_keras_model, evaluate_sklearn_model,
    benchmark_latency, wilcoxon_test,
    save_training_curves, save_confusion_matrix,
    save_threshold_sensitivity_plot, save_baseline_comparison_chart,
    save_latency_distribution_plot,
)
from research.preprocessing import compute_skeleton_quality, preprocess_single
from research.report_generator import generate_report

# =====================================================================
# LOGGING SETUP
# =====================================================================
OUTPUT_DIR = CONFIG["project"]["output_dir"]
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, CONFIG["project"].get("log_level", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "experiment.log"),
                            encoding="utf-8"),
    ],
)
logger = logging.getLogger("orchestrator")
SEED = CONFIG["project"]["random_seed"]
tf.random.set_seed(SEED)
np.random.seed(SEED)

ALL_RESULTS = {}  # Agregasi seluruh hasil

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def _banner(text):
    """Print a banner line."""
    sep = "=" * 65
    logger.info(sep)
    logger.info(f"  {text}")
    logger.info(sep)


def _timer(func, *args, **kwargs):
    """Run function and return (result, elapsed_seconds)."""
    t0 = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - t0


def _train_keras(model, X_train, y_train, X_test, y_test, model_name,
                 save_key=None):
    """Train a Keras model with config settings. Returns (model, history)."""
    tc = CONFIG["training"]
    callbacks = []
    if tc["early_stopping"]["enabled"]:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=tc["early_stopping"]["monitor"],
            patience=tc["early_stopping"]["patience"],
            restore_best_weights=tc["early_stopping"]["restore_best_weights"],
            verbose=1,
        ))

    # TensorBoard logging setup
    tb_log_dir = os.path.join(OUTPUT_DIR, "tensorboard", save_key or model_name.replace(" ", "_").lower())
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=tb_log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch",
    ))

    logger.info(f"Training {model_name}: epochs={tc['epochs']}, "
                f"batch_size={tc['batch_size']}, "
                f"params={count_params(model):,}")

    history = model.fit(
        X_train, y_train,
        epochs=tc["epochs"],
        batch_size=tc["batch_size"],
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=2,
    )

    # Save training curves
    save_training_curves(
        history,
        os.path.join(FIGURES_DIR, f"training_{save_key or model_name}.png"),
        model_name,
    )

    # Save model weights
    if save_key:
        model_path = os.path.join(MODELS_DIR, f"{save_key}.keras")
        model.save(model_path)
        logger.info(f"Model saved: {model_path}")

    return model, history


def _eval_and_record(model, X_test, y_test, le, model_name, model_type="keras",
                     save_key=None, extra_fields=None):
    """Evaluate model and return results dict."""
    if model_type == "keras":
        metrics = evaluate_keras_model(
            model, X_test, y_test, le,
            batch_size=CONFIG["training"]["batch_size"],
        )
    else:
        metrics = evaluate_sklearn_model(model, X_test, y_test, le)

    # Save confusion matrix
    key = save_key or model_name.replace(" ", "_").lower()
    save_confusion_matrix(
        np.array(metrics["y_true"]), np.array(metrics["y_pred"]),
        list(le.classes_),
        os.path.join(FIGURES_DIR, f"cm_{key}.png"),
        model_name,
    )

    # Save classification report
    report_path = os.path.join(OUTPUT_DIR, f"report_{key}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Samples: {metrics['n_samples']}\n\n")
        f.write(metrics["classification_report"])

    result = {
        "model_name": model_name,
        "strict_accuracy": metrics["strict_accuracy"],
        "tolerant_accuracy": metrics["tolerant_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "cluster_f1": metrics["cluster_f1"],
        "avg_inference_time_ms": metrics["avg_inference_time_ms"],
        "params": count_params(model) if hasattr(model, "count_params") else 0,
        "params_k": (count_params(model) / 1000
                     if hasattr(model, "count_params") else 0),
        "n_samples": metrics["n_samples"],
    }
    if extra_fields:
        result.update(extra_fields)

    logger.info(
        f"  -> {model_name}: strict={metrics['strict_accuracy']:.2f}%, "
        f"tolerant={metrics['tolerant_accuracy']:.2f}%, "
        f"F1={metrics['macro_f1']:.4f}, "
        f"latency={metrics['avg_inference_time_ms']:.3f}ms"
    )

    return result


# =====================================================================
# EXPERIMENT 0: DATASET STATISTICS
# =====================================================================

def run_dataset_stats():
    """Hitung dan catat statistik dataset."""
    _banner("EXPERIMENT 0: Dataset Statistics")

    X_raw, y_raw, le = load_chars74k_raw(CONFIG)
    X_train, X_test, y_train, y_test = split_dataset(
        X_raw, y_raw, CONFIG["datasets"]["chars74k"]["test_split"], SEED
    )

    dist_df = get_class_distribution(y_raw, le)
    stats = {
        "total_samples": len(y_raw),
        "num_classes": len(le.classes_),
        "train_size": len(y_train),
        "test_size": len(y_test),
    }

    # Per-category distribution
    distribution = {}
    for cat in ["digit", "uppercase", "lowercase"]:
        subset = dist_df[dist_df["category"] == cat]
        distribution[cat] = {
            "n_classes": len(subset),
            "min": int(subset["count"].min()),
            "max": int(subset["count"].max()),
            "mean": float(subset["count"].mean()),
            "std": float(subset["count"].std()),
        }
    stats["distribution"] = distribution

    # Save distribution CSV
    dist_df.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"), index=False)
    logger.info(f"Dataset: {stats['total_samples']} samples, "
                f"{stats['num_classes']} classes")
    logger.info(f"  Digits: {distribution['digit']}")
    logger.info(f"  Uppercase: {distribution['uppercase']}")
    logger.info(f"  Lowercase: {distribution['lowercase']}")

    return stats, X_raw, y_raw, le


# =====================================================================
# EXPERIMENT 1: BASELINE COMPARISON
# =====================================================================

def run_baselines(X_raw, y_raw, le):
    """Jalankan 5 baseline models pada Chars74K."""
    _banner("EXPERIMENT 1: Baseline Comparison (5 Models)")

    X_train, X_test, y_train, y_test = split_dataset(
        X_raw, y_raw, CONFIG["datasets"]["chars74k"]["test_split"], SEED
    )
    num_classes = len(le.classes_)
    aug_cfg = CONFIG["training"]["augmentation"]
    results = []
    trained_models = {}

    exp = CONFIG["experiments"]["baselines"]["models"]

    # 1a. HOG + SVM
    if exp["hog_svm"]["enabled"]:
        logger.info("--- Training HOG + SVM ---")
        svm_pipeline = build_hog_svm(CONFIG)
        svm_pipeline.fit(X_train, y_train)
        r = _eval_and_record(svm_pipeline, X_test, y_test, le,
                             "HOG + SVM", model_type="sklearn",
                             save_key="hog_svm")
        results.append(r)
        trained_models["hog_svm"] = svm_pipeline

    # 1b. MobileNetV2-tiny
    if exp["mobilenetv2"]["enabled"]:
        logger.info("--- Training MobileNetV2-tiny ---")
        m = build_mobilenetv2(num_classes, alpha=exp["mobilenetv2"]["alpha"])
        m = compile_keras_model(m, CONFIG)
        m, _ = _train_keras(m, X_train, y_train, X_test, y_test,
                            "MobileNetV2-tiny", save_key="mobilenetv2")
        r = _eval_and_record(m, X_test, y_test, le, "MobileNetV2-tiny",
                             save_key="mobilenetv2")
        results.append(r)
        trained_models["mobilenetv2"] = m

    # 1c. Iso-parameter CNN (same arch as hybrid, raw input)
    if exp["iso_parameter_cnn"]["enabled"]:
        logger.info("--- Training Iso-Parameter CNN (raw) ---")
        m = build_iso_parameter_cnn(num_classes)
        m = compile_keras_model(m, CONFIG)
        m, _ = _train_keras(m, X_train, y_train, X_test, y_test,
                            "Iso-Param CNN (raw)", save_key="iso_param")
        r = _eval_and_record(m, X_test, y_test, le, "Iso-Param CNN (raw)",
                             save_key="iso_param")
        results.append(r)
        trained_models["iso_param"] = m

    # 1d. Standard CNN (raw)
    if exp["standard_cnn"]["enabled"]:
        logger.info("--- Training Standard CNN (raw) ---")
        m = build_standard_cnn(num_classes)
        m = compile_keras_model(m, CONFIG)
        m, _ = _train_keras(m, X_train, y_train, X_test, y_test,
                            "Standard CNN (raw)", save_key="standard_cnn")
        r = _eval_and_record(m, X_test, y_test, le, "Standard CNN (raw)",
                             save_key="standard_cnn")
        results.append(r)
        trained_models["standard_cnn"] = m

    # 1e. Hybrid Skeleton CNN
    if exp["hybrid_skeleton"]["enabled"]:
        logger.info("--- Loading skeleton data & training Hybrid CNN ---")
        X_skel, y_skel, le_skel = load_chars74k_skeleton(CONFIG)
        Xs_tr, Xs_te, ys_tr, ys_te = split_dataset(
            X_skel, y_skel, CONFIG["datasets"]["chars74k"]["test_split"], SEED
        )
        m = build_hybrid_cnn(num_classes, aug_config=aug_cfg)
        m = compile_keras_model(m, CONFIG)
        m, _ = _train_keras(m, Xs_tr, ys_tr, Xs_te, ys_te,
                            "Hybrid Skeleton CNN", save_key="hybrid_skeleton")
        r = _eval_and_record(m, Xs_te, ys_te, le_skel,
                             "Hybrid Skeleton CNN", save_key="hybrid_skeleton")
        results.append(r)
        trained_models["hybrid_skeleton"] = m
        trained_models["_skel_test"] = (Xs_te, ys_te, le_skel)

    # Save comparison chart
    if len(results) >= 2:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(OUTPUT_DIR, "baseline_comparison.csv"), index=False)
        save_baseline_comparison_chart(
            df, os.path.join(FIGURES_DIR, "baseline_comparison.png")
        )

    return results, trained_models


# =====================================================================
# EXPERIMENT 2: CONTROLLED VARIABLE EXPERIMENT
# =====================================================================

def run_controlled_experiment(X_raw, y_raw, le):
    """Fixed architecture × varied input untuk mengisolasi efek preprocessing."""
    _banner("EXPERIMENT 2: Controlled Variable Experiment")

    X_train_r, X_test_r, y_train, y_test = split_dataset(
        X_raw, y_raw, CONFIG["datasets"]["chars74k"]["test_split"], SEED
    )
    num_classes = len(le.classes_)
    aug_cfg = CONFIG["training"]["augmentation"]

    # Load skeleton data
    X_skel, y_skel, _ = load_chars74k_skeleton(CONFIG)
    X_train_s, X_test_s, _, _ = split_dataset(
        X_skel, y_skel, CONFIG["datasets"]["chars74k"]["test_split"], SEED
    )

    results = []
    exp_cfg = CONFIG["experiments"]["controlled_experiment"]

    for arch in exp_cfg["architectures"]:
        arch_name = arch["name"]
        filters = arch["filters"]
        dense = arch["dense_units"]

        for input_type in exp_cfg["input_types"]:
            label = f"{arch_name}_{input_type}"
            logger.info(f"--- Controlled: arch={arch_name}, input={input_type} ---")

            use_aug = input_type == "skeleton_augmented"
            if input_type in ("skeleton", "skeleton_augmented"):
                Xtr, Xte = X_train_s, X_test_s
            else:
                Xtr, Xte = X_train_r, X_test_r

            m = build_cnn(filters, dense, num_classes,
                          use_augmentation=use_aug,
                          aug_config=aug_cfg if use_aug else None)
            m = compile_keras_model(m, CONFIG)
            m, _ = _train_keras(m, Xtr, y_train, Xte, y_test,
                                f"Ctrl-{label}", save_key=f"ctrl_{label}")
            r = _eval_and_record(m, Xte, y_test, le,
                                 f"Ctrl-{label}", save_key=f"ctrl_{label}")
            r["architecture"] = f"{arch_name} ({filters}, Dense {dense})"
            r["input_type"] = input_type
            results.append(r)

    # Save CSV
    pd.DataFrame(results).to_csv(
        os.path.join(OUTPUT_DIR, "controlled_experiment.csv"), index=False
    )
    return results


# =====================================================================
# EXPERIMENT 3: THRESHOLD SENSITIVITY ANALYSIS
# =====================================================================

def run_threshold_sensitivity(y_raw_full, le):
    """Sweep threshold hole-filling dan ukur dampak pada akurasi + kualitas."""
    _banner("EXPERIMENT 3: Threshold Sensitivity Analysis")

    thresholds = CONFIG["preprocessing"]["threshold_sweep"]
    num_classes = len(le.classes_)
    aug_cfg = CONFIG["training"]["augmentation"]
    results = []

    for thr in thresholds:
        logger.info(f"--- Threshold = {thr} px ---")

        # Preprocess with this threshold
        X_skel, y_skel, le_t = load_chars74k_skeleton(CONFIG, threshold=thr)
        X_tr, X_te, y_tr, y_te = split_dataset(
            X_skel, y_skel, CONFIG["datasets"]["chars74k"]["test_split"], SEED
        )

        # Train hybrid model
        m = build_hybrid_cnn(num_classes, aug_config=aug_cfg)
        m = compile_keras_model(m, CONFIG)
        m, _ = _train_keras(m, X_tr, y_tr, X_te, y_te,
                            f"Hybrid-thr{thr}", save_key=f"thr_{thr}")
        r = _eval_and_record(m, X_te, y_te, le_t,
                             f"Hybrid (thr={thr})", save_key=f"thr_{thr}")
        r["threshold"] = thr

        # Skeleton quality untuk threshold ini
        sq = _compute_skeleton_quality_stats(CONFIG, thr)
        r["skeleton_quality"] = sq
        results.append(r)

    # Save plot & CSV
    save_threshold_sensitivity_plot(
        results, os.path.join(FIGURES_DIR, "threshold_sensitivity.png")
    )
    pd.DataFrame([{
        "threshold": r["threshold"],
        "strict_accuracy": r["strict_accuracy"],
        "tolerant_accuracy": r["tolerant_accuracy"],
        "macro_f1": r["macro_f1"],
        **{f"sq_{k}": v for k, v in r.get("skeleton_quality", {}).items()},
    } for r in results]).to_csv(
        os.path.join(OUTPUT_DIR, "threshold_sensitivity.csv"), index=False
    )
    return results


def _compute_skeleton_quality_stats(config, threshold, max_samples=500):
    """Hitung rata-rata skeleton quality metrics untuk threshold tertentu."""
    import cv2 as cv2_local
    from research.preprocessing import preprocess_single as ps
    from research.preprocessing import compute_skeleton_quality as csq

    ds = config["datasets"]["chars74k"]
    pp_size = tuple(ds["preprocessing_size"])
    csv_path = ds["csv_path"]
    raw_dir = ds["raw_dir"]

    df = pd.read_csv(csv_path)
    all_metrics = []
    count = 0

    for _, row in df.iterrows():
        folder = os.path.join(raw_dir, row["Folder Name"])
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            img = cv2_local.imread(os.path.join(folder, fname),
                                   cv2_local.IMREAD_GRAYSCALE)
            if img is None:
                continue
            skel, _ = ps(img, threshold, pp_size)
            m = csq(skel > 0)
            all_metrics.append(m)
            count += 1
            if count >= max_samples:
                break
        if count >= max_samples:
            break

    if not all_metrics:
        return {}

    keys = all_metrics[0].keys()
    return {
        f"avg_{k}": float(np.mean([m[k] for m in all_metrics]))
        for k in keys
    }


# =====================================================================
# EXPERIMENT 4: CROSS-DATASET VALIDATION (EMNIST)
# =====================================================================

def run_cross_dataset(trained_models, le):
    """Evaluasi model trained-on-Chars74K pada EMNIST test set."""
    _banner("EXPERIMENT 4: Cross-Dataset Validation (EMNIST)")

    if not CONFIG["datasets"]["emnist"]["enabled"]:
        logger.warning("EMNIST disabled in config — skipping")
        return []

    X_emnist_raw, X_emnist_skel, y_emnist, le_emnist = load_emnist(CONFIG)
    results = []

    # Evaluate raw-input models on EMNIST raw
    for key, name in [("standard_cnn", "Standard CNN"),
                      ("iso_param", "Iso-Param CNN"),
                      ("mobilenetv2", "MobileNetV2-tiny")]:
        if key not in trained_models:
            continue
        logger.info(f"--- Cross-dataset: {name} on EMNIST raw ---")
        m = trained_models[key]
        metrics = evaluate_keras_model(
            m, X_emnist_raw, y_emnist, le_emnist,
            batch_size=CONFIG["training"]["batch_size"],
        )
        # Get Chars74K results from baselines
        chars74k_strict = 0
        if "baselines" in ALL_RESULTS:
            for br in ALL_RESULTS["baselines"]:
                if br["model_name"] == f"{name} (raw)" or br["model_name"] == name:
                    chars74k_strict = br["strict_accuracy"]
                    break

        results.append({
            "model_name": name,
            "input_type": "raw",
            "chars74k_strict": chars74k_strict,
            "emnist_strict": metrics["strict_accuracy"],
            "emnist_tolerant": metrics["tolerant_accuracy"],
            "emnist_macro_f1": metrics["macro_f1"],
        })

    # Evaluate skeleton model on EMNIST skeleton
    if "hybrid_skeleton" in trained_models:
        logger.info("--- Cross-dataset: Hybrid Skeleton on EMNIST skeleton ---")
        m = trained_models["hybrid_skeleton"]
        metrics = evaluate_keras_model(
            m, X_emnist_skel, y_emnist, le_emnist,
            batch_size=CONFIG["training"]["batch_size"],
        )
        chars74k_strict = 0
        if "baselines" in ALL_RESULTS:
            for br in ALL_RESULTS["baselines"]:
                if "Hybrid" in br["model_name"]:
                    chars74k_strict = br["strict_accuracy"]
                    break

        results.append({
            "model_name": "Hybrid Skeleton CNN",
            "input_type": "skeleton",
            "chars74k_strict": chars74k_strict,
            "emnist_strict": metrics["strict_accuracy"],
            "emnist_tolerant": metrics["tolerant_accuracy"],
            "emnist_macro_f1": metrics["macro_f1"],
        })

    # HOG+SVM on EMNIST
    if "hog_svm" in trained_models:
        logger.info("--- Cross-dataset: HOG+SVM on EMNIST raw ---")
        m = trained_models["hog_svm"]
        metrics = evaluate_sklearn_model(m, X_emnist_raw, y_emnist, le_emnist)
        chars74k_strict = 0
        if "baselines" in ALL_RESULTS:
            for br in ALL_RESULTS["baselines"]:
                if "HOG" in br["model_name"]:
                    chars74k_strict = br["strict_accuracy"]
                    break
        results.append({
            "model_name": "HOG + SVM",
            "input_type": "raw",
            "chars74k_strict": chars74k_strict,
            "emnist_strict": metrics["strict_accuracy"],
            "emnist_tolerant": metrics["tolerant_accuracy"],
            "emnist_macro_f1": metrics["macro_f1"],
        })

    pd.DataFrame(results).to_csv(
        os.path.join(OUTPUT_DIR, "cross_dataset_emnist.csv"), index=False
    )
    return results


# =====================================================================
# EXPERIMENT 5: RIGOROUS LATENCY BENCHMARK
# =====================================================================

def run_latency_benchmark(trained_models):
    """Benchmark latensi N=1000 runs + Wilcoxon test."""
    _banner("EXPERIMENT 5: Rigorous Latency Benchmark")

    lb_cfg = CONFIG["experiments"]["latency_benchmark"]
    n_warmup = lb_cfg.get("warmup_iterations", 100)
    n_runs = lb_cfg.get("benchmark_iterations", 1000)

    # Prepare test sample (satu gambar random)
    test_sample = np.random.rand(1, 32, 32, 1).astype(np.float32)

    latency_results = {}
    keras_models_for_test = {}

    for key, name in [("standard_cnn", "Standard CNN"),
                      ("hybrid_skeleton", "Hybrid Skeleton CNN"),
                      ("iso_param", "Iso-Param CNN"),
                      ("mobilenetv2", "MobileNetV2-tiny")]:
        if key not in trained_models:
            continue
        logger.info(f"--- Benchmarking: {name} ({n_runs} runs) ---")
        m = trained_models[key]
        result = benchmark_latency(m, test_sample, n_warmup, n_runs)
        latency_results[name] = result
        keras_models_for_test[name] = result
        logger.info(f"  -> {name}: mean={result['mean_ms']:.3f}ms +/- "
                     f"{result['std_ms']:.3f}, CI95=[{result['ci95_lower']:.3f}, "
                     f"{result['ci95_upper']:.3f}]")

    # Wilcoxon tests (semua pasangan vs Hybrid)
    wilcoxon_results = []
    if "Hybrid Skeleton CNN" in latency_results:
        hybrid_lat = latency_results["Hybrid Skeleton CNN"]["all_latencies"]
        for name, data in latency_results.items():
            if name == "Hybrid Skeleton CNN":
                continue
            wt = wilcoxon_test(hybrid_lat, data["all_latencies"],
                              "Hybrid Skeleton CNN", name)
            wilcoxon_results.append(wt)
            logger.info(f"  Wilcoxon {name}: p={wt['p_value']:.6f}, "
                         f"sig={'YES' if wt['significant_005'] else 'NO'}")

    # Save plot
    if len(latency_results) >= 2:
        save_latency_distribution_plot(
            latency_results,
            os.path.join(FIGURES_DIR, "latency_distributions.png"),
        )

    return {
        "warmup": n_warmup,
        "n_runs": n_runs,
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "all_latencies"}
                   for k, v in latency_results.items()},
        "wilcoxon_tests": wilcoxon_results,
    }


# =====================================================================
# EXPERIMENT 6: SKELETON QUALITY ANALYSIS
# =====================================================================

def run_skeleton_quality():
    """Analisis kualitas skeleton per threshold."""
    _banner("EXPERIMENT 6: Skeleton Quality Analysis")

    thresholds = CONFIG["preprocessing"]["threshold_sweep"]
    results = []

    for thr in thresholds:
        logger.info(f"--- Skeleton quality for threshold = {thr} ---")
        sq = _compute_skeleton_quality_stats(CONFIG, thr, max_samples=1000)
        sq["threshold"] = thr
        results.append(sq)
        logger.info(f"  -> avg_total_pixels={sq.get('avg_total_pixels', 0):.1f}, "
                     f"avg_junctions={sq.get('avg_junctions', 0):.1f}, "
                     f"connectivity={sq.get('avg_connectivity_score', 0):.4f}")

    # Rename for consistent report format
    formatted = []
    for r in results:
        formatted.append({
            "threshold": r.get("threshold", 0),
            "avg_total_pixels": r.get("avg_total_pixels", 0),
            "avg_endpoints": r.get("avg_endpoints", 0),
            "avg_junctions": r.get("avg_junctions", 0),
            "avg_loops": r.get("avg_loops", 0),
            "avg_connectivity": r.get("avg_connectivity_score", 0),
            "avg_branch_ratio": r.get("avg_branch_point_ratio", 0),
        })

    pd.DataFrame(formatted).to_csv(
        os.path.join(OUTPUT_DIR, "skeleton_quality.csv"), index=False
    )
    return formatted


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    global ALL_RESULTS

    total_start = time.time()
    _banner(f"RESEARCH PIPELINE START — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Config: {args.config}")

    # Save config copy
    with open(os.path.join(OUTPUT_DIR, "config_used.yaml"), "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

    # --- Exp 0: Dataset Stats ---
    stats, X_raw, y_raw, le = run_dataset_stats()
    ALL_RESULTS["dataset_stats"] = stats

    # --- Exp 1: Baselines ---
    trained_models = {}
    if CONFIG["experiments"]["baselines"]["enabled"]:
        baseline_results, trained_models = run_baselines(X_raw, y_raw, le)
        ALL_RESULTS["baselines"] = baseline_results
        _save_checkpoint("after_baselines")

    # --- Exp 2: Controlled Experiment ---
    if CONFIG["experiments"]["controlled_experiment"]["enabled"]:
        ctrl_results = run_controlled_experiment(X_raw, y_raw, le)
        ALL_RESULTS["controlled_experiment"] = ctrl_results
        _save_checkpoint("after_controlled")

    # --- Exp 3: Threshold Sensitivity ---
    if CONFIG["experiments"]["threshold_sensitivity"]["enabled"]:
        thr_results = run_threshold_sensitivity(y_raw, le)
        ALL_RESULTS["threshold_sensitivity"] = thr_results
        _save_checkpoint("after_threshold")

    # --- Exp 4: Cross-Dataset ---
    if CONFIG["experiments"]["cross_dataset"]["enabled"]:
        cross_results = run_cross_dataset(trained_models, le)
        ALL_RESULTS["cross_dataset"] = cross_results
        _save_checkpoint("after_crossdataset")

    # --- Exp 5: Latency Benchmark ---
    if CONFIG["experiments"]["latency_benchmark"]["enabled"]:
        lat_results = run_latency_benchmark(trained_models)
        ALL_RESULTS["latency_benchmark"] = lat_results
        _save_checkpoint("after_latency")

    # --- Exp 6: Skeleton Quality ---
    if CONFIG["experiments"]["skeleton_quality"]["enabled"]:
        sq_results = run_skeleton_quality()
        ALL_RESULTS["skeleton_quality"] = sq_results
        _save_checkpoint("after_skeleton_quality")

    # --- Generate Final Report ---
    _banner("GENERATING FINAL REPORT")
    report_path = generate_report(ALL_RESULTS, CONFIG, OUTPUT_DIR)

    total_time = time.time() - total_start
    _banner(f"ALL EXPERIMENTS COMPLETE — {total_time/60:.1f} minutes")
    logger.info(f"Report: {report_path}")
    logger.info(f"Raw data: {os.path.join(OUTPUT_DIR, 'all_results.json')}")
    logger.info(f"Figures: {FIGURES_DIR}")

    print("\n" + "=" * 65)
    print(f"  [DONE] SELESAI! Total waktu: {total_time/60:.1f} menit")
    print(f"  [REPORT]  Laporan: {report_path}")
    print(f"  [FIGURES] Figures: {FIGURES_DIR}")
    print(f"  [DATA]    Data mentah: {os.path.join(OUTPUT_DIR, 'all_results.json')}")
    print("=" * 65)


def _save_checkpoint(name):
    """Save intermediate results as checkpoint."""
    from research.report_generator import _save_json_safe
    path = os.path.join(OUTPUT_DIR, f"checkpoint_{name}.json")
    _save_json_safe(ALL_RESULTS, path)
    logger.info(f"Checkpoint saved: {path}")


if __name__ == "__main__":
    main()
