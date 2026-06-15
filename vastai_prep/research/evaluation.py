"""
Evaluasi lengkap: metrik akurasi, F1, latensi statistik, uji Wilcoxon,
confusion matrix, dan per-cluster analysis.
"""

import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_recall_fscore_support)

logger = logging.getLogger(__name__)


# =====================================================================
# CORE EVALUATION
# =====================================================================

def evaluate_keras_model(model, X_test, y_test, label_encoder, batch_size=32):
    """
    Evaluasi lengkap model Keras.

    Returns
    -------
    dict dengan seluruh metrik: strict_accuracy, tolerant_accuracy,
    macro_f1, per_cluster_f1, classification_report_text, y_pred, dll.
    """
    # Warmup
    if len(X_test) > 0:
        _ = model.predict(X_test[:1], batch_size=1, verbose=0)

    start = time.perf_counter()
    predictions = model.predict(X_test, batch_size=batch_size, verbose=0)
    end = time.perf_counter()

    y_pred = np.argmax(predictions, axis=1)
    return _compute_metrics(y_test, y_pred, label_encoder, end - start,
                            len(y_test))


def evaluate_sklearn_model(model, X_test, y_test, label_encoder):
    """Evaluasi lengkap model sklearn (HOG+SVM)."""
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    end = time.perf_counter()

    return _compute_metrics(y_test, y_pred, label_encoder, end - start,
                            len(y_test))


def _compute_metrics(y_true, y_pred, label_encoder, total_time, n_samples):
    """Hitung semua metrik dari y_true dan y_pred."""
    y_true_chars = label_encoder.inverse_transform(y_true)
    y_pred_chars = label_encoder.inverse_transform(y_pred)
    class_names = list(label_encoder.classes_)

    # Strict & tolerant accuracy
    strict_correct = 0
    case_error_correct = 0
    total_wrong = 0
    for tc, pc in zip(y_true_chars, y_pred_chars):
        if tc == pc:
            strict_correct += 1
        elif tc.lower() == pc.lower():
            case_error_correct += 1
        else:
            total_wrong += 1

    strict_acc = (strict_correct / n_samples) * 100
    tolerant_acc = ((strict_correct + case_error_correct) / n_samples) * 100

    # Macro F1
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-cluster F1 (digits, uppercase, lowercase)
    cluster_f1 = _per_cluster_f1(y_true, y_pred, label_encoder)

    # Classification report
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0,
    )

    # Weighted F1
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "n_samples": n_samples,
        "strict_correct": strict_correct,
        "case_error_correct": case_error_correct,
        "total_wrong": total_wrong,
        "strict_accuracy": round(strict_acc, 2),
        "tolerant_accuracy": round(tolerant_acc, 2),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "cluster_f1": cluster_f1,
        "total_inference_time_sec": round(total_time, 4),
        "avg_inference_time_ms": round((total_time / n_samples) * 1000, 4),
        "classification_report": report_text,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def _per_cluster_f1(y_true, y_pred, label_encoder):
    """Hitung F1 terpisah untuk digits (0-9), uppercase (A-Z), lowercase (a-z)."""
    classes = list(label_encoder.classes_)
    clusters = {
        "digits": [i for i, c in enumerate(classes) if c.isdigit()],
        "uppercase": [i for i, c in enumerate(classes) if c.isupper()],
        "lowercase": [i for i, c in enumerate(classes) if c.islower()],
    }
    result = {}
    for name, indices in clusters.items():
        mask = np.isin(y_true, indices)
        if mask.sum() == 0:
            result[name] = {"f1": 0.0, "support": 0}
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        f1 = f1_score(yt, yp, average="macro", zero_division=0, labels=indices)
        result[name] = {
            "f1": round(float(f1), 4),
            "support": int(mask.sum()),
        }
    return result


# =====================================================================
# LATENCY BENCHMARK
# =====================================================================

def benchmark_latency(model, X_test, n_warmup=100, n_runs=1000):
    """
    Benchmark latensi per-sampel yang rigorous.

    Returns
    -------
    dict: mean_ms, std_ms, median_ms, ci95_lower, ci95_upper, p99_ms,
          all_latencies (list of float in ms)
    """
    sample = X_test[:1]

    # Warmup
    for _ in range(n_warmup):
        _ = model.predict(sample, batch_size=1, verbose=0)

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(sample, batch_size=1, verbose=0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    arr = np.array(latencies)
    ci = scipy_stats.t.interval(
        0.95, len(arr) - 1, loc=np.mean(arr), scale=scipy_stats.sem(arr)
    )

    return {
        "mean_ms": round(float(np.mean(arr)), 4),
        "std_ms": round(float(np.std(arr)), 4),
        "median_ms": round(float(np.median(arr)), 4),
        "ci95_lower": round(float(ci[0]), 4),
        "ci95_upper": round(float(ci[1]), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "min_ms": round(float(np.min(arr)), 4),
        "max_ms": round(float(np.max(arr)), 4),
        "n_runs": n_runs,
        "all_latencies": arr.tolist(),
    }


def wilcoxon_test(latencies_a, latencies_b, model_a_name, model_b_name):
    """
    Wilcoxon signed-rank test antara dua distribusi latensi.
    Memerlukan jumlah sampel yang sama — gunakan min(len_a, len_b).
    """
    n = min(len(latencies_a), len(latencies_b))
    a = np.array(latencies_a[:n])
    b = np.array(latencies_b[:n])

    stat, p_value = scipy_stats.wilcoxon(a, b, alternative="two-sided")

    return {
        "test": "Wilcoxon signed-rank",
        "model_a": model_a_name,
        "model_b": model_b_name,
        "n_pairs": n,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "mean_diff_ms": round(float(np.mean(a) - np.mean(b)), 4),
    }


# =====================================================================
# VISUALIZATION
# =====================================================================

def save_training_curves(history, output_path, title):
    """Plot kurva training accuracy dan loss."""
    h = history.history
    epochs = range(1, len(h.get("accuracy", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, h.get("accuracy", []), label="Train Acc")
    ax1.plot(epochs, h.get("val_accuracy", []), label="Val Acc")
    ax1.set_title(f"{title} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, h.get("loss", []), label="Train Loss")
    ax2.plot(epochs, h.get("val_loss", []), label="Val Loss")
    ax2.set_title(f"{title} — Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training curves: {output_path}")


def save_confusion_matrix(y_true, y_pred, class_names, output_path, title):
    """Plot normalized confusion matrix."""
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float32)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm /= row_sums

    fig_size = max(12, len(class_names) * 0.28)
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{title} — Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90, fontsize=6)
    plt.yticks(ticks, class_names, fontsize=6)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    threshold = cm_norm.max() * 0.5 if cm_norm.size else 0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            v = cm_norm[i, j]
            if v > 0:
                plt.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v > threshold else "black",
                         fontsize=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix: {output_path}")


def save_threshold_sensitivity_plot(results, output_path):
    """Plot kurva sensitivitas threshold vs accuracy/F1."""
    thresholds = [r["threshold"] for r in results]
    strict = [r["strict_accuracy"] for r in results]
    tolerant = [r["tolerant_accuracy"] for r in results]
    f1 = [r["macro_f1"] * 100 for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(thresholds, strict, "o-", label="Strict Accuracy", linewidth=2)
    ax1.plot(thresholds, tolerant, "s-", label="Tolerant Accuracy", linewidth=2)
    ax1.plot(thresholds, f1, "^-", label="Macro F1 (×100)", linewidth=2)
    ax1.set_xlabel("Hole-Filling Threshold (pixels)")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Threshold Sensitivity — Accuracy & F1")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xticks(thresholds)

    # Skeleton quality metrics
    if "skeleton_quality" in results[0]:
        avg_len = [r["skeleton_quality"]["avg_total_pixels"] for r in results]
        avg_junc = [r["skeleton_quality"]["avg_junctions"] for r in results]

        ax2.bar([t - 1 for t in thresholds], avg_len, width=2,
                label="Avg Skeleton Length", alpha=0.7)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(thresholds, avg_junc, "ro-", label="Avg Junctions",
                      linewidth=2)
        ax2.set_xlabel("Hole-Filling Threshold (pixels)")
        ax2.set_ylabel("Avg Skeleton Length (pixels)")
        ax2_twin.set_ylabel("Avg Junctions")
        ax2.set_title("Threshold vs Skeleton Quality")
        ax2.legend(loc="upper left")
        ax2_twin.legend(loc="upper right")
        ax2.grid(alpha=0.3)
        ax2.set_xticks(thresholds)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved threshold sensitivity plot: {output_path}")


def save_baseline_comparison_chart(results_df, output_path):
    """Bar chart perbandingan semua baseline models."""
    models = results_df["model_name"].tolist()
    x = np.arange(len(models))
    w = 0.2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Accuracy
    axes[0].bar(x - w, results_df["strict_accuracy"], w, label="Strict Acc")
    axes[0].bar(x, results_df["tolerant_accuracy"], w, label="Tolerant Acc")
    axes[0].bar(x + w, results_df["macro_f1"] * 100, w, label="Macro F1 (×100)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Accuracy & F1 Comparison")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0, 100)

    # Parameters
    axes[1].bar(models, results_df["params_k"], color="#2ca02c")
    axes[1].set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Parameters (×1000)")
    axes[1].set_title("Model Size")
    axes[1].grid(axis="y", alpha=0.25)

    # Latency
    axes[2].bar(models, results_df["avg_inference_time_ms"], color="#ff7f0e")
    axes[2].set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    axes[2].set_ylabel("Avg Inference (ms)")
    axes[2].set_title("Inference Speed")
    axes[2].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved baseline comparison chart: {output_path}")


def save_latency_distribution_plot(latency_results, output_path):
    """Histogram distribusi latensi untuk semua model."""
    n = len(latency_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for i, (name, data) in enumerate(latency_results.items()):
        ax = axes[0][i]
        arr = np.array(data["all_latencies"])
        ax.hist(arr, bins=50, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.axvline(data["mean_ms"], color="red", linestyle="--",
                   label=f"Mean={data['mean_ms']:.3f}")
        ax.axvline(data["median_ms"], color="green", linestyle="--",
                   label=f"Median={data['median_ms']:.3f}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved latency distribution: {output_path}")
