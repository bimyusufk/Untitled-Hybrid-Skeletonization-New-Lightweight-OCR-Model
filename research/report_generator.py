"""
Report generator: mengagregasi seluruh hasil eksperimen menjadi satu laporan
Markdown komprehensif yang siap dijadikan bahan revisi paper LaTeX.
"""

import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_report(all_results, config, output_dir):
    """
    Generate laporan markdown lengkap dari seluruh hasil eksperimen.

    Parameters
    ----------
    all_results : dict
        Dictionary berisi semua hasil eksperimen.
    config : dict
        Konfigurasi penelitian.
    output_dir : str
        Direktori output untuk laporan.

    Returns
    -------
    str : Path ke file laporan.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "LAPORAN_PENELITIAN.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = []
    sections.append(f"# Laporan Hasil Penelitian — {config['project']['name']}\n")
    sections.append(f"**Generated**: {timestamp}\n")
    sections.append(f"**Config**: `config.yaml`\n")
    sections.append("---\n")

    # --- Dataset Statistics ---
    if "dataset_stats" in all_results:
        sections.append(_section_dataset_stats(all_results["dataset_stats"]))

    # --- Baseline Comparison ---
    if "baselines" in all_results:
        sections.append(_section_baselines(all_results["baselines"], output_dir))

    # --- Controlled Experiment ---
    if "controlled_experiment" in all_results:
        sections.append(_section_controlled(all_results["controlled_experiment"]))

    # --- Threshold Sensitivity ---
    if "threshold_sensitivity" in all_results:
        sections.append(_section_threshold(all_results["threshold_sensitivity"],
                                           output_dir))

    # --- Cross-Dataset Validation ---
    if "cross_dataset" in all_results:
        sections.append(_section_cross_dataset(all_results["cross_dataset"]))

    # --- Latency Benchmark ---
    if "latency_benchmark" in all_results:
        sections.append(_section_latency(all_results["latency_benchmark"], output_dir))

    # --- Skeleton Quality ---
    if "skeleton_quality" in all_results:
        sections.append(_section_skeleton_quality(all_results["skeleton_quality"]))

    report_text = "\n".join(sections)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Save raw results as JSON
    json_path = os.path.join(output_dir, "all_results.json")
    _save_json_safe(all_results, json_path)

    logger.info(f"Report saved: {report_path}")
    logger.info(f"Raw results saved: {json_path}")
    return report_path


def _save_json_safe(data, path):
    """Save dict ke JSON, handle numpy types."""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


# =====================================================================
# SECTION GENERATORS
# =====================================================================

def _section_dataset_stats(stats):
    lines = [
        "## 1. Statistik Dataset\n",
        f"**Total sampel Chars74K**: {stats.get('total_samples', 'N/A')}",
        f"**Jumlah kelas**: {stats.get('num_classes', 62)}",
        f"**Train/Test split**: {stats.get('train_size', 'N/A')} / {stats.get('test_size', 'N/A')}\n",
    ]

    if "distribution" in stats:
        lines.append("### Distribusi Kelas\n")
        lines.append("| Kategori | Jumlah Kelas | Min Sampel | Max Sampel | Mean Sampel | Std |")
        lines.append("|----------|-------------|------------|------------|-------------|-----|")
        for cat, info in stats["distribution"].items():
            lines.append(
                f"| {cat} | {info['n_classes']} | {info['min']} | "
                f"{info['max']} | {info['mean']:.1f} | {info['std']:.1f} |"
            )
        lines.append("")

    return "\n".join(lines)


def _section_baselines(results, output_dir=None):
    lines = [
        "## 2. Perbandingan Baseline\n",
        "| Model | Strict Acc (%) | Tolerant Acc (%) | Macro F1 | Params (K) | Latency (ms) |",
        "|-------|---------------|-----------------|----------|------------|-------------|",
    ]
    for r in results:
        params_k = r.get("params_k", r.get("params", 0) / 1000)
        lines.append(
            f"| {r['model_name']} | {r['strict_accuracy']:.2f} | "
            f"{r['tolerant_accuracy']:.2f} | {r['macro_f1']:.4f} | "
            f"{params_k:.1f} | {r['avg_inference_time_ms']:.2f} |"
        )
    lines.append("")

    # Per-cluster F1
    lines.append("### Per-Cluster F1 Score\n")
    lines.append("| Model | Digits F1 | Uppercase F1 | Lowercase F1 |")
    lines.append("|-------|-----------|-------------|-------------|")
    for r in results:
        cf = r.get("cluster_f1", {})
        lines.append(
            f"| {r['model_name']} | "
            f"{cf.get('digits', {}).get('f1', 0):.4f} | "
            f"{cf.get('uppercase', {}).get('f1', 0):.4f} | "
            f"{cf.get('lowercase', {}).get('f1', 0):.4f} |"
        )
    lines.append("")
    if output_dir:
        plot_path = os.path.join(output_dir, "figures", "baseline_comparison.png")
        if os.path.exists(plot_path):
            lines.append("![Baseline Comparison](figures/baseline_comparison.png)\n")
    return "\n".join(lines)


def _section_controlled(results):
    lines = [
        "## 3. Controlled Variable Experiment\n",
        "**Tujuan**: Mengisolasi efek preprocessing skeletonization dari "
        "perbedaan kapasitas arsitektur.\n",
        "| Architecture | Input Type | Strict Acc (%) | Tolerant Acc (%) | Macro F1 |",
        "|-------------|-----------|---------------|-----------------|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['architecture']} | {r['input_type']} | "
            f"{r['strict_accuracy']:.2f} | {r['tolerant_accuracy']:.2f} | "
            f"{r['macro_f1']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_threshold(results, output_dir):
    lines = [
        "## 4. Analisis Sensitivitas Threshold Hole-Filling\n",
        "| Threshold (px) | Strict Acc (%) | Tolerant Acc (%) | Macro F1 | "
        "Avg Skel. Length | Avg Junctions | Connectivity |",
        "|---------------|---------------|-----------------|----------|"
        "----------------|---------------|-------------|",
    ]
    for r in results:
        sq = r.get("skeleton_quality", {})
        lines.append(
            f"| {r['threshold']} | {r['strict_accuracy']:.2f} | "
            f"{r['tolerant_accuracy']:.2f} | {r['macro_f1']:.4f} | "
            f"{sq.get('avg_total_pixels', 0):.1f} | "
            f"{sq.get('avg_junctions', 0):.1f} | "
            f"{sq.get('avg_connectivity', 0):.4f} |"
        )
    lines.append("")

    plot_path = os.path.join(output_dir, "figures", "threshold_sensitivity.png")
    if os.path.exists(plot_path):
        lines.append("![Threshold Sensitivity](figures/threshold_sensitivity.png)\n")

    return "\n".join(lines)


def _section_cross_dataset(results):
    lines = [
        "## 5. Cross-Dataset Validation (EMNIST)\n",
        "**Dataset training**: Chars74K (natural images)",
        "**Dataset testing**: EMNIST ByClass (handwritten)\n",
        "| Model | Input | Chars74K Strict (%) | EMNIST Strict (%) | "
        "EMNIST Macro F1 | Domain Gap |",
        "|-------|-------|--------------------|--------------------|"
        "----------------|-----------|",
    ]
    for r in results:
        gap = r.get("chars74k_strict", 0) - r.get("emnist_strict", 0)
        lines.append(
            f"| {r['model_name']} | {r['input_type']} | "
            f"{r.get('chars74k_strict', 0):.2f} | "
            f"{r.get('emnist_strict', 0):.2f} | "
            f"{r.get('emnist_macro_f1', 0):.4f} | "
            f"{gap:+.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_latency(results, output_dir=None):
    lines = [
        "## 6. Latency Benchmark (Rigorous)\n",
        f"**Warmup**: {results.get('warmup', 100)} iterations",
        f"**Benchmark**: {results.get('n_runs', 1000)} individual runs\n",
        "| Model | Mean (ms) | Std (ms) | Median (ms) | 95% CI | P99 (ms) |",
        "|-------|-----------|---------|-------------|--------|----------|",
    ]
    for name, data in results.get("models", {}).items():
        ci = f"[{data['ci95_lower']:.3f}, {data['ci95_upper']:.3f}]"
        lines.append(
            f"| {name} | {data['mean_ms']:.3f} | {data['std_ms']:.3f} | "
            f"{data['median_ms']:.3f} | {ci} | {data['p99_ms']:.3f} |"
        )
    lines.append("")

    # Wilcoxon test results
    if "wilcoxon_tests" in results:
        lines.append("### Statistical Significance Test\n")
        for test in results["wilcoxon_tests"]:
            sig = "✅ Signifikan" if test["significant_005"] else "❌ Tidak signifikan"
            lines.append(
                f"- **{test['model_a']} vs {test['model_b']}**: "
                f"p={test['p_value']:.6f}, Δ={test['mean_diff_ms']:+.3f}ms "
                f"→ {sig} (α=0.05)"
            )
        lines.append("")

    if output_dir:
        plot_path = os.path.join(output_dir, "figures", "latency_distributions.png")
        if os.path.exists(plot_path):
            lines.append("![Latency Distributions](figures/latency_distributions.png)\n")

    return "\n".join(lines)


def _section_skeleton_quality(results):
    lines = [
        "## 7. Analisis Kualitas Skeleton\n",
        "| Threshold (px) | Avg Length | Avg Endpoints | Avg Junctions | "
        "Avg Loops | Connectivity | Branch Ratio |",
        "|---------------|-----------|--------------|--------------|"
        "----------|-------------|-------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['threshold']} | {r['avg_total_pixels']:.1f} | "
            f"{r['avg_endpoints']:.1f} | {r['avg_junctions']:.1f} | "
            f"{r['avg_loops']:.2f} | {r['avg_connectivity']:.4f} | "
            f"{r['avg_branch_ratio']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)
