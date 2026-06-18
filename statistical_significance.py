"""
=======================================================================
PHASE 4: STATISTICAL SIGNIFICANCE TESTING
=======================================================================
This script runs the models across multiple random seeds to confirm that
the difference in performance is statistically significant.
It uses Wilcoxon signed-rank test and paired student t-test to report p-values.
=======================================================================
"""

import os
import time
import random
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon, ttest_rel

from super_hybrid_benchmarking import (
    CLASS_LIST, NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR, IMAGE_SIZE,
    SuperDataset, SuperHybridCNN, ShallowCNNHybrid, Proposed1MModel,
    count_parameters, verify_model_forward, train_model, evaluate_model,
    train_test_split, load_binary_dataset
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

def main():
    print("=" * 75)
    print("  PHASE 4: STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load dataset
    try:
        X_bin, y = load_binary_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating mock dataset for verification...")
        X_bin = np.random.choice([0, 255], size=(200, 64, 64)).astype(np.uint8)
        y = np.random.randint(0, NUM_CLASSES, size=(200,))

    # We use 5 seeds as specified in the plan
    seeds = [42, 123, 456, 789, 1024]
    
    epochs = int(os.getenv("OCR_EPOCHS", "50"))
    patience = 10
    batch_size = 64
    output_dir = "ocr_evaluation_outputs_statistical"
    os.makedirs(output_dir, exist_ok=True)
    
    is_dry_run = os.getenv("DRY_RUN", "False").lower() == "true"
    if is_dry_run:
        print("\n[DRY RUN] Restricting seeds and sizes.")
        seeds = [42, 123]  # Wilcoxon needs at least 2, though 5 is standard
        epochs = 2
        patience = 2
        X_bin = X_bin[:128]
        y = y[:128]
        batch_size = 32

    # Results table: Model -> List of accuracies for each seed
    acc_results = {
        "SuperHybrid_Gradient": [],
        "SuperHybrid_Binary": [],
        "Gradient_CNN_Hybrid_Baseline": [],
        "Proposed_1M_Raw_Baseline": []
    }
    
    latencies = {k: [] for k in acc_results.keys()}
    num_workers = int(os.getenv("NUM_WORKERS", "4"))

    for seed in seeds:
        print(f"\n\n====================== RUNNING FOR SEED: {seed} ======================")
        set_seed(seed)
        
        # Split Data
        indices = np.arange(len(y))
        train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=seed, stratify=y)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed, stratify=y[temp_idx])
        
        X_train, y_train = X_bin[train_idx], y[train_idx]
        X_val, y_val = X_bin[val_idx], y[val_idx]
        X_test, y_test = X_bin[test_idx], y[test_idx]
        
        # Dataloaders
        train_loader_super_bin = DataLoader(SuperDataset(X_train, y_train, is_training=True, use_gradient=False, is_super_hybrid=True),
                                            batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader_super_bin = DataLoader(SuperDataset(X_val, y_val, is_training=False, use_gradient=False, is_super_hybrid=True),
                                          batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader_super_bin = DataLoader(SuperDataset(X_test, y_test, is_training=False, use_gradient=False, is_super_hybrid=True),
                                           batch_size=batch_size, shuffle=False, num_workers=num_workers)
                                           
        train_loader_super_grad = DataLoader(SuperDataset(X_train, y_train, is_training=True, use_gradient=True, is_super_hybrid=True),
                                             batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader_super_grad = DataLoader(SuperDataset(X_val, y_val, is_training=False, use_gradient=True, is_super_hybrid=True),
                                           batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader_super_grad = DataLoader(SuperDataset(X_test, y_test, is_training=False, use_gradient=True, is_super_hybrid=True),
                                            batch_size=batch_size, shuffle=False, num_workers=num_workers)
                                            
        train_loader_std_grad = DataLoader(SuperDataset(X_train, y_train, is_training=False, use_gradient=True, is_super_hybrid=False),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader_std_grad = DataLoader(SuperDataset(X_val, y_val, is_training=False, use_gradient=True, is_super_hybrid=False),
                                         batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader_std_grad = DataLoader(SuperDataset(X_test, y_test, is_training=False, use_gradient=True, is_super_hybrid=False),
                                          batch_size=batch_size, shuffle=False, num_workers=num_workers)
                                          
        train_loader_raw = DataLoader(SuperDataset(X_train, y_train, is_training=False, use_gradient=False, is_super_hybrid=False),
                                      batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader_raw = DataLoader(SuperDataset(X_val, y_val, is_training=False, use_gradient=False, is_super_hybrid=False),
                                    batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader_raw = DataLoader(SuperDataset(X_test, y_test, is_training=False, use_gradient=False, is_super_hybrid=False),
                                     batch_size=batch_size, shuffle=False, num_workers=num_workers)
                                     
        configs = [
            ("SuperHybrid_Gradient", lambda: SuperHybridCNN(NUM_CLASSES, feat_dim=12), 
             train_loader_super_grad, val_loader_super_grad, test_loader_super_grad, True),
             
            ("SuperHybrid_Binary", lambda: SuperHybridCNN(NUM_CLASSES, feat_dim=12), 
             train_loader_super_bin, val_loader_super_bin, test_loader_super_bin, True),
             
            ("Gradient_CNN_Hybrid_Baseline", lambda: ShallowCNNHybrid(NUM_CLASSES, feat_dim=5), 
             train_loader_std_grad, val_loader_std_grad, test_loader_std_grad, True),
             
            ("Proposed_1M_Raw_Baseline", lambda: Proposed1MModel(NUM_CLASSES), 
             train_loader_raw, val_loader_raw, test_loader_raw, False)
        ]
        
        for name, model_fn, train_ldr, val_ldr, test_ldr, is_hybrid in configs:
            model = model_fn()
            
            # Train
            train_model(
                model=model,
                train_loader=train_ldr,
                val_loader=val_ldr,
                epochs=epochs,
                device=device,
                model_name=f"{name}_seed_{seed}",
                patience=patience,
                is_hybrid=is_hybrid
            )
            
            # Evaluate
            metrics = evaluate_model(
                model=model,
                test_loader=test_ldr,
                device=device,
                model_name=f"{name}_seed_{seed}",
                is_hybrid=is_hybrid,
                output_dir=output_dir
            )
            
            acc_results[name].append(metrics["strict_accuracy"])
            latencies[name].append(metrics["avg_latency_ms"])

    # =====================================================================
    # STATISTICAL ANALYSIS
    # =====================================================================
    print(f"\n\n{'='*75}")
    print("                       STATISTICAL COMPARISON")
    print(f"{'='*75}")
    
    summary_stats = []
    
    proposed_key = "SuperHybrid_Gradient"
    proposed_accs = np.array(acc_results[proposed_key])
    
    for name, accs in acc_results.items():
        accs_np = np.array(accs)
        mean_acc = np.mean(accs_np)
        std_acc = np.std(accs_np)
        
        # Wilcoxon and Paired t-test relative to SuperHybrid_Gradient
        if name != proposed_key:
            # We wrap in try-except in case differences are identical (e.g. in dry run or zero variance)
            try:
                # Wilcoxon signed-rank
                w_stat, w_p = wilcoxon(proposed_accs, accs_np)
            except Exception as e:
                w_stat, w_p = None, 1.0
                
            try:
                # Paired t-test
                t_stat, t_p = ttest_rel(proposed_accs, accs_np)
            except Exception as e:
                t_stat, t_p = None, 1.0
        else:
            w_p = "-"
            t_p = "-"
            
        summary_stats.append({
            "Model": name,
            "Accuracy (Seed Runs)": ", ".join([f"{a:.2f}%" for a in accs]),
            "Mean ± Std": f"{mean_acc:.2f} ± {std_acc:.2f}%",
            "Wilcoxon p-val": f"{w_p:.4f}" if isinstance(w_p, float) else w_p,
            "Paired T-test p-val": f"{t_p:.4f}" if isinstance(t_p, float) else t_p
        })
        
    summary_df = pd.DataFrame(summary_stats)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(os.path.join(output_dir, "statistical_summary.csv"), index=False)
    
    # Save Report
    report_md_path = os.path.join(output_dir, "statistical_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Laporan Analisis Signifikansi Statistik (Statistical Significance Report)\n")
        f.write("## Verifikasi Keunggulan SuperHybrid_Gradient Terhadap Model Baseline Melalui Multi-Seed Runs\n\n")
        f.write("### Hasil Perbandingan Komparatif\n\n")
        f.write(summary_df.to_markdown(index=False) + "\n\n")
        f.write("### Metodologi Analisis Statistik\n")
        f.write(f"- **Jumlah Seeds**: {len(seeds)} (Seeds: {seeds})\n")
        f.write("- **Uji Statistik**: Wilcoxon signed-rank test (Uji non-parametrik berpasangan) dan Paired t-test (Uji parametrik berpasangan).\n")
        f.write("- **Tingkat Signifikansi (Alpha)**: 0.05. Jika p-value < 0.05, perbedaan performa dianggap signifikan secara statistik.\n")
        
    with open(os.path.join(output_dir, "statistical_results.json"), "w") as f:
        json.dump({
            "accuracies": acc_results,
            "latencies": latencies
        }, f, indent=2)
        
    print(f"\n[OK] Statistical significance reports saved to: {output_dir}/")

if __name__ == "__main__":
    main()
