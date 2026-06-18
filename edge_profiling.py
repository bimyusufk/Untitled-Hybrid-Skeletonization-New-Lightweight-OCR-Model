"""
=======================================================================
PHASE 5E: ONNX EXPORT & EDGE PROFILING
=======================================================================
Exports the SuperHybridCNN model to ONNX format and profiles its latency,
throughput, and memory footprint on a CPU (simulating edge deployment constraints).
=======================================================================
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

from super_hybrid_benchmarking import (
    SuperHybridCNN, NUM_CLASSES
)

def count_flops_est(model):
    """
    Manually estimate multiply-accumulate operations (MACs) for SuperHybridCNN:
    - Conv1: 32 filters, 3x3 kernel, input size 64x64, channels=1
      MACs = 32 * (3*3*1) * 64 * 64 = 1.18 M
    - Conv2: 64 filters, 3x3 kernel, input size 32x32, input channels=32
      MACs = 64 * (3*3*32) * 32 * 32 = 18.87 M
    - Conv3: 128 filters, 3x3 kernel, input size 16x16, input channels=64
      MACs = 128 * (3*3*64) * 16 * 16 = 18.87 M
    - FC1: 128 * 8 * 8 -> 128
      MACs = (128 * 8 * 8) * 128 = 1.05 M
    - FC2: 128 + 12 -> 128
      MACs = 140 * 128 = 17.9 K
    - FC_Out: 128 -> 62
      MACs = 128 * 62 = 7.9 K
    Total MACs ~ 40.0 M
    FLOPs ~ 2 * MACs ~ 80.0 M
    """
    macs = (32 * 9 * 1 * 64 * 64) + (64 * 9 * 32 * 32 * 32) + (128 * 9 * 64 * 16 * 16) + (128 * 8 * 8 * 128) + (140 * 128) + (128 * NUM_CLASSES)
    flops = macs * 2
    return flops

def main():
    print("=" * 75)
    print("  PHASE 5E: ONNX EXPORT & EDGE PROFILING")
    print("=" * 75)
    
    output_dir = "ocr_evaluation_outputs_edge"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize Model
    model = SuperHybridCNN(num_classes=NUM_CLASSES, feat_dim=12)
    model.eval()
    
    # Check if we have trained weights to load
    weights_path = "ocr_evaluation_outputs_super_hybrid/SuperHybrid_Gradient.pth"
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print("Using initialized weights for profiling.")
        
    # 2. Export to ONNX
    dummy_img = torch.randn(1, 1, 64, 64)
    dummy_feats = torch.randn(1, 12)
    onnx_path = os.path.join(output_dir, "SuperHybrid_Gradient.onnx")
    
    print(f"Exporting model to ONNX format at {onnx_path}...")
    torch.onnx.export(
        model, 
        (dummy_img, dummy_feats), 
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['image', 'topo_features'],
        output_names=['output']
    )
    print("  [OK] Export completed.")
    
    # 3. Model Size Profile
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    est_flops = count_flops_est(model)
    
    print("\n--- Model Architecture Footprint ---")
    print(f"  Total Parameters:   {num_params:,}")
    print(f"  Exported ONNX Size: {onnx_size_mb:.4f} MB")
    print(f"  Estimated FLOPs:    {est_flops / 1e6:.2f} MFLOPs")
    
    # 4. Profile Latency (Single Image CPU Inference - simulating Edge device environment)
    print("\nProfiling CPU Latency (Single-Image Batch size = 1)...")
    warmup_iters = 50
    profile_iters = 1000
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_img, dummy_feats)
            
    # Measure
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(profile_iters):
            _ = model(dummy_img, dummy_feats)
    t_end = time.perf_counter()
    
    total_time_ms = (t_end - t_start) * 1000
    avg_latency_ms = total_time_ms / profile_iters
    throughput = profile_iters / (t_end - t_start)
    
    print("\n--- PyTorch CPU Profiling Results ---")
    print(f"  Average Latency: {avg_latency_ms:.4f} ms")
    print(f"  Throughput:      {throughput:.2f} images/sec")
    
    # 5. Check if ONNX Runtime is installed for additional profiling
    ort_latency_ms = None
    ort_throughput = None
    try:
        import onnxruntime as ort
        print("\nONNX Runtime found! Profiling ONNX model...")
        
        session = ort.InferenceSession(onnx_path)
        input_names = [i.name for i in session.get_inputs()]
        
        # Prepare inputs
        ort_inputs = {
            'image': dummy_img.numpy(),
            'topo_features': dummy_feats.numpy()
        }
        
        # Warmup
        for _ in range(warmup_iters):
            _ = session.run(None, ort_inputs)
            
        # Measure
        t_start = time.perf_counter()
        for _ in range(profile_iters):
            _ = session.run(None, ort_inputs)
        t_end = time.perf_counter()
        
        ort_total_time_ms = (t_end - t_start) * 1000
        ort_latency_ms = ort_total_time_ms / profile_iters
        ort_throughput = profile_iters / (t_end - t_start)
        
        print("\n--- ONNX Runtime CPU Profiling Results ---")
        print(f"  ONNX Latency: {ort_latency_ms:.4f} ms")
        print(f"  ONNX Throughput: {ort_throughput:.2f} images/sec")
    except ImportError:
        print("\nONNX Runtime not installed. Skipping ONNX profiling. Run 'pip install onnxruntime' to enable.")
        
    # Save Profiling Report
    report_path = os.path.join(output_dir, "edge_profiling_report.json")
    report_data = {
        "parameters": num_params,
        "onnx_size_mb": onnx_size_mb,
        "estimated_flops_m": est_flops / 1e6,
        "pytorch_cpu_latency_ms": avg_latency_ms,
        "pytorch_cpu_throughput": throughput,
        "onnxruntime_cpu_latency_ms": ort_latency_ms,
        "onnxruntime_cpu_throughput": ort_throughput
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
        
    # Text Report
    text_report_path = os.path.join(output_dir, "edge_profiling_report.txt")
    with open(text_report_path, "w") as f:
        f.write("=========================================\n")
        f.write("       EDGE COMPATIBILITY PROFILE\n")
        f.write("=========================================\n")
        f.write(f"Model Parameters:   {num_params:,}\n")
        f.write(f"Model Size (ONNX):  {onnx_size_mb:.4f} MB\n")
        f.write(f"Estimated FLOPs:    {est_flops / 1e6:.2f} MFLOPs\n")
        f.write(f"PyTorch CPU Latency: {avg_latency_ms:.4f} ms\n")
        f.write(f"PyTorch Throughput:  {throughput:.2f} imgs/sec\n")
        if ort_latency_ms:
            f.write(f"ONNX Runtime Latency: {ort_latency_ms:.4f} ms\n")
            f.write(f"ONNX Runtime Throughput: {ort_throughput:.2f} imgs/sec\n")
        f.write("=========================================\n")
        
    print(f"\n[OK] Profiling reports saved to {output_dir}/")

if __name__ == "__main__":
    main()
