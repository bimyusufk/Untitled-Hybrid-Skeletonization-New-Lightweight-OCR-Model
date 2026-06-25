# profile_latency_native.py
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import numpy as np
import pandas as pd
from thop import profile
import onnxruntime as ort

# Import TopoGradNet from our codebase
from super_hybrid_benchmarking import TopoGradNet

# --- Baseline model wrappers matching comprehensive_benchmarking.py ---
class LeNet5(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        if x.shape[2] != 32 or x.shape[3] != 32:
            x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class MobileNetV3SmallWrapper(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.base = tv_models.mobilenet_v3_small(weights=None)
        self.base.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.base.classifier[3] = nn.Linear(self.base.classifier[3].in_features, num_classes)
        
    def forward(self, x):
        return self.base(x)

class SqueezeNetWrapper(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.base = tv_models.squeezenet1_1(weights=None)
        self.base.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.base.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        
    def forward(self, x):
        return self.base(x)

class ShuffleNetV2Wrapper(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.base = tv_models.shufflenet_v2_x0_5(weights=None)
        self.base.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.base(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("=== STARTING NATIVE LATENCY PROFILING (NO TRAINING) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    configs = [
        ("LeNet-5", lambda: LeNet5(), False),
        ("MobileNetV3-Small", lambda: MobileNetV3SmallWrapper(), False),
        ("SqueezeNet", lambda: SqueezeNetWrapper(), False),
        ("ShuffleNetV2", lambda: ShuffleNetV2Wrapper(), False),
        ("TopoGrad-Net (Proposed)", lambda: TopoGradNet(feat_dim=12), True)
    ]
    
    results = []
    output_dir = "ocr_evaluation_outputs_native_linux"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model_fn, is_hybrid in configs:
        print(f"\n--- Profiling {name} ---")
        model = model_fn()
        model.eval()
        
        # Profile FLOPs & Params
        dummy_img = torch.randn(1, 1, 64, 64)
        dummy_feat = torch.randn(1, 12)
        try:
            flops, params = profile(model, inputs=(dummy_img, dummy_feat) if is_hybrid else (dummy_img,), verbose=False)
        except Exception as e:
            flops = 0.0
            params = count_parameters(model)
            print(f"Error profiling FLOPs for {name}: {e}")
            
        print(f"Params: {params:,} | FLOPs: {flops/1e6:.2f} MFLOPs")
        
        latency_profile = {}
        
        # Profile each batch size
        for bs in [1, 8, 32, 64]:
            print(f"  Batch Size: {bs}")
            dummy_batch_img = torch.randn(bs, 1, 64, 64)
            dummy_batch_feat = torch.randn(bs, 12)
            
            # --- 1. PyTorch CPU Latency ---
            model.cpu()
            with torch.no_grad():
                # Warm up
                for _ in range(10):
                    _ = model(dummy_batch_img, dummy_batch_feat) if is_hybrid else model(dummy_batch_img)
                # Profiling run
                t0 = time.perf_counter()
                for _ in range(50):
                    _ = model(dummy_batch_img, dummy_batch_feat) if is_hybrid else model(dummy_batch_img)
                t_elapsed = time.perf_counter() - t0
                cpu_latency_ms = (t_elapsed / (50 * bs)) * 1000
                cpu_throughput = (50 * bs) / t_elapsed
                
            latency_profile[f"CPU_Latency_B{bs}_ms"] = round(cpu_latency_ms, 4)
            latency_profile[f"CPU_Throughput_B{bs}_fps"] = round(cpu_throughput, 2)
            
            # --- 2. PyTorch GPU Latency (if available) ---
            if device.type == "cuda":
                model.to(device)
                img_cuda = dummy_batch_img.to(device)
                feat_cuda = dummy_batch_feat.to(device)
                with torch.no_grad():
                    # Warm up
                    for _ in range(15):
                        _ = model(img_cuda, feat_cuda) if is_hybrid else model(img_cuda)
                    torch.cuda.synchronize()
                    
                    # Profiling run
                    t0 = time.perf_counter()
                    for _ in range(100):
                        _ = model(img_cuda, feat_cuda) if is_hybrid else model(img_cuda)
                    torch.cuda.synchronize()
                    t_elapsed = time.perf_counter() - t0
                    gpu_latency_ms = (t_elapsed / (100 * bs)) * 1000
                    gpu_throughput = (100 * bs) / t_elapsed
            else:
                gpu_latency_ms = 0.0
                gpu_throughput = 0.0
                
            latency_profile[f"GPU_Latency_B{bs}_ms"] = round(gpu_latency_ms, 4)
            latency_profile[f"GPU_Throughput_B{bs}_fps"] = round(gpu_throughput, 2)
            
            # --- 3. ONNX Runtime CPU Latency ---
            model.cpu()
            onnx_path = os.path.join(output_dir, f"temp_{name.replace(' ', '_').replace('(', '').replace(')', '')}_B{bs}.onnx")
            
            try:
                # Export model to ONNX
                if is_hybrid:
                    torch.onnx.export(
                        model,
                        (dummy_batch_img, dummy_batch_feat),
                        onnx_path,
                        input_names=["img", "feat"],
                        output_names=["output"],
                        opset_version=14,
                        dynamic_axes={"img": {0: "batch_size"}, "feat": {0: "batch_size"}, "output": {0: "batch_size"}}
                    )
                else:
                    torch.onnx.export(
                        model,
                        dummy_batch_img,
                        onnx_path,
                        input_names=["img"],
                        output_names=["output"],
                        opset_version=14,
                        dynamic_axes={"img": {0: "batch_size"}, "output": {0: "batch_size"}}
                    )
                
                # Load ONNX Session (CPU)
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1  # 1 thread for standard comparative testing
                ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
                
                if is_hybrid:
                    ort_inputs = {
                        "img": dummy_batch_img.numpy(),
                        "feat": dummy_batch_feat.numpy()
                    }
                else:
                    ort_inputs = {
                        "img": dummy_batch_img.numpy()
                    }
                    
                # Warm up
                for _ in range(10):
                    _ = ort_session.run(None, ort_inputs)
                    
                # Profiling run
                t0 = time.perf_counter()
                for _ in range(100):
                    _ = ort_session.run(None, ort_inputs)
                t_elapsed = time.perf_counter() - t0
                onnx_latency_ms = (t_elapsed / (100 * bs)) * 1000
                onnx_throughput = (100 * bs) / t_elapsed
                
            except Exception as e:
                print(f"Error profiling ONNX for {name} B{bs}: {e}")
                onnx_latency_ms = 0.0
                onnx_throughput = 0.0
            finally:
                if os.path.exists(onnx_path):
                    os.remove(onnx_path)
                    
            latency_profile[f"ONNX_Latency_B{bs}_ms"] = round(onnx_latency_ms, 4)
            latency_profile[f"ONNX_Throughput_B{bs}_fps"] = round(onnx_throughput, 2)
            
        results.append({
            "Model": name,
            "Parameters": int(params),
            "FLOPs (M)": round(flops / 1e6, 2),
            **latency_profile
        })
        
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "native_latency_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults successfully saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "="*50)
    print("           LATENCY BENCHMARK SUMMARY (ms/img)")
    print("="*50)
    for res in results:
        print(f"Model: {res['Model']}")
        print(f"  Params: {res['Parameters']:,} | FLOPs: {res['FLOPs (M)']}M")
        print(f"  Batch 1  | CPU: {res['CPU_Latency_B1_ms']:.4f} ms | GPU: {res['GPU_Latency_B1_ms']:.4f} ms | ONNX CPU: {res['ONNX_Latency_B1_ms']:.4f} ms")
        print(f"  Batch 8  | CPU: {res['CPU_Latency_B8_ms']:.4f} ms | GPU: {res['GPU_Latency_B8_ms']:.4f} ms | ONNX CPU: {res['ONNX_Latency_B8_ms']:.4f} ms")
        print(f"  Batch 32 | CPU: {res['CPU_Latency_B32_ms']:.4f} ms | GPU: {res['GPU_Latency_B32_ms']:.4f} ms | ONNX CPU: {res['ONNX_Latency_B32_ms']:.4f} ms")
        print(f"  Batch 64 | CPU: {res['CPU_Latency_B64_ms']:.4f} ms | GPU: {res['GPU_Latency_B64_ms']:.4f} ms | ONNX CPU: {res['ONNX_Latency_B64_ms']:.4f} ms")
        print("-"*50)

if __name__ == "__main__":
    main()
