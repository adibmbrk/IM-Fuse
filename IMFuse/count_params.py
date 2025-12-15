#!/usr/bin/env python3
"""
Standalone script to count FLOPs, parameters, and measure inference time for IM-Fuse models.
Usage:
    python count_params.py [--first_skip] [--interleaved_tokenization] [--mamba_skip]
    
Metrics computed:
    - Trainable/Total parameters (M)
    - FLOPs (G) 
    - Inference time (ms) - averaged over multiple runs with warmup
"""

import argparse
import time
import torch
from thop import profile, clever_format
from IMFuse import IMFuse
from IMFuse_no1skip import Model

def count_parameters(model):
    """Count the number of trainable and total parameters in the model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def count_flops_and_params(model, input_shape=(1, 4, 128, 128, 128), mask_shape=(1, 4)):
    """
    Count FLOPs and parameters for the model
    Args:
        model: The neural network model
        input_shape: Shape of input tensor (batch_size, channels, depth, height, width)
        mask_shape: Shape of mask tensor (batch_size, num_modalities)
    Returns:
        flops: Number of FLOPs
        params: Number of parameters
    """
    device = next(model.parameters()).device
    input_tensor = torch.randn(input_shape).to(device)
    mask_tensor = torch.ones(mask_shape).bool().to(device)
    
    # Count FLOPs and parameters using thop
    flops, params = profile(model, inputs=(input_tensor, mask_tensor), verbose=False)
    
    # Format the numbers for better readability
    flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
    
    return flops, params, flops_formatted, params_formatted


def measure_inference_time(model, input_shape=(1, 4, 128, 128, 128), mask_shape=(1, 4), 
                           warmup_runs=10, num_runs=100):
    """
    Measure inference time for the model
    Args:
        model: The neural network model (should be in eval mode)
        input_shape: Shape of input tensor (batch_size, channels, depth, height, width)
        mask_shape: Shape of mask tensor (batch_size, num_modalities)
        warmup_runs: Number of warmup runs to stabilize GPU (default: 10)
        num_runs: Number of runs to average over (default: 100)
    Returns:
        avg_time_ms: Average inference time in milliseconds
        std_time_ms: Standard deviation of inference time in milliseconds
        min_time_ms: Minimum inference time in milliseconds
        max_time_ms: Maximum inference time in milliseconds
    """
    device = next(model.parameters()).device
    input_tensor = torch.randn(input_shape).to(device)
    mask_tensor = torch.ones(mask_shape).bool().to(device)
    
    model.eval()
    
    # Warmup runs to stabilize GPU performance
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor, mask_tensor)
    
    # Synchronize CUDA before timing (if using GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure inference time
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(input_tensor, mask_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    import numpy as np
    times = np.array(times)
    
    return times.mean(), times.std(), times.min(), times.max()

def main():
    parser = argparse.ArgumentParser(description='Count FLOPs, parameters, and inference time for IM-Fuse models')
    parser.add_argument('--first_skip', action='store_true', default=False,
                        help='Use IMFuse model with first skip connection')
    parser.add_argument('--interleaved_tokenization', action='store_true', default=False,
                        help='Use interleaved tokenization')
    parser.add_argument('--mamba_skip', action='store_true', default=False,
                        help='Use mamba skip connections')
    parser.add_argument('--num_cls', default=4, type=int,
                        help='Number of classes (default: 4 for BraTS)')
    parser.add_argument('--input_size', default=128, type=int,
                        help='Input size (default: 128)')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--warmup_runs', default=10, type=int,
                        help='Number of warmup runs for inference timing (default: 10)')
    parser.add_argument('--num_runs', default=100, type=int,
                        help='Number of runs to average inference time over (default: 100)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # Create model
    print("\n" + "="*70)
    print("Model Configuration")
    print("="*70)
    print(f"Model Type: {'IMFuse (with first skip)' if args.first_skip else 'IMFuse_no1skip'}")
    print(f"Number of classes: {args.num_cls}")
    print(f"Interleaved tokenization: {args.interleaved_tokenization}")
    print(f"Mamba skip connections: {args.mamba_skip}")
    print(f"Input size: {args.input_size}x{args.input_size}x{args.input_size}")
    print(f"Device: {args.device}")
    print("="*70 + "\n")
    
    if args.first_skip:
        model = IMFuse(
            num_cls=args.num_cls,
            interleaved_tokenization=args.interleaved_tokenization,
            mamba_skip=args.mamba_skip,
        )
    else:
        model = Model(
            num_cls=args.num_cls,
            interleaved_tokenization=args.interleaved_tokenization,
            mamba_skip=args.mamba_skip,
        )
    
    model = model.to(args.device)
    model.eval()
    
    # Count parameters
    trainable_params, total_params = count_parameters(model)
    
    print("="*70)
    print("Model Statistics")
    print("="*70)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Count FLOPs
    input_size = args.input_size
    try:
        flops, params, flops_formatted, params_formatted = count_flops_and_params(
            model,
            input_shape=(1, 4, input_size, input_size, input_size),
            mask_shape=(1, 4)
        )
        print(f"\nFLOPs: {flops_formatted} ({flops:,})")
        print(f"FLOPs (G): {flops / 1e9:.3f}")
        print(f"Parameters (thop count): {params_formatted} ({params:,})")
        
        # Calculate model size in MB
        param_size_mb = (total_params * 4) / (1024 ** 2)  # Assuming float32
        print(f"\nEstimated model size: {param_size_mb:.2f} MB (float32)")
        
    except Exception as e:
        print(f"\nWarning: Could not count FLOPs. Error: {e}")
    
    print("="*70 + "\n")
    
    # Measure Inference Time
    print("="*70)
    print("Inference Time Measurement")
    print("="*70)
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Timing runs: {args.num_runs}")
    print(f"Input shape: (1, 4, {input_size}, {input_size}, {input_size})")
    
    try:
        avg_time, std_time, min_time, max_time = measure_inference_time(
            model,
            input_shape=(1, 4, input_size, input_size, input_size),
            mask_shape=(1, 4),
            warmup_runs=args.warmup_runs,
            num_runs=args.num_runs
        )
        print(f"\nInference Time (ms):")
        print(f"  Average: {avg_time:.2f} ms")
        print(f"  Std Dev: {std_time:.2f} ms")
        print(f"  Min: {min_time:.2f} ms")
        print(f"  Max: {max_time:.2f} ms")
        print(f"\nThroughput: {1000/avg_time:.2f} samples/sec")
        
    except Exception as e:
        print(f"\nWarning: Could not measure inference time. Error: {e}")
    
    print("="*70 + "\n")
    
    # Test forward pass
    print("="*70)
    print("Forward Pass Test")
    print("="*70)
    try:
        with torch.no_grad():
            x = torch.randn(1, 4, input_size, input_size, input_size).to(args.device)
            mask = torch.ones(1, 4).bool().to(args.device)
            
            model.is_training = False
            output = model(x, mask)
            
            if isinstance(output, tuple):
                fuse_pred = output[0]
            else:
                fuse_pred = output
            
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {fuse_pred.shape}")
            print("✓ Forward pass successful!")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    
    print("="*70 + "\n")
    
    # Print Summary Table
    print("="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Metric':<30} {'Value':<20}")
    print("-"*50)
    print(f"{'Parameters (M)':<30} {total_params/1e6:.2f}")
    print(f"{'Trainable Parameters (M)':<30} {trainable_params/1e6:.2f}")
    try:
        print(f"{'FLOPs (G)':<30} {flops/1e9:.2f}")
    except:
        print(f"{'FLOPs (G)':<30} {'N/A'}")
    try:
        print(f"{'Inference Time (ms)':<30} {avg_time:.2f} ± {std_time:.2f}")
    except:
        print(f"{'Inference Time (ms)':<30} {'N/A'}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()



