"""
Benchmark script to compare original vs optimized dataloader performance.
"""

import time
import torch
import numpy as np
from typing import Dict, List
import psutil
import os


def benchmark_dataloader(dataloader, num_batches: int = 100, warmup_batches: int = 5) -> Dict:
    """
    Benchmark a dataloader's performance.
    
    Returns:
        Dictionary with timing and throughput statistics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Warmup
    print(f"Warming up with {warmup_batches} batches...")
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches:
            break
        # Move to device to simulate real training
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
    
    # Benchmark
    print(f"Benchmarking {num_batches} batches...")
    batch_times = []
    samples_per_batch = []
    memory_usage = []
    
    # Get initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        # Move to device
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
        
        # Record batch size
        if isinstance(batch, dict) and 'labels' in batch:
            samples_per_batch.append(batch['labels'].shape[0])
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Record memory usage periodically
        if i % 10 == 0:
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_usage.append(current_memory - initial_memory)
        
        # Progress
        if (i + 1) % 20 == 0:
            avg_time = np.mean(batch_times[-20:])
            print(f"  Batch {i+1}/{num_batches}: {avg_time*1000:.2f}ms/batch")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    batch_times = np.array(batch_times)
    total_samples = sum(samples_per_batch)
    
    stats = {
        'total_time': total_time,
        'num_batches': len(batch_times),
        'total_samples': total_samples,
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'min_batch_time': np.min(batch_times),
        'max_batch_time': np.max(batch_times),
        'p50_batch_time': np.percentile(batch_times, 50),
        'p90_batch_time': np.percentile(batch_times, 90),
        'p99_batch_time': np.percentile(batch_times, 99),
        'samples_per_second': total_samples / total_time,
        'avg_memory_increase_mb': np.mean(memory_usage) if memory_usage else 0,
        'max_memory_increase_mb': np.max(memory_usage) if memory_usage else 0,
    }
    
    return stats


def print_benchmark_results(name: str, stats: Dict):
    """Pretty print benchmark results."""
    print(f"\n{'='*60}")
    print(f"Benchmark Results: {name}")
    print(f"{'='*60}")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Throughput: {stats['samples_per_second']:.2f} samples/second")
    print(f"\nBatch timing (ms):")
    print(f"  Average: {stats['avg_batch_time']*1000:.2f} ± {stats['std_batch_time']*1000:.2f}")
    print(f"  Min: {stats['min_batch_time']*1000:.2f}")
    print(f"  Max: {stats['max_batch_time']*1000:.2f}")
    print(f"  P50: {stats['p50_batch_time']*1000:.2f}")
    print(f"  P90: {stats['p90_batch_time']*1000:.2f}")
    print(f"  P99: {stats['p99_batch_time']*1000:.2f}")
    print(f"\nMemory usage:")
    print(f"  Average increase: {stats['avg_memory_increase_mb']:.1f} MB")
    print(f"  Max increase: {stats['max_memory_increase_mb']:.1f} MB")
    print(f"{'='*60}\n")


def compare_dataloaders(
    original_dataloader,
    optimized_dataloader,
    num_batches: int = 100,
    warmup_batches: int = 5
):
    """Compare performance between original and optimized dataloaders."""
    print("Benchmarking original dataloader...")
    original_stats = benchmark_dataloader(original_dataloader, num_batches, warmup_batches)
    
    print("\nBenchmarking optimized dataloader...")
    optimized_stats = benchmark_dataloader(optimized_dataloader, num_batches, warmup_batches)
    
    # Print results
    print_benchmark_results("Original DataLoader", original_stats)
    print_benchmark_results("Optimized DataLoader", optimized_stats)
    
    # Calculate improvements
    speedup = original_stats['avg_batch_time'] / optimized_stats['avg_batch_time']
    throughput_improvement = optimized_stats['samples_per_second'] / original_stats['samples_per_second']
    
    print(f"\n{'='*60}")
    print("Performance Improvements:")
    print(f"{'='*60}")
    print(f"Batch loading speedup: {speedup:.2f}x faster")
    print(f"Throughput improvement: {throughput_improvement:.2f}x")
    print(f"Time saved per epoch: {(original_stats['total_time'] - optimized_stats['total_time']):.2f}s")
    print(f"Memory efficiency: {(original_stats['avg_memory_increase_mb'] - optimized_stats['avg_memory_increase_mb']):.1f} MB saved")
    print(f"{'='*60}\n")
    
    return original_stats, optimized_stats


if __name__ == "__main__":
    # Example usage
    print("Import your original and optimized data modules and run:")
    print("compare_dataloaders(original_dm.train_dataloader(), optimized_dm.train_dataloader())")
    
    # You can also test specific optimizations
    print("\nTo test specific optimizations, you can create variations:")
    print("- Test with/without memory mapping")
    print("- Test with different cache sizes")
    print("- Test with/without JIT compilation")
    print("- Test with different numbers of workers")