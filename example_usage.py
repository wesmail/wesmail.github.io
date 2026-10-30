"""
Example usage of the optimized JetClass dataloader.
Shows how to migrate from the original to the optimized version.
"""

import torch
from optimized_jetclass_dataloader import OptimizedJetClassLightningDataModule
from benchmark_dataloader import compare_dataloaders


def main():
    # Example file paths - replace with your actual HDF5 files
    train_files = [
        "/path/to/train_file1.h5",
        "/path/to/train_file2.h5",
        # Can also use glob patterns:
        # "/data/jetclass/train_*.h5"
    ]
    val_files = ["/path/to/val_*.h5"]
    test_files = ["/path/to/test_*.h5"]
    
    # Create optimized data module with recommended settings
    optimized_dm = OptimizedJetClassLightningDataModule(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        batch_size=512,  # Larger batch size for better GPU utilization
        num_workers=16,  # 2-4x CPU cores
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        cache_size_mb=4096,  # 4GB cache per dataset
        use_memory_map=True,  # Enable for faster reads
        reconstruct_full_adjacency=True,
    )
    
    # Optional: Add data transforms
    def normalize_features(sample):
        """Example transform to normalize features."""
        # Normalize node features
        sample['node_features'] = (sample['node_features'] - 0.5) / 0.5
        return sample
    
    optimized_dm.train_transform = normalize_features
    
    # Setup the data module
    optimized_dm.setup(stage='fit')
    
    # Get dataloaders
    train_loader = optimized_dm.train_dataloader()
    val_loader = optimized_dm.val_dataloader()
    
    print(f"Training dataset size: {len(optimized_dm.train_dataset):,}")
    print(f"Validation dataset size: {len(optimized_dm.val_dataset):,}")
    print(f"Batch size: {optimized_dm.batch_size}")
    print(f"Training batches per epoch: {len(train_loader):,}")
    
    # Example: Iterate through a few batches
    print("\nTesting dataloader...")
    for i, batch in enumerate(train_loader):
        if i >= 3:  # Just test a few batches
            break
        
        print(f"\nBatch {i}:")
        print(f"  Node features shape: {batch['node_features'].shape}")
        print(f"  Edge features shape: {batch['edge_features'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Num particles: {batch['n_particles'][:5].tolist()}...")
    
    # Check cache statistics
    if hasattr(optimized_dm.train_dataset, 'get_cache_stats'):
        stats = optimized_dm.train_dataset.get_cache_stats()
        print(f"\nCache statistics:")
        print(f"  Hit rate: {stats['hit_rate']:.2%}")
        print(f"  Cache size: {stats['cache_size_mb']:.1f} MB")
        print(f"  Cached chunks: {stats['num_cached_chunks']}")
    
    return optimized_dm


def benchmark_example():
    """Example of benchmarking the optimized dataloader."""
    # Create both original and optimized dataloaders
    from your_original_module import JetClassLightningDataModule  # Import your original
    
    # Common settings
    train_files = ["/path/to/train_*.h5"]
    val_files = ["/path/to/val_*.h5"]
    test_files = ["/path/to/test_*.h5"]
    
    # Original dataloader
    original_dm = JetClassLightningDataModule(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        batch_size=256,
        num_workers=8,
    )
    original_dm.setup(stage='fit')
    
    # Optimized dataloader
    optimized_dm = OptimizedJetClassLightningDataModule(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        batch_size=256,
        num_workers=8,
        cache_size_mb=4096,
        use_memory_map=True,
    )
    optimized_dm.setup(stage='fit')
    
    # Compare performance
    compare_dataloaders(
        original_dm.train_dataloader(),
        optimized_dm.train_dataloader(),
        num_batches=100,
        warmup_batches=5
    )


def advanced_tuning_example():
    """Example of advanced performance tuning."""
    import os
    
    # Set environment variables for optimal performance
    os.environ['OMP_NUM_THREADS'] = '1'  # Prevent thread oversubscription
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async GPU operations
    
    # Create dataloader with aggressive optimization settings
    dm = OptimizedJetClassLightningDataModule(
        train_files=["/path/to/train_*.h5"],
        val_files=["/path/to/val_*.h5"],
        test_files=["/path/to/test_*.h5"],
        batch_size=1024,  # Large batch size
        num_workers=32,  # Many workers
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,  # Aggressive prefetching
        cache_size_mb=8192,  # 8GB cache
        use_memory_map=True,
    )
    
    # For distributed training with DDP
    # Set different cache sizes per rank to avoid memory issues
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        # Reduce cache size for multi-GPU to avoid OOM
        dm.cache_size_mb = 8192 // world_size
    
    return dm


def migration_guide():
    """
    Migration guide from original to optimized dataloader.
    
    The optimized dataloader is designed to be a drop-in replacement
    with the same interface but better performance.
    
    Key differences:
    1. Additional parameters for performance tuning
    2. Built-in caching and prefetching
    3. Optional JIT compilation (requires numba)
    4. Memory-mapped file access option
    
    Migration steps:
    1. Replace import:
       from jetclass_dataloader import JetClassLightningDataModule
       → from optimized_jetclass_dataloader import OptimizedJetClassLightningDataModule
    
    2. Add optimization parameters:
       dm = OptimizedJetClassLightningDataModule(
           # ... existing parameters ...
           cache_size_mb=4096,  # Add cache
           use_memory_map=True,  # Enable memory mapping
       )
    
    3. No changes needed to training code!
    """
    print(migration_guide.__doc__)


if __name__ == "__main__":
    # Run the example
    print("Running optimized JetClass dataloader example...")
    dm = main()
    
    print("\n" + "="*60)
    print("Migration guide:")
    print("="*60)
    migration_guide()