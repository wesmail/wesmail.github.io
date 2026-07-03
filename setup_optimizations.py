"""
Setup script to install optional dependencies for maximum performance.
"""

import subprocess
import sys


def install_package(package_name, pip_name=None):
    """Install a package using pip."""
    if pip_name is None:
        pip_name = package_name
    
    try:
        __import__(package_name)
        print(f"✓ {package_name} is already installed")
        return True
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✓ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package_name}")
            return False


def main():
    print("Setting up optimizations for JetClass dataloader...")
    print("=" * 60)
    
    # Required packages
    print("\nChecking required packages:")
    required = [
        ("h5py", "h5py"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("lightning", "lightning"),
    ]
    
    for package, pip_name in required:
        if not install_package(package, pip_name):
            print(f"ERROR: Required package {package} could not be installed!")
            sys.exit(1)
    
    # Optional optimizations
    print("\nInstalling optional optimizations:")
    optional = [
        ("numba", "numba"),  # For JIT compilation
        ("psutil", "psutil"),  # For memory monitoring
    ]
    
    for package, pip_name in optional:
        install_package(package, pip_name)
    
    # Performance tips
    print("\n" + "=" * 60)
    print("Performance Optimization Tips:")
    print("=" * 60)
    print("\n1. HDF5 File Optimization:")
    print("   - Use chunked storage in HDF5 files")
    print("   - Enable compression (gzip or lzf)")
    print("   - Align chunks with your typical access patterns")
    print("   - Example HDF5 creation with optimal settings:")
    print("""
   import h5py
   with h5py.File('data.h5', 'w') as f:
       # Create chunked, compressed datasets
       f.create_dataset('feature_matrix', 
                       shape=(n_samples, max_particles, n_features),
                       chunks=(32, max_particles, n_features),
                       compression='gzip',
                       compression_opts=4)
   """)
    
    print("\n2. System-level optimizations:")
    print("   - Use SSD/NVMe storage for HDF5 files")
    print("   - Increase system file handles: ulimit -n 65536")
    print("   - Use RAID 0 for multiple HDF5 files")
    print("   - Enable huge pages for large memory allocations")
    
    print("\n3. PyTorch DataLoader settings:")
    print("   - num_workers: 2-4x number of CPU cores")
    print("   - pin_memory=True for GPU training")
    print("   - persistent_workers=True to avoid worker restart overhead")
    print("   - prefetch_factor=2-4 for better GPU utilization")
    
    print("\n4. Environment variables for better performance:")
    print("   export OMP_NUM_THREADS=1  # Prevent thread oversubscription")
    print("   export CUDA_LAUNCH_BLOCKING=0  # Enable async GPU operations")
    print("   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Better GPU memory management")
    
    print("\n" + "=" * 60)
    print("Setup complete! Your optimized dataloader is ready to use.")
    print("=" * 60)


if __name__ == "__main__":
    main()