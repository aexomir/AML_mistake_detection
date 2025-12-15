#!/usr/bin/env python3
"""
Installation script for project dependencies.
Supports both CUDA and CPU-only installations.
"""
import argparse
import subprocess
import sys


def install_cuda_deps():
    """Install CUDA-enabled dependencies."""
    print("Installing CUDA-enabled dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements-cuda.txt"
        ])
        print("✓ CUDA dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing CUDA dependencies: {e}")
        return False


def install_cpu_deps():
    """Install CPU-only dependencies."""
    print("Installing CPU-only dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements-cpu.txt"
        ])
        print("✓ CPU dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing CPU dependencies: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Install project dependencies with optional CUDA support"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Install CUDA-enabled PyTorch and dependencies"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Install CPU-only PyTorch and dependencies (default if --cuda not specified)"
    )
    
    args = parser.parse_args()
    
    if args.cuda:
        success = install_cuda_deps()
    else:
        success = install_cpu_deps()
    
    if not success:
        sys.exit(1)
    
    print("\nInstallation complete!")
    print("Note: The project will automatically detect CUDA availability at runtime.")


if __name__ == "__main__":
    main()

