#!/usr/bin/env python3
"""
Entry script for Step 2 (feature sanity check) and Step 3 (evaluation reproduction).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def step2_sanity_check(features_root):
    """
    Step 2: Feature sanity check for Omnivore and SlowFast.
    
    Verifies:
    - Expected feature directories exist
    - Lists example files
    - Attempts to load a sample file and show shape/dtype
    """
    print("=" * 60)
    print("Step 2: Feature Sanity Check")
    print("=" * 60)
    
    features_root = Path(features_root).expanduser().resolve()
    print(f"\nFeatures root: {features_root}")
    
    backbones = ["omnivore", "slowfast"]
    results = {}
    
    for backbone in backbones:
        print(f"\n--- Checking {backbone.upper()} ---")
        backbone_dir = features_root / "video" / backbone
        
        exists = backbone_dir.exists() and backbone_dir.is_dir()
        results[backbone] = {
            "exists": exists,
            "path": str(backbone_dir),
            "files": []
        }
        
        if exists:
            print(f"✓ Directory found: {backbone_dir}")
            
            # List .npz files
            npz_files = list(backbone_dir.glob("*.npz"))
            results[backbone]["file_count"] = len(npz_files)
            print(f"  Found {len(npz_files)} .npz files")
            
            if npz_files:
                # Show first few examples
                example_files = sorted(npz_files)[:5]
                results[backbone]["files"] = [f.name for f in example_files]
                print(f"  Example files:")
                for f in example_files:
                    print(f"    - {f.name}")
                
                # Try to load the first file
                try:
                    sample_file = example_files[0]
                    print(f"\n  Loading sample file: {sample_file.name}")
                    data = np.load(sample_file)
                    
                    # Check what keys are in the npz file
                    keys = list(data.keys())
                    print(f"  Keys in file: {keys}")
                    
                    # Load the first array (typically 'arr_0')
                    if keys:
                        arr = data[keys[0]]
                        print(f"  Shape: {arr.shape}")
                        print(f"  Dtype: {arr.dtype}")
                        print(f"  Min: {arr.min():.4f}, Max: {arr.max():.4f}, Mean: {arr.mean():.4f}")
                    else:
                        print("  Warning: No arrays found in npz file")
                    
                    data.close()
                    results[backbone]["sample_loaded"] = True
                    results[backbone]["sample_shape"] = arr.shape if keys else None
                    results[backbone]["sample_dtype"] = str(arr.dtype) if keys else None
                    
                except Exception as e:
                    print(f"  ✗ Error loading sample file: {e}")
                    results[backbone]["sample_loaded"] = False
                    results[backbone]["error"] = str(e)
            else:
                print("  ⚠ No .npz files found in directory")
                results[backbone]["file_count"] = 0
        else:
            print(f"✗ Directory NOT found: {backbone_dir}")
            results[backbone]["file_count"] = 0
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for backbone, info in results.items():
        status = "✓" if info["exists"] and info.get("file_count", 0) > 0 else "✗"
        print(f"{status} {backbone.upper()}: "
              f"exists={info['exists']}, "
              f"files={info.get('file_count', 0)}")
        if info.get("sample_loaded"):
            print(f"    Sample shape: {info.get('sample_shape')}, dtype: {info.get('sample_dtype')}")
    
    return results


def step3_evaluation(args):
    """
    Step 3: Evaluation reproduction.
    
    Forwards arguments to python -m core.evaluate using subprocess.
    """
    print("=" * 60)
    print("Step 3: Evaluation Reproduction")
    print("=" * 60)
    
    # Build the command
    cmd = [sys.executable, "-m", "core.evaluate"] + args
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    # Run the evaluation
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Evaluation failed with return code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n✗ Error running evaluation: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Entry script for Step 2 (feature sanity check) and Step 3 (evaluation reproduction)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Step 2 parser
    step2_parser = subparsers.add_parser("step2", help="Run Step 2: feature sanity check")
    step2_parser.add_argument(
        "--features_root",
        type=str,
        default=os.getenv("FEATURES_ROOT", "data"),
        help="Root directory containing features (default: 'data' or FEATURES_ROOT env var)"
    )
    
    # Step 3 parser - accepts all remaining args
    # Use parse_known_args to properly handle arguments that will be passed to core.evaluate
    step3_parser = subparsers.add_parser("step3", help="Run Step 3: evaluation reproduction")
    # Don't add any arguments here - we'll use parse_known_args to capture remaining args
    
    # Parse known args first to get the command
    args, remaining = parser.parse_known_args()
    
    if args.command == "step2":
        step2_sanity_check(args.features_root)
    elif args.command == "step3":
        if not remaining:
            print("Error: Step 3 requires arguments for core.evaluate")
            print("\nExample:")
            print("  python scripts/run.py step3 --split step --backbone omnivore --variant MLP --ckpt checkpoints/model.pt --threshold 0.6")
            sys.exit(1)
        step3_evaluation(remaining)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
