"""
Error-type-aware analysis for SupervisedER task.

This script computes per-error-type performance metrics (Accuracy, Precision, Recall, F1, AUC)
for existing baselines without modifying core/evaluate.py.
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from base import fetch_model
from constants import Constants as const
from core.config import Config
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset, collate_fn


@dataclass
class SampleRecord:
    """Record for a single sample with predictions and metadata."""
    sample_id: int
    recording_id: str
    step_id: str
    y_true: int
    y_pred: int
    y_prob: float
    error_types: set  # Set of error type labels (0, 2, 3, 4, 5, 6)


def get_device():
    """Check if CUDA is available, return 'cuda' if available, 'cpu' otherwise."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_inference_with_metadata(
    model,
    test_loader,
    dataset: CaptainCookStepDataset,
    device: str,
    threshold: float = 0.5,
    step_normalization: bool = True,
    sub_step_normalization: bool = True,
) -> List[SampleRecord]:
    """
    Run inference and collect per-sample predictions with error type metadata.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        dataset: Dataset instance (needed to access metadata)
        device: Device to run inference on
        threshold: Threshold for binary classification
        step_normalization: Whether to apply step-level normalization
        sub_step_normalization: Whether to apply sub-step-level normalization
    
    Returns:
        List of SampleRecord objects with predictions and metadata
    """
    model.eval()
    all_records = []
    
    # Track step boundaries for normalization
    test_step_start_end_list = []
    counter = 0
    
    # Collect all outputs and targets first (similar to test_er_model)
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Running inference")):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            sigmoid_output = output.sigmoid()
            all_outputs.append(sigmoid_output.detach().cpu().numpy().reshape(-1))
            all_targets.append(target.detach().cpu().numpy().reshape(-1))
            
            test_step_start_end_list.append((counter, counter + data.shape[0]))
            counter += data.shape[0]
    
    # Flatten lists
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    # Apply step-level normalization if needed
    all_step_outputs = []
    all_step_targets = []
    
    for step_idx, (start, end) in enumerate(test_step_start_end_list):
        step_output = all_outputs[start:end]
        step_target = all_targets[start:end]
        
        # Apply sub-step normalization if needed
        if end - start > 1 and sub_step_normalization:
            prob_range = np.max(step_output) - np.min(step_output)
            if prob_range > 0:
                step_output = (step_output - np.min(step_output)) / prob_range
        
        mean_step_output = np.mean(step_output)
        step_target = 1 if np.mean(step_target) > 0.95 else 0
        
        all_step_outputs.append(mean_step_output)
        all_step_targets.append(step_target)
    
    all_step_outputs = np.array(all_step_outputs)
    
    # Apply step-level normalization if needed
    if step_normalization:
        prob_range = np.max(all_step_outputs) - np.min(all_step_outputs)
        if prob_range > 0:
            all_step_outputs = (all_step_outputs - np.min(all_step_outputs)) / prob_range
    
    all_step_targets = np.array(all_step_targets)
    
    # Now create records with error type information
    # Map dataset indices to recording_id and error types
    for step_idx, (start, end) in enumerate(test_step_start_end_list):
        # Get the dataset index for this step (batch_idx == step_idx for batch_size=1)
        if step_idx < len(dataset._step_dict):
            recording_id, step_start_end_list = dataset._step_dict[step_idx]
            
            # Get error types for this step from step_start_end_list
            # step_start_end_list is a list of tuples: (start_time, end_time, has_errors, error_category_labels)
            error_types = set()
            if step_start_end_list:
                # Get error_category_labels from the first entry (all should have the same labels)
                _, _, _, error_category_labels = step_start_end_list[0]
                error_types.update(error_category_labels)
            
            # If no errors, error_types should contain {0}
            if not error_types:
                error_types = {0}
            
            # Get step-level predictions
            step_output = all_step_outputs[step_idx]
            step_target = all_step_targets[step_idx]
            step_pred = 1 if step_output > threshold else 0
            
            # Create records for each sub-step in this step
            num_sub_steps = end - start
            for sub_step_idx in range(num_sub_steps):
                sub_step_output = all_outputs[start + sub_step_idx]
                sub_step_target = all_targets[start + sub_step_idx]
                sub_step_pred = 1 if sub_step_output > 0.5 else 0
                
                record = SampleRecord(
                    sample_id=start + sub_step_idx,
                    recording_id=recording_id,
                    step_id=f"step_{step_idx}",
                    y_true=int(sub_step_target),
                    y_pred=sub_step_pred,
                    y_prob=float(sub_step_output),
                    error_types=error_types.copy(),
                )
                all_records.append(record)
    
    return all_records


def get_error_type_name(error_label: int) -> str:
    """Convert error label to human-readable name."""
    label_to_name = {
        0: "No Error",
        2: const.PREPARATION_ERROR,
        3: const.TEMPERATURE_ERROR,
        4: const.MEASUREMENT_ERROR,
        5: const.TIMING_ERROR,
        6: const.TECHNIQUE_ERROR,
    }
    return label_to_name.get(error_label, f"Unknown ({error_label})")


def compute_metrics_per_error_type(
    records: List[SampleRecord],
    error_type_label: int,
) -> Optional[Dict[str, float]]:
    """
    Compute metrics for samples with a specific error type.
    
    Args:
        records: List of all sample records
        error_type_label: Error type label to filter by
    
    Returns:
        Dictionary of metrics or None if insufficient samples
    """
    # Filter records that have this error type
    filtered_records = [
        r for r in records
        if error_type_label in r.error_types
    ]
    
    if len(filtered_records) == 0:
        return None
    
    y_true = np.array([r.y_true for r in filtered_records])
    y_pred = np.array([r.y_pred for r in filtered_records])
    y_prob = np.array([r.y_prob for r in filtered_records])
    
    # Check if we have both classes for AUC
    unique_classes = np.unique(y_true)
    has_both_classes = len(unique_classes) == 2
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sample_count": len(filtered_records),
    }
    
    # Only compute AUC if both classes are present
    if has_both_classes:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = np.nan
    else:
        metrics["auc"] = np.nan
    
    return metrics


def analyze_error_types(
    config: Config,
    threshold: float = 0.5,
    step_normalization: bool = True,
    sub_step_normalization: bool = True,
) -> pd.DataFrame:
    """
    Main analysis function that runs inference and computes per-error-type metrics.
    
    Args:
        config: Configuration object
        threshold: Threshold for binary classification
        step_normalization: Whether to apply step-level normalization
        sub_step_normalization: Whether to apply sub-step-level normalization
    
    Returns:
        DataFrame with per-error-type metrics
    """
    print("=" * 80)
    print("Error-Type-Aware Analysis")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Split: {config.split}")
    print(f"  Backbone: {config.backbone}")
    print(f"  Variant: {config.variant}")
    print(f"  Threshold: {threshold}")
    print(f"  Step normalization: {step_normalization}")
    print(f"  Sub-step normalization: {sub_step_normalization}")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = fetch_model(config)
    model.load_state_dict(torch.load(config.ckpt_directory, map_location=config.device))
    model.to(config.device)
    model.eval()
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )
    
    # Run inference and collect records
    print("Running inference...")
    records = run_inference_with_metadata(
        model,
        test_loader,
        test_dataset,
        config.device,
        threshold,
        step_normalization,
        sub_step_normalization,
    )
    
    print(f"Collected {len(records)} sample records")
    
    # Compute global metrics for comparison
    y_true_global = np.array([r.y_true for r in records])
    y_pred_global = np.array([r.y_pred for r in records])
    y_prob_global = np.array([r.y_prob for r in records])
    
    global_metrics = {
        "error_type": "Global (All)",
        "accuracy": accuracy_score(y_true_global, y_pred_global),
        "precision": precision_score(y_true_global, y_pred_global, zero_division=0),
        "recall": recall_score(y_true_global, y_pred_global, zero_division=0),
        "f1": f1_score(y_true_global, y_pred_global, zero_division=0),
        "auc": roc_auc_score(y_true_global, y_prob_global) if len(np.unique(y_true_global)) == 2 else np.nan,
        "sample_count": len(records),
    }
    
    # Compute metrics per error type
    error_types = [0, 2, 3, 4, 5, 6]  # All possible error type labels
    results = []
    
    for error_label in error_types:
        metrics = compute_metrics_per_error_type(records, error_label)
        if metrics is not None:
            error_name = get_error_type_name(error_label)
            results.append({
                "error_type": error_name,
                "error_label": error_label,
                **metrics,
            })
    
    # Add global metrics
    results.append(global_metrics)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by sample count (descending), then by F1 (descending)
    df = df.sort_values(
        by=["sample_count", "f1"],
        ascending=[False, False],
        na_position="last",
    )
    
    return df


def save_results(df: pd.DataFrame, output_dir: str, config: Config, threshold: float):
    """Save results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    model_name = f"{config.split}_{config.backbone}_{config.variant}"
    filename_base = f"error_type_analysis_{model_name}_threshold_{threshold}"
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{filename_base}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{filename_base}.json")
    df_dict = df.to_dict(orient="records")
    with open(json_path, "w") as f:
        json.dump(df_dict, f, indent=2, default=str)
    print(f"Results saved to: {json_path}")
    
    return csv_path, json_path


def plot_results(df: pd.DataFrame, output_dir: str, config: Config, threshold: float):
    """Create visualization plots for per-error-type metrics."""
    if not HAS_PLOTTING:
        print("Matplotlib/seaborn not available, skipping visualization.")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out global row for plotting
    plot_df = df[df["error_type"] != "Global (All)"].copy()
    
    if len(plot_df) == 0:
        print("No error types to plot (excluding global).")
        return None
    
    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: F1 Score per Error Type
    ax1 = axes[0]
    plot_df_sorted = plot_df.sort_values("f1", ascending=True, na_position="last")
    colors = sns.color_palette("husl", len(plot_df_sorted))
    bars1 = ax1.barh(
        plot_df_sorted["error_type"],
        plot_df_sorted["f1"],
        color=colors,
    )
    ax1.set_xlabel("F1 Score", fontsize=12)
    ax1.set_ylabel("Error Type", fontsize=12)
    ax1.set_title("F1 Score per Error Type", fontsize=14, fontweight="bold")
    ax1.set_xlim(0, 1.0)
    ax1.grid(axis="x", alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df_sorted.iterrows()):
        if not pd.isna(row["f1"]):
            ax1.text(
                row["f1"] + 0.01,
                i,
                f"{row['f1']:.3f}",
                va="center",
                fontsize=9,
            )
    
    # Plot 2: Sample Count per Error Type
    ax2 = axes[1]
    plot_df_sorted_count = plot_df.sort_values("sample_count", ascending=True)
    bars2 = ax2.barh(
        plot_df_sorted_count["error_type"],
        plot_df_sorted_count["sample_count"],
        color=colors[:len(plot_df_sorted_count)],
    )
    ax2.set_xlabel("Sample Count", fontsize=12)
    ax2.set_ylabel("Error Type", fontsize=12)
    ax2.set_title("Sample Count per Error Type", fontsize=14, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df_sorted_count.iterrows()):
        ax2.text(
            row["sample_count"] + max(plot_df_sorted_count["sample_count"]) * 0.01,
            i,
            f"{int(row['sample_count'])}",
            va="center",
            fontsize=9,
        )
    
    plt.tight_layout()
    
    # Save figure
    model_name = f"{config.split}_{config.backbone}_{config.variant}"
    filename_base = f"error_type_analysis_{model_name}_threshold_{threshold}"
    fig_path = os.path.join(output_dir, f"{filename_base}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {fig_path}")
    
    plt.close()
    
    return fig_path


def print_results_table(df: pd.DataFrame):
    """Print a formatted results table."""
    print("\n" + "=" * 80)
    print("Per-Error-Type Performance Metrics")
    print("=" * 80)
    
    # Format the DataFrame for display
    display_df = df.copy()
    for col in ["accuracy", "precision", "recall", "f1", "auc"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
            )
    
    print(display_df.to_string(index=False))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Error-type-aware analysis for SupervisedER task"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=[const.STEP_SPLIT, const.RECORDINGS_SPLIT],
        required=True,
        help="Data split to use",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=[const.SLOWFAST, const.OMNIVORE],
        required=True,
        help="Backbone model",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=[const.MLP_VARIANT, const.TRANSFORMER_VARIANT],
        required=True,
        help="Model variant",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification (default: 0.5)",
    )
    parser.add_argument(
        "--step-normalization",
        action="store_true",
        default=True,
        help="Apply step-level normalization (default: True)",
    )
    parser.add_argument(
        "--no-step-normalization",
        dest="step_normalization",
        action="store_false",
        help="Disable step-level normalization",
    )
    parser.add_argument(
        "--sub-step-normalization",
        action="store_true",
        default=True,
        help="Apply sub-step-level normalization (default: True)",
    )
    parser.add_argument(
        "--no-sub-step-normalization",
        dest="sub_step_normalization",
        action="store_false",
        help="Disable sub-step-level normalization",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/outputs",
        help="Output directory for results (default: analysis/outputs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If not specified, auto-detects based on CUDA availability",
    )
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.split = args.split
    config.backbone = args.backbone
    config.variant = args.variant
    config.ckpt_directory = args.ckpt
    config.test_batch_size = 1
    if args.device is not None:
        config.device = args.device
    else:
        config.device = get_device()
    
    # Run analysis
    df = analyze_error_types(
        config,
        threshold=args.threshold,
        step_normalization=args.step_normalization,
        sub_step_normalization=args.sub_step_normalization,
    )
    
    # Print results
    print_results_table(df)
    
    # Save results
    save_results(df, args.output_dir, config, args.threshold)
    
    # Create visualizations
    plot_results(df, args.output_dir, config, args.threshold)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

