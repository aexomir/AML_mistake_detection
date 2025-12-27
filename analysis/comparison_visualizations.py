"""
Generate comparison visualizations (bar charts, heatmaps, etc.).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_metrics():
    """Load extracted metrics."""
    csv_path = Path(__file__).parent / "outputs" / "comparison_metrics.csv"
    df = pd.read_csv(csv_path)
    return df


def create_grouped_bar_chart(df, metric='F1', level='Step', split=None, save_path=None):
    """Create grouped bar chart comparing models."""
    # Filter data
    filtered = df[df['Level'] == level].copy()
    if split:
        filtered = filtered[filtered['Split'] == split]
    
    # Create model-backbone identifier
    filtered['Model_Backbone'] = filtered['Model'] + ' + ' + filtered['Backbone'].str.capitalize()
    
    # Group by split if not specified
    if not split:
        filtered['Split'] = filtered['Split'].fillna('N/A')
        x_col = 'Split'
        hue_col = 'Model_Backbone'
    else:
        x_col = 'Model_Backbone'
        hue_col = None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if hue_col:
        # Grouped bar chart
        unique_models = filtered['Model_Backbone'].unique()
        unique_splits = filtered['Split'].unique() if not split else [split]
        
        x = np.arange(len(unique_splits))
        width = 0.8 / len(unique_models)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_models)))
        
        for i, model in enumerate(unique_models):
            model_data = filtered[filtered['Model_Backbone'] == model]
            values = []
            for s in unique_splits:
                split_data = model_data[model_data['Split'] == s]
                if len(split_data) > 0:
                    values.append(split_data[metric].iloc[0])
                else:
                    values.append(0)
            
            offset = width * (i - len(unique_models) / 2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Split', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(unique_splits)
    else:
        # Simple bar chart
        filtered = filtered.sort_values(metric, ascending=False)
        bars = ax.bar(range(len(filtered)), filtered[metric], alpha=0.8, edgecolor='black')
        
        # Color by model
        model_colors = {'MLP': '#1f77b4', 'Transformer': '#ff7f0e', 'RNN': '#2ca02c'}
        for i, (_, row) in enumerate(filtered.iterrows()):
            bars[i].set_color(model_colors.get(row['Model'], '#d62728'))
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, filtered[metric])):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model + Backbone', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(filtered)))
        ax.set_xticklabels(filtered['Model_Backbone'], rotation=45, ha='right')
    
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Comparison - {level} Level' + (f' ({split} split)' if split else ''), 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(filtered[metric].max() * 1.2, 0.1)])
    
    if hue_col:
        ax.legend(title='Model + Backbone', fontsize=9, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def create_heatmap(df, level='Step', split=None, save_path=None):
    """Create heatmap showing all metrics."""
    # Filter data
    filtered = df[df['Level'] == level].copy()
    if split:
        filtered = filtered[filtered['Split'] == split]
    
    # Create model-backbone identifier
    filtered['Model_Backbone'] = filtered['Model'] + ' + ' + filtered['Backbone'].str.capitalize()
    
    # Select metrics
    metrics = ['Precision', 'Recall', 'F1', 'Accuracy', 'AUC']
    
    # Create pivot table
    if split:
        index_col = 'Model_Backbone'
    else:
        filtered['Split'] = filtered['Split'].fillna('N/A')
        index_col = ['Model_Backbone', 'Split']
    
    # Prepare data for heatmap
    heatmap_data = []
    row_labels = []
    
    for _, row in filtered.iterrows():
        if split:
            row_label = row['Model_Backbone']
        else:
            row_label = f"{row['Model_Backbone']} ({row['Split']})"
        
        if row_label not in row_labels:
            row_labels.append(row_label)
            values = [row[m] if not pd.isna(row[m]) else 0 for m in metrics]
            heatmap_data.append(values)
    
    heatmap_array = np.array(heatmap_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(row_labels) * 0.5)))
    
    # Create heatmap
    im = ax.imshow(heatmap_array, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(row_labels)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{heatmap_array[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20)
    
    ax.set_title(f'Metrics Heatmap - {level} Level' + (f' ({split} split)' if split else ''),
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def create_multi_metric_comparison(df, level='Step', split=None, save_path=None):
    """Create subplot comparing multiple metrics."""
    # Filter data
    filtered = df[df['Level'] == level].copy()
    if split:
        filtered = filtered[filtered['Split'] == split]
    
    # Create model-backbone identifier
    filtered['Model_Backbone'] = filtered['Model'] + ' + ' + filtered['Backbone'].str.capitalize()
    
    # Select metrics
    metrics = ['Precision', 'Recall', 'F1', 'AUC']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Multi-Metric Comparison - {level} Level' + (f' ({split} split)' if split else ''),
                 fontsize=16, fontweight='bold')
    
    # Sort by F1 for consistency
    if split:
        filtered = filtered.sort_values('F1', ascending=False)
    else:
        # Group by model-backbone and get average F1
        filtered['Avg_F1'] = filtered.groupby('Model_Backbone')['F1'].transform('mean')
        filtered = filtered.sort_values('Avg_F1', ascending=False)
    
    unique_models = filtered['Model_Backbone'].unique()
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Get values for each model
        values = []
        labels = []
        
        if split:
            for model in unique_models:
                model_data = filtered[filtered['Model_Backbone'] == model]
                if len(model_data) > 0:
                    values.append(model_data[metric].iloc[0])
                    labels.append(model)
        else:
            # Average across splits
            for model in unique_models:
                model_data = filtered[filtered['Model_Backbone'] == model]
                if len(model_data) > 0:
                    values.append(model_data[metric].mean())
                    labels.append(model)
        
        # Create bars
        bars = ax.bar(range(len(labels)), values, alpha=0.8, edgecolor='black')
        
        # Color by model type
        model_colors = {'MLP': '#1f77b4', 'Transformer': '#ff7f0e', 'RNN': '#2ca02c'}
        for i, label in enumerate(labels):
            model_type = label.split(' + ')[0]
            bars[i].set_color(model_colors.get(model_type, '#d62728'))
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, max(max(values) * 1.2, 0.1) if values else 0.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def create_backbone_comparison(df, level='Step', split=None, save_path=None):
    """Create comparison grouped by backbone."""
    # Filter data
    filtered = df[df['Level'] == level].copy()
    if split:
        filtered = filtered[filtered['Split'] == split]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Backbone Comparison - {level} Level' + (f' ({split} split)' if split else ''),
                 fontsize=14, fontweight='bold')
    
    backbones = ['omnivore', 'slowfast']
    metrics = ['F1', 'AUC']
    
    for idx, (backbone, metric) in enumerate(zip(backbones, metrics)):
        ax = axes[idx]
        
        backbone_data = filtered[filtered['Backbone'] == backbone]
        
        if len(backbone_data) > 0:
            models = backbone_data['Model'].unique()
            values = []
            labels = []
            
            for model in models:
                model_data = backbone_data[backbone_data['Model'] == model]
                if len(model_data) > 0:
                    if split:
                        values.append(model_data[metric].iloc[0])
                    else:
                        values.append(model_data[metric].mean())
                    labels.append(model)
            
            bars = ax.bar(range(len(labels)), values, alpha=0.8, edgecolor='black')
            
            # Color by model
            model_colors = {'MLP': '#1f77b4', 'Transformer': '#ff7f0e', 'RNN': '#2ca02c'}
            for i, label in enumerate(labels):
                bars[i].set_color(model_colors.get(label, '#d62728'))
            
            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{backbone.capitalize()} - {metric}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim([0, max(max(values) * 1.2, 0.1) if values else 0.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def generate_all_visualizations():
    """Generate all visualizations."""
    df = load_metrics()
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    print("Generating visualizations...")
    
    # For each level and split combination
    for level in ['Step', 'Sub Step']:
        for split in [None, 'step', 'recordings']:
            split_name = split if split else 'all'
            
            print(f"  Creating {level} level, {split_name} split visualizations...")
            
            # F1 comparison
            create_grouped_bar_chart(
                df, metric='F1', level=level, split=split,
                save_path=output_dir / f"chart_f1_{level.lower().replace(' ', '_')}_{split_name}.png"
            )
            
            # AUC comparison
            create_grouped_bar_chart(
                df, metric='AUC', level=level, split=split,
                save_path=output_dir / f"chart_auc_{level.lower().replace(' ', '_')}_{split_name}.png"
            )
            
            # Heatmap
            create_heatmap(
                df, level=level, split=split,
                save_path=output_dir / f"heatmap_{level.lower().replace(' ', '_')}_{split_name}.png"
            )
            
            # Multi-metric comparison
            create_multi_metric_comparison(
                df, level=level, split=split,
                save_path=output_dir / f"multimetric_{level.lower().replace(' ', '_')}_{split_name}.png"
            )
    
    # Backbone comparison (only for recordings split where we have all models)
    print("  Creating backbone comparison...")
    create_backbone_comparison(
        df, level='Step', split='recordings',
        save_path=output_dir / "backbone_comparison_step_recordings.png"
    )
    create_backbone_comparison(
        df, level='Sub Step', split='recordings',
        save_path=output_dir / "backbone_comparison_substep_recordings.png"
    )
    
    print("✓ All visualizations generated successfully!")


if __name__ == "__main__":
    generate_all_visualizations()

