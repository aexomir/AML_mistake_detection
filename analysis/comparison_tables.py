"""
Generate comparison tables from extracted metrics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.table import Table


def load_metrics():
    """Load extracted metrics."""
    csv_path = Path(__file__).parent / "outputs" / "comparison_metrics.csv"
    df = pd.read_csv(csv_path)
    return df


def format_metric(value, decimals=4):
    """Format metric value for display."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def create_comparison_table(df, level='Step', split=None):
    """Create a comparison table for a specific level and split."""
    # Filter by level
    filtered = df[df['Level'] == level].copy()
    
    # Filter by split if specified
    if split:
        filtered = filtered[filtered['Split'] == split]
    
    # Create a pivot table
    if split:
        index_cols = ['Model', 'Backbone']
    else:
        index_cols = ['Model', 'Backbone', 'Split']
    
    # Create tables for each metric
    metrics = ['Precision', 'Recall', 'F1', 'Accuracy', 'AUC']
    tables = {}
    
    for metric in metrics:
        pivot = filtered.pivot_table(
            values=metric,
            index=index_cols,
            aggfunc='first'
        ).reset_index()
        tables[metric] = pivot
    
    return tables, filtered


def generate_table_visualization(df, level='Step', split=None, save_path=None):
    """Generate a visual table as PNG."""
    tables, filtered = create_comparison_table(df, level, split)
    
    # Combine all metrics into one table
    if split:
        index_cols = ['Model', 'Backbone']
    else:
        index_cols = ['Model', 'Backbone', 'Split']
    
    # Merge all metrics
    combined = filtered[index_cols + ['Precision', 'Recall', 'F1', 'Accuracy', 'AUC']].copy()
    combined = combined.sort_values(['Model', 'Backbone'] + (['Split'] if not split else []))
    
    # Format values
    for col in ['Precision', 'Recall', 'F1', 'Accuracy', 'AUC']:
        combined[col] = combined[col].apply(lambda x: format_metric(x, 3))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(8, len(combined) * 0.5 + 2)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = []
    headers = list(combined.columns)
    table_data.append(headers)
    
    for _, row in combined.iterrows():
        table_data.append([str(val) for val in row.values])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows (alternating colors)
    for i in range(1, len(table_data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Title
    title = f"Model Comparison - {level} Level"
    if split:
        title += f" ({split} split)"
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved table to: {save_path}")
    
    plt.close()
    return combined


def generate_all_tables():
    """Generate all comparison tables."""
    df = load_metrics()
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    print("Generating comparison tables...")
    
    # Step-level tables
    print("  Creating Step-level tables...")
    step_all = generate_table_visualization(
        df, level='Step', split=None,
        save_path=output_dir / "comparison_table_step_all.png"
    )
    step_step = generate_table_visualization(
        df, level='Step', split='step',
        save_path=output_dir / "comparison_table_step_split.png"
    )
    step_recordings = generate_table_visualization(
        df, level='Step', split='recordings',
        save_path=output_dir / "comparison_table_step_recordings.png"
    )
    
    # Sub Step-level tables
    print("  Creating Sub Step-level tables...")
    substep_all = generate_table_visualization(
        df, level='Sub Step', split=None,
        save_path=output_dir / "comparison_table_substep_all.png"
    )
    substep_step = generate_table_visualization(
        df, level='Sub Step', split='step',
        save_path=output_dir / "comparison_table_substep_split.png"
    )
    substep_recordings = generate_table_visualization(
        df, level='Sub Step', split='recordings',
        save_path=output_dir / "comparison_table_substep_recordings.png"
    )
    
    # Save CSV tables
    print("  Saving CSV tables...")
    step_all.to_csv(output_dir / "comparison_table_step_all.csv", index=False)
    step_step.to_csv(output_dir / "comparison_table_step_split.csv", index=False)
    step_recordings.to_csv(output_dir / "comparison_table_step_recordings.csv", index=False)
    substep_all.to_csv(output_dir / "comparison_table_substep_all.csv", index=False)
    substep_step.to_csv(output_dir / "comparison_table_substep_split.csv", index=False)
    substep_recordings.to_csv(output_dir / "comparison_table_substep_recordings.csv", index=False)
    
    print("✓ All tables generated successfully!")
    
    return {
        'step_all': step_all,
        'step_step': step_step,
        'step_recordings': step_recordings,
        'substep_all': substep_all,
        'substep_step': substep_step,
        'substep_recordings': substep_recordings,
    }


if __name__ == "__main__":
    tables = generate_all_tables()

