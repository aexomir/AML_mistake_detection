"""
Extract evaluation metrics from notebook files.
"""
import json
import re
import pandas as pd
from pathlib import Path


def extract_metrics_from_output(output_text):
    """Extract metrics from evaluation output text."""
    metrics = {}
    
    # Pattern to match metric dictionaries
    pattern = r"test (Sub Step|Step) Level Metrics: ({[^}]+})"
    matches = re.findall(pattern, output_text)
    
    for level, metrics_dict_str in matches:
        level_name = "Sub Step" if "Sub Step" in level else "Step"
        
        # Extract individual metrics
        precision = re.search(r"'precision':\s*([\d.]+)", metrics_dict_str)
        recall = re.search(r"'recall':\s*([\d.]+)", metrics_dict_str)
        f1 = re.search(r"'f1':\s*([\d.]+)", metrics_dict_str)
        accuracy = re.search(r"'accuracy':\s*([\d.]+)", metrics_dict_str)
        auc = re.search(r"'auc':\s*np\.float64\(([\d.]+)\)", metrics_dict_str)
        pr_auc = re.search(r"'pr_auc':\s*tensor\(([\d.]+)\)", metrics_dict_str)
        
        if level_name not in metrics:
            metrics[level_name] = {}
        
        if precision:
            metrics[level_name]['precision'] = float(precision.group(1))
        if recall:
            metrics[level_name]['recall'] = float(recall.group(1))
        if f1:
            metrics[level_name]['f1'] = float(f1.group(1))
        if accuracy:
            metrics[level_name]['accuracy'] = float(accuracy.group(1))
        if auc:
            metrics[level_name]['auc'] = float(auc.group(1))
        if pr_auc:
            metrics[level_name]['pr_auc'] = float(pr_auc.group(1))
    
    return metrics


def extract_from_colab_reproduce():
    """Extract metrics from colab_reproduce_results.ipynb"""
    notebook_path = Path(__file__).parent.parent / "notebooks" / "colab_reproduce_results.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    results = []
    
    for cell in notebook['cells']:
        if 'outputs' in cell and len(cell['outputs']) > 0:
            # Check if this is an evaluation cell
            source_text = ''.join(cell.get('source', []))
            
            # Determine model variant, backbone, and split from source
            variant = None
            backbone = None
            split = None
            threshold = None
            
            if '--variant MLP' in source_text:
                variant = 'MLP'
            elif '--variant Transformer' in source_text:
                variant = 'Transformer'
            
            if '--backbone omnivore' in source_text:
                backbone = 'omnivore'
            elif '--backbone slowfast' in source_text:
                backbone = 'slowfast'
            
            if '--split step' in source_text:
                split = 'step'
            elif '--split recordings' in source_text:
                split = 'recordings'
            
            if '--threshold 0.6' in source_text:
                threshold = 0.6
            elif '--threshold 0.4' in source_text:
                threshold = 0.4
            
            # Extract metrics from outputs
            for output in cell['outputs']:
                if 'text' in output:
                    output_text = ''.join(output['text'])
                    
                    if 'test Step Level Metrics' in output_text or 'test Sub Step Level Metrics' in output_text:
                        extracted_metrics = extract_metrics_from_output(output_text)
                        
                        for level, level_metrics in extracted_metrics.items():
                            if level_metrics:
                                results.append({
                                    'Model': variant,
                                    'Backbone': backbone,
                                    'Split': split,
                                    'Threshold': threshold,
                                    'Level': level,
                                    'Precision': level_metrics.get('precision'),
                                    'Recall': level_metrics.get('recall'),
                                    'F1': level_metrics.get('f1'),
                                    'Accuracy': level_metrics.get('accuracy'),
                                    'AUC': level_metrics.get('auc'),
                                    'PR_AUC': level_metrics.get('pr_auc'),
                                })
    
    return results


def extract_from_rnn_baseline():
    """Extract metrics from rnn_baseline_colab.ipynb"""
    notebook_path = Path(__file__).parent.parent / "notebooks" / "rnn_baseline_colab.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    results = []
    
    for cell in notebook['cells']:
        if 'outputs' in cell and len(cell['outputs']) > 0:
            # Check if this is an evaluation cell
            source_text = ''.join(cell.get('source', []))
            
            # Determine model variant, backbone, and split from source
            variant = 'RNN'
            backbone = None
            split = None
            threshold = None
            
            if '--backbone omnivore' in source_text or 'omnivore' in source_text.lower():
                backbone = 'omnivore'
            elif '--backbone slowfast' in source_text or 'slowfast' in source_text.lower():
                backbone = 'slowfast'
            
            if '--split step' in source_text:
                split = 'step'
            elif '--split recordings' in source_text:
                split = 'recordings'
            
            if '--threshold 0.6' in source_text:
                threshold = 0.6
            elif '--threshold 0.4' in source_text:
                threshold = 0.4
            
            # Also check for evaluation results in comparison table
            for output in cell['outputs']:
                if 'text' in output:
                    output_text = ''.join(output['text'])
                    
                    # Extract from test metrics
                    if 'test Step Level Metrics' in output_text or 'test Sub Step Level Metrics' in output_text:
                        extracted_metrics = extract_metrics_from_output(output_text)
                        
                        for level, level_metrics in extracted_metrics.items():
                            if level_metrics:
                                results.append({
                                    'Model': variant,
                                    'Backbone': backbone,
                                    'Split': split,
                                    'Threshold': threshold,
                                    'Level': level,
                                    'Precision': level_metrics.get('precision'),
                                    'Recall': level_metrics.get('recall'),
                                    'F1': level_metrics.get('f1'),
                                    'Accuracy': level_metrics.get('accuracy'),
                                    'AUC': level_metrics.get('auc'),
                                    'PR_AUC': level_metrics.get('pr_auc'),
                                })
                    
                    # Extract from comparison table (Cell 31 output)
                    if 'Step Precision' in output_text and 'RNN' in output_text:
                        # Parse table format - this is from recordings split with threshold 0.6
                        lines = output_text.split('\n')
                        for line in lines:
                            if 'RNN' in line and ('omnivore' in line.lower() or 'slowfast' in line.lower()):
                                parts = line.split()
                                if len(parts) >= 6:
                                    try:
                                        backbone_from_table = parts[1].lower()
                                        precision_val = float(parts[2])
                                        recall_val = float(parts[3])
                                        f1_val = float(parts[4])
                                        accuracy_val = float(parts[5])
                                        auc_val = float(parts[6]) if len(parts) > 6 else None
                                        
                                        # This is step-level metrics from recordings split
                                        # Values are already in percentage, convert to decimal
                                        results.append({
                                            'Model': 'RNN',
                                            'Backbone': backbone_from_table,
                                            'Split': 'recordings',
                                            'Threshold': 0.6,  # From comparison table context
                                            'Level': 'Step',
                                            'Precision': precision_val / 100.0,
                                            'Recall': recall_val / 100.0,
                                            'F1': f1_val / 100.0,
                                            'Accuracy': accuracy_val / 100.0,
                                            'AUC': auc_val / 100.0 if auc_val else None,
                                            'PR_AUC': None,
                                        })
                                    except (ValueError, IndexError):
                                        pass
    
    return results


def main():
    """Main extraction function."""
    print("Extracting metrics from colab_reproduce_results.ipynb...")
    colab_results = extract_from_colab_reproduce()
    print(f"  Found {len(colab_results)} metric entries")
    
    print("Extracting metrics from rnn_baseline_colab.ipynb...")
    rnn_results = extract_from_rnn_baseline()
    print(f"  Found {len(rnn_results)} metric entries")
    
    # Combine results
    all_results = colab_results + rnn_results
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Clean up RNN data - fill missing splits
    # RNN models were evaluated on recordings split (from context)
    df.loc[(df['Model'] == 'RNN') & (df['Split'].isna()), 'Split'] = 'recordings'
    df.loc[(df['Model'] == 'RNN') & (df['Threshold'].isna()), 'Threshold'] = 0.6
    
    # Remove duplicates - prefer entries with complete information
    # Sort by completeness (non-null values)
    df['completeness'] = df[['Precision', 'Recall', 'F1', 'Accuracy', 'AUC']].notna().sum(axis=1)
    df = df.sort_values(['completeness', 'PR_AUC'], ascending=[False, False])
    df = df.drop_duplicates(subset=['Model', 'Backbone', 'Split', 'Level'], keep='first')
    df = df.drop(columns=['completeness'])
    
    # Sort by Model, Backbone, Split, Level
    df = df.sort_values(['Model', 'Backbone', 'Split', 'Level']).reset_index(drop=True)
    
    # Save to CSV
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "comparison_metrics.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nExtracted {len(df)} unique metric entries")
    print(f"Saved to: {output_path}")
    print("\nDataFrame summary:")
    print(df.to_string())
    
    return df


if __name__ == "__main__":
    df = main()

