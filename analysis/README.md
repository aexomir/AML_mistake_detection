# Error-Type-Aware Analysis

This directory contains scripts for performing error-type-aware analysis of the SupervisedER task.

## Files

- `error_type_analysis.py`: Main analysis script that computes per-error-type performance metrics
- `outputs/`: Directory where analysis results (CSV, JSON, plots) are saved
- `figures/`: Directory for saved visualization figures (optional)

## Usage

Run the analysis script with your model checkpoint:

```bash
python analysis/error_type_analysis.py \
    --split step \
    --backbone omnivore \
    --variant MLP \
    --ckpt checkpoints/error_recognition_best/MLP/omnivore/error_recognition_MLP_omnivore_step_epoch_43.pt \
    --threshold 0.6 \
    --output-dir analysis/outputs
```

### Arguments

- `--split`: Data split to use (`step` or `recordings`)
- `--backbone`: Backbone model (`omnivore` or `slowfast`)
- `--variant`: Model variant (`MLP` or `Transformer`)
- `--ckpt`: Path to model checkpoint file
- `--threshold`: Threshold for binary classification (default: 0.5)
- `--step-normalization`: Apply step-level normalization (default: True)
- `--no-step-normalization`: Disable step-level normalization
- `--sub-step-normalization`: Apply sub-step-level normalization (default: True)
- `--no-sub-step-normalization`: Disable sub-step-level normalization
- `--output-dir`: Output directory for results (default: `analysis/outputs`)
- `--device`: Device to use (`cuda` or `cpu`). Auto-detects if not specified

## Output

The script generates:

1. **CSV file**: `error_type_analysis_{split}_{backbone}_{variant}_threshold_{threshold}.csv`
   - Contains per-error-type metrics (Accuracy, Precision, Recall, F1, AUC, sample count)
   - Includes a "Global (All)" row for comparison

2. **JSON file**: `error_type_analysis_{split}_{backbone}_{variant}_threshold_{threshold}.json`
   - Same data as CSV in JSON format

3. **Plot file**: `error_type_analysis_{split}_{backbone}_{variant}_threshold_{threshold}.png`
   - Bar plots showing F1 score and sample count per error type

## Error Types

The analysis computes metrics for the following error types:

- **No Error** (label 0): Steps without errors
- **Preparation Error** (label 2)
- **Temperature Error** (label 3)
- **Measurement Error** (label 4)
- **Timing Error** (label 5)
- **Technique Error** (label 6)

## Notes

- The analysis is performed at the **sub-step level** (each 1-second segment is a sample)
- Error types are assigned at the step level, so all sub-steps within a step share the same error types
- Metrics are computed using the same thresholding and normalization logic as the main evaluation
- AUC is only computed when both positive and negative samples are present for an error type

