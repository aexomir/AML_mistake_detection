# Model Comparison Report: MLP vs Transformer vs RNN

## Overview

This report presents a comprehensive comparison of three model architectures (MLP, Transformer, and RNN) across two video backbones (Omnivore and SlowFast) for mistake detection tasks. The evaluation is performed on two data splits (step and recordings) at both step-level and substep-level granularities.

## Data Summary

- **Models Evaluated**: MLP, Transformer, RNN
- **Backbones**: Omnivore, SlowFast
- **Splits**: step (threshold 0.6), recordings (threshold 0.4)
- **Evaluation Levels**: Step-level, Sub Step-level
- **Metrics**: Precision, Recall, F1 Score, Accuracy, AUC, PR-AUC

## Key Findings

### Best Performing Models

#### Step-Level Performance

**Step Split (Threshold 0.6):**
- **Best F1**: Transformer + Omnivore (0.554)
- **Best AUC**: MLP + Omnivore (0.757) and Transformer + Omnivore (0.756) - nearly tied
- **Best Precision**: MLP + Omnivore (0.661)
- **Best Recall**: Transformer + Omnivore (0.598)

**Recordings Split (Threshold 0.4):**
- **Best F1**: MLP + Omnivore (0.554)
- **Best AUC**: RNN + Omnivore (0.646)
- **Best Precision**: RNN + Omnivore (0.466)
- **Best Recall**: MLP + Omnivore (0.859)

#### Sub Step-Level Performance

**Step Split (Threshold 0.6):**
- **Best F1**: Transformer + Omnivore (0.532)
- **Best AUC**: Transformer + Omnivore (0.746)
- **Best Precision**: MLP + Omnivore (0.410)
- **Best Recall**: Transformer + Omnivore (0.661)

**Recordings Split (Threshold 0.4):**
- **Best F1**: RNN + SlowFast (0.529) - Note: This model has perfect recall but very low precision
- **Best AUC**: RNN + Omnivore (0.646)
- **Best Precision**: Transformer + Omnivore (0.449)
- **Best Recall**: RNN + SlowFast (1.000) - Perfect recall but at cost of precision

### Model-Specific Observations

1. **MLP Models**:
   - Strong performance on recordings split, especially with Omnivore backbone
   - High recall on recordings split (0.859 for Omnivore)
   - Good AUC scores across configurations
   - Balanced precision-recall trade-off

2. **Transformer Models**:
   - Excellent performance on step split with Omnivore backbone
   - Best overall F1 scores on step split
   - Strong AUC performance (0.756 on step split with Omnivore)
   - More balanced metrics compared to MLP

3. **RNN Models**:
   - Good performance with Omnivore backbone on recordings split
   - Competitive AUC scores (0.646 with Omnivore)
   - Poor performance with SlowFast backbone (F1=0.0 at step level)
   - RNN + SlowFast shows perfect recall at substep level but very low precision

### Backbone Comparison

**Omnivore Backbone:**
- Generally outperforms SlowFast across all model types
- Best results: Transformer + Omnivore on step split
- More consistent performance across metrics

**SlowFast Backbone:**
- Lower overall performance compared to Omnivore
- MLP + SlowFast shows good recall on recordings split
- RNN + SlowFast struggles significantly

### Split Comparison

**Step Split (Threshold 0.6):**
- More challenging evaluation setting
- Transformer models excel in this setting
- Higher precision, lower recall overall

**Recordings Split (Threshold 0.4):**
- More balanced precision-recall trade-off
- MLP models perform well here
- Generally higher recall values

## Detailed Metrics Tables

### Step-Level Metrics

#### Step Split (Threshold 0.6)

| Model | Backbone | Precision | Recall | F1 | Accuracy | AUC |
|-------|----------|-----------|--------|----|----------|-----|
| MLP | omnivore | 0.661 | 0.149 | 0.243 | 0.711 | 0.757 |
| MLP | slowfast | 0.319 | 0.996 | 0.483 | 0.336 | 0.631 |
| Transformer | omnivore | 0.516 | 0.598 | 0.554 | 0.699 | 0.756 |
| Transformer | slowfast | 0.477 | 0.249 | 0.327 | 0.680 | 0.671 |

#### Recordings Split (Threshold 0.4)

| Model | Backbone | Precision | Recall | F1 | Accuracy | AUC |
|-------|----------|-----------|--------|----|----------|-----|
| MLP | omnivore | 0.409 | 0.859 | 0.554 | 0.504 | 0.630 |
| MLP | slowfast | 0.414 | 0.718 | 0.525 | 0.534 | 0.569 |
| RNN | omnivore | 0.466 | 0.369 | 0.412 | 0.621 | 0.646 |
| RNN | slowfast | 0.000 | 0.000 | 0.000 | 0.641 | 0.500 |
| Transformer | omnivore | 0.454 | 0.369 | 0.407 | 0.614 | 0.623 |
| Transformer | slowfast | 0.412 | 0.531 | 0.464 | 0.559 | 0.598 |

### Sub Step-Level Metrics

#### Step Split (Threshold 0.6)

| Model | Backbone | Precision | Recall | F1 | Accuracy | AUC |
|-------|----------|-----------|--------|----|----------|-----|
| MLP | omnivore | 0.410 | 0.299 | 0.346 | 0.683 | 0.654 |
| MLP | slowfast | 0.391 | 0.038 | 0.069 | 0.714 | 0.578 |
| Transformer | omnivore | 0.445 | 0.661 | 0.532 | 0.674 | 0.746 |
| Transformer | slowfast | 0.442 | 0.319 | 0.371 | 0.697 | 0.653 |

#### Recordings Split (Threshold 0.4)

| Model | Backbone | Precision | Recall | F1 | Accuracy | AUC |
|-------|----------|-----------|--------|----|----------|-----|
| MLP | omnivore | 0.396 | 0.569 | 0.467 | 0.574 | 0.599 |
| MLP | slowfast | 0.345 | 0.174 | 0.231 | 0.620 | 0.538 |
| RNN | omnivore | 0.457 | 0.398 | 0.426 | 0.614 | 0.646 |
| RNN | slowfast | 0.359 | 1.000 | 0.529 | 0.359 | 0.500 |
| Transformer | omnivore | 0.449 | 0.351 | 0.394 | 0.645 | 0.625 |
| Transformer | slowfast | 0.389 | 0.465 | 0.423 | 0.584 | 0.602 |

## Visualizations

### Comparison Tables

- **Step Level - All Splits**: `outputs/comparison_table_step_all.png`
- **Step Level - Step Split**: `outputs/comparison_table_step_split.png`
- **Step Level - Recordings Split**: `outputs/comparison_table_step_recordings.png`
- **Sub Step Level - All Splits**: `outputs/comparison_table_substep_all.png`
- **Sub Step Level - Step Split**: `outputs/comparison_table_substep_split.png`
- **Sub Step Level - Recordings Split**: `outputs/comparison_table_substep_recordings.png`

### Charts and Graphs

#### F1 Score Comparisons
- Step level F1 comparisons for all splits
- Sub step level F1 comparisons for all splits

#### AUC Comparisons
- Step level AUC comparisons for all splits
- Sub step level AUC comparisons for all splits

#### Heatmaps
- Comprehensive heatmaps showing all metrics across models
- Separate heatmaps for step and substep levels

#### Multi-Metric Comparisons
- Side-by-side comparison of Precision, Recall, F1, and AUC
- Grouped by model and backbone

#### Backbone Comparisons
- Direct comparison of Omnivore vs SlowFast performance
- Grouped by model type

## Recommendations

1. **For Step Split Evaluation**:
   - Use **Transformer + Omnivore** for best overall F1 and balanced metrics
   - Use **MLP + Omnivore** if high precision is critical

2. **For Recordings Split Evaluation**:
   - Use **MLP + Omnivore** for best F1 score
   - Use **RNN + Omnivore** for competitive AUC and more balanced precision-recall

3. **Backbone Selection**:
   - **Omnivore** is generally preferred over SlowFast
   - SlowFast shows promise with MLP on recordings split

4. **Model Architecture**:
   - **Transformer** excels in step split scenarios
   - **MLP** performs well on recordings split
   - **RNN** shows potential with Omnivore but struggles with SlowFast

## Limitations and Notes

1. RNN models were only evaluated on recordings split (no step split results available)
2. RNN + SlowFast shows concerning performance (F1=0.0 at step level)
3. Threshold values differ between splits (0.6 for step, 0.4 for recordings)
4. Some models show extreme precision-recall trade-offs (e.g., MLP + SlowFast on step split has 0.996 recall but 0.319 precision)

## Files Generated

All comparison tables, charts, and visualizations are available in the `analysis/outputs/` directory:

- CSV files with detailed metrics
- PNG tables for easy viewing
- Bar charts for F1 and AUC comparisons
- Heatmaps for comprehensive metric visualization
- Multi-metric comparison charts
- Backbone comparison charts

## Conclusion

The comparison reveals that **Transformer + Omnivore** is the best choice for step split evaluation, while **MLP + Omnivore** excels on recordings split. The RNN baseline shows competitive performance with Omnivore backbone but requires further investigation, especially for SlowFast backbone integration. The choice of backbone (Omnivore vs SlowFast) has a significant impact on performance, with Omnivore generally providing better results across all model architectures.

