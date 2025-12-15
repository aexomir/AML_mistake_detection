# Step-by-Step Guide

This guide covers Step 2 (Feature Sanity Check) and Step 3 (Evaluation Reproduction) for the AML Mistake Detection project.

## Prerequisites

- Environment set up (see `README.md`)
- Pre-extracted features downloaded to `data/video/{backbone}/` directory
- Checkpoints downloaded from [official repository](https://utdallas.app.box.com/s/uz3s1alrzucz03sleify8kazhuc1ksl3) and placed in `checkpoints/error_recognition_best/`

---

## Step 2: Feature Sanity Check

Verify that feature directories exist and contain valid `.npz` files for Omnivore and SlowFast backbones.

### Basic Usage

```bash
# Using default path (data/)
python scripts/run.py step2
```

### Custom Features Root

```bash
# Specify custom path
python scripts/run.py step2 --features_root /path/to/features

# Or use environment variable
export FEATURES_ROOT=/path/to/features
python scripts/run.py step2
```

### What It Checks

- Verifies existence of `{features_root}/video/omnivore/` and `{features_root}/video/slowfast/` directories
- Lists example `.npz` files found
- Attempts to load a sample file and displays:
  - Shape and dtype
  - Min, max, and mean values

---

## Step 3: Evaluation Reproduction

Run evaluation for different backbones (Omnivore/SlowFast), variants (MLP/Transformer), and splits (step/recordings).

### Important Notes

- **Thresholds**: Use `0.6` for `step` split and `0.4` for `recordings` split
- **Checkpoint paths**: Replace `XX` with actual epoch numbers from your downloaded checkpoints
- **Device**: Auto-detects CUDA if available. Force CPU with `--device cpu`

### Command Structure

```bash
python scripts/run.py step3 --split {step|recordings} --backbone {omnivore|slowfast} --variant {MLP|Transformer} --ckpt <checkpoint_path> --threshold <0.4|0.6>
```

---

### Omnivore - MLP

```bash
# Step split
python scripts/run.py step3 --split step --backbone omnivore --variant MLP \
  --ckpt checkpoints/error_recognition_best/MLP/omnivore/error_recognition_MLP_omnivore_step_epoch_43.pt \
  --threshold 0.6

# Recordings split
python scripts/run.py step3 --split recordings --backbone omnivore --variant MLP \
  --ckpt checkpoints/error_recognition_best/MLP/omnivore/error_recognition_MLP_omnivore_recordings_epoch_XX.pt \
  --threshold 0.4
```

### Omnivore - Transformer

```bash
# Step split
python scripts/run.py step3 --split step --backbone omnivore --variant Transformer \
  --ckpt checkpoints/error_recognition_best/Transformer/omnivore/error_recognition_Transformer_omnivore_step_epoch_XX.pt \
  --threshold 0.6

# Recordings split
python scripts/run.py step3 --split recordings --backbone omnivore --variant Transformer \
  --ckpt checkpoints/error_recognition_best/Transformer/omnivore/error_recognition_Transformer_omnivore_recordings_epoch_XX.pt \
  --threshold 0.4
```

### SlowFast - MLP

```bash
# Step split
python scripts/run.py step3 --split step --backbone slowfast --variant MLP \
  --ckpt checkpoints/error_recognition_best/MLP/slowfast/error_recognition_MLP_slowfast_step_epoch_XX.pt \
  --threshold 0.6

# Recordings split
python scripts/run.py step3 --split recordings --backbone slowfast --variant MLP \
  --ckpt checkpoints/error_recognition_best/MLP/slowfast/error_recognition_MLP_slowfast_recordings_epoch_XX.pt \
  --threshold 0.4
```

### SlowFast - Transformer

```bash
# Step split
python scripts/run.py step3 --split step --backbone slowfast --variant Transformer \
  --ckpt checkpoints/error_recognition_best/Transformer/slowfast/error_recognition_Transformer_slowfast_step_epoch_XX.pt \
  --threshold 0.6

# Recordings split
python scripts/run.py step3 --split recordings --backbone slowfast --variant Transformer \
  --ckpt checkpoints/error_recognition_best/Transformer/slowfast/error_recognition_Transformer_slowfast_recordings_epoch_XX.pt \
  --threshold 0.4
```

---

## Expected Results

Based on the paper (Table 2), you should see results close to:

| Split      | Model              | F1    | AUC   |
| ---------- | ------------------ | ----- | ----- |
| Step       | MLP (Omnivore)     | 24.26 | 75.74 |
| Recordings | MLP (Omnivore)     | 55.42 | 63.03 |
| Step       | Transf. (Omnivore) | 55.39 | 75.62 |
| Recordings | Transf. (Omnivore) | 40.73 | 62.27 |

---

## Troubleshooting

- **Features not found**: Ensure features are in `data/video/{backbone}/` with `.npz` files
- **Checkpoint not found**: Verify checkpoint paths match your downloaded files
- **Import errors**: Make sure you're running from the project root directory
- **Device issues**: Use `--device cpu` to force CPU usage if CUDA causes problems
