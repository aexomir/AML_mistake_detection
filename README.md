# AML/DAAI 2025 - Mistake Detection Project

## Environment Setup

First of all, create a python environment with

```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Installing Dependencies

The project supports both CUDA and CPU-only installations. The code will automatically detect CUDA availability and use it if available, otherwise it will fall back to CPU.

**Option 1: Using the installation script (Recommended)**

```bash
# For CUDA support (if you have CUDA-capable GPU)
python install_deps.py --cuda

# For CPU-only installation
python install_deps.py --cpu
# or simply
python install_deps.py
```

**Option 2: Manual installation**

```bash
# For CUDA support
pip install -r requirements-cuda.txt

# For CPU-only
pip install -r requirements-cpu.txt
```

**Note**: The project will automatically detect and use CUDA if available, regardless of which installation method you used. You can also explicitly specify the device using the `--device` CLI argument (e.g., `--device cpu` or `--device cuda`).

Then, download the pre-extracted features for 1s segments and put them in the `data/features` directory.

## Step 1: Baselines reproduction

Download the official best checkpoints from [here](https://utdallas.app.box.com/s/uz3s1alrzucz03sleify8kazhuc1ksl3) (`error_recognition_best` directory) and place them in the `checkpoints`. Then run the evaluation for the error recognition task.

**Example command**:

```
python -m core.evaluate --variant MLP --backbone omnivore --ckpt checkpoints/error_recognition_best/MLP/omnivore/error_recognition_MLP_omnivore_step_epoch_43.pt --split step --threshold 0.6
```

You should be able to reproduce results close to those reported in the paper (Table 2):

| Split      | Model              | F1    | AUC   |
| ---------- | ------------------ | ----- | ----- |
| Step       | MLP (Omnivore)     | 24.26 | 75.74 |
| Recordings | MLP (Omnivore)     | 55.42 | 63.03 |
| Step       | Transf. (Omnivore) | 55.39 | 75.62 |
| Recordings | Transf. (Omnivore) | 40.73 | 62.27 |

**NOTE**: Use the thresholds indicated in the official README.md of project (0.6 for step and 0.4 for recordings steps).

## Acknowledgements

This project builds on many repositories from the CaptainCook4D release. Please refer to the original codebases for more details.

**Error Recognition**: https://github.com/CaptainCook4D/error_recognition

**Features Extraction**: https://github.com/CaptainCook4D/feature_extractors
