# Pretrained Model Checkpoints

This directory is designated for saving trained Keras deep learning model checkpoints (`.keras` format).

Because binary checkpoint files (such as `resnet50_best.keras` at ~93.7 MB and `densenet121_best.keras` at ~29.9 MB) exceed or approach GitHub's recommended maximum repository file limits (50 MB), they are excluded from version control via `.gitignore`.

### Reproducing Model Weights
To train all three CNN architectures locally and regenerate these checkpoints:
```bash
python src/train.py
```
This will automatically train and save the following best-performing checkpoints based on validation loss:
- `mobilenetv2_best.keras` (~11.12 MB)
- `resnet50_best.keras` (~93.71 MB)
- `densenet121_best.keras` (~29.86 MB)
