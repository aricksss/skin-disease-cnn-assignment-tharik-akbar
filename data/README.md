# Dataset Directory

The raw dataset images are excluded from Git tracking via `.gitignore` to comply with repository size limits and data licensing.

### Dataset Source
- **Dataset Title**: Skin Disease Classification Image Dataset
- **Author**: Riya Eliza Shaju
- **Kaggle URL**: https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset

### Automatic Acquisition
Running the project's data inspection script automatically downloads and unpacks the Kaggle dataset into `data/raw/`:
```bash
python src/dataset_inspector.py
```
Alternatively, extract the dataset manually into `data/raw/` preserving the original class folders.
