# Skin Disease Classification Using Deep Convolutional Neural Networks

### A Comparative Study of MobileNetV2, ResNet50, and DenseNet121 in Dermatological Image Analysis

**Student Name**: THARIK AKBAR  
**Student ID (NIM)**: 001202607005  
**Study Program**: S2 Informatics  
**Course**: Deep Learning — Assignment 1  

---

## 1. Project Objective

This project investigates dermatological skin lesion classification across 9 distinct disease categories using Convolutional Neural Networks (CNNs) and transfer learning. We implement, train, evaluate, and critically compare three prominent CNN backbone architectures:

1. **MobileNetV2** (Lightweight depthwise separable convolutions for edge clinical diagnosis)
2. **ResNet50** (Residual bottleneck skip-connections mitigating gradient degradation)
3. **DenseNet121** (Dense connectivity promoting maximum multi-scale feature reuse and gradient flow)

The entire pipeline—from data inspection and deduplication, stratified partitioning, and training-only data augmentation to full test-set evaluation and error pattern analysis—is implemented to academic research standards. All exact duplicate leakage identified through SHA-256 hashing was removed before dataset splitting.

---

## 2. Dataset & Data Leakage Prevention

- **Source**: [Kaggle: Skin Disease Classification Image Dataset](https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset) by Riya Eliza Shaju.
- **Raw Files Downloaded**: 876 image files across 9 classes.
- **Dermatological Classes (9)**:
  1. Actinic keratosis (Pre-malignant squamous lesion)
  2. Atopic Dermatitis (Eczematous inflammatory lesion)
  3. Benign keratosis (Seborrheic keratosis / solar lentigo)
  4. Dermatofibroma (Benign fibrohistiocytic lesion)
  5. Melanocytic nevus (Benign melanocytic mole)
  6. Melanoma (Malignant melanocytic neoplasm)
  7. Squamous cell carcinoma (Invasive malignant keratinocytic carcinoma)
  8. Tinea Ringworm Candidiasis (Fungal superficial cutaneous infection)
  9. Vascular lesion (Angiomas, pyogenic granulomas, hemorrhage)

### Critical Finding: Predefined Split Data Leakage
During exploratory data inspection using cryptographic SHA-256 file hashing, **33 duplicate images** were identified in the raw Kaggle dataset. Crucially, **32 images in the Kaggle author's default `Split_smol/val` directory were bit-for-bit identical duplicates of images already present in `Split_smol/train`** (along with 1 intra-train duplicate).

All exact duplicate leakage identified through SHA-256 hashing was removed before dataset splitting. Note that patient-level duplication and visually near-duplicate images could not be completely excluded because patient identifiers were unavailable in the public dataset.

After removing duplicates, the remaining **843 unique images** were split using stratified sampling. Stratified sampling was used to preserve class proportions as closely as possible across the training, validation, and test subsets with a fixed random seed (`seed = 42`):

### Partition Statistics
| Split | Image Count | Percentage |
| :--- | :---: | :---: |
| **Training Set** | 590 | 70.0% |
| **Validation Set** | 126 | 14.9% |
| **Test Set (Untouched)** | 127 | 15.1% |
| **Total Unique** | **843** | **100.0%** |

---

## 3. Project Directory Structure

```text
skin_disease_cnn_assignment/
│
├── data/
│   └── raw/                                # Extracted raw dataset images
│       ├── train/
│       └── val/
│
├── src/                                    # Modular source code
│   ├── __init__.py
│   ├── config.py                           # Paths, hyperparams, and class constants
│   ├── dataset.py                          # Deduplication, stratified split & tf.data pipeline
│   ├── dataset_inspector.py                # Raw data inspection, hashing & sample grid
│   ├── models.py                           # MobileNetV2, ResNet50 & DenseNet121 factories
│   ├── train.py                            # Unified training pipeline with callbacks
│   ├── evaluate.py                         # Test-set evaluation & latency benchmarking
│   ├── visualize.py                        # Loss/accuracy curves & confusion matrix plots
│   ├── error_analysis.py                   # Dermatological error breakdown & mistake visualizer
│   └── flowchart.py                        # Academic methodology flowchart generator
│
├── models/                                 # Best saved Keras checkpoints
│   ├── mobilenetv2_best.keras
│   ├── resnet50_best.keras
│   └── densenet121_best.keras
│
├── results/
│   ├── figures/                            # Publication-quality charts & learning curves
│   │   ├── class_distribution.png
│   │   ├── dataset_sample_grid.png
│   │   ├── methodology_flowchart.png
│   │   ├── methodology_flowchart.svg
│   │   ├── models_training_comparison.png
│   │   ├── mobilenetv2_learning_curves.png
│   │   ├── resnet50_learning_curves.png
│   │   ├── densenet121_learning_curves.png
│   │   ├── mobilenetv2_misclassifications.png
│   │   ├── resnet50_misclassifications.png
│   │   └── densenet121_misclassifications.png
│   │
│   ├── confusion_matrix/                   # Confusion matrix heatmaps & CSVs
│   │   ├── mobilenetv2_confusion_matrix.png
│   │   ├── mobilenetv2_confusion_matrix.csv
│   │   ├── resnet50_confusion_matrix.png
│   │   ├── resnet50_confusion_matrix.csv
│   │   ├── densenet121_confusion_matrix.png
│   │   └── densenet121_confusion_matrix.csv
│   │
│   ├── classification_reports/             # Detailed per-class precision/recall/F1
│   │   ├── mobilenetv2_classification_report.txt
│   │   ├── mobilenetv2_classification_report.json
│   │   ├── resnet50_classification_report.txt
│   │   ├── resnet50_classification_report.json
│   │   ├── densenet121_classification_report.txt
│   │   └── densenet121_classification_report.json
│   │
│   └── metrics/                            # Raw experimental tabular data & JSONs
│       ├── all_images_metadata.csv
│       ├── train_split.csv
│       ├── val_split.csv
│       ├── test_split.csv
│       ├── split_distribution_summary.csv
│       ├── dataset_summary.json
│       ├── training_summary_all_models.json
│       ├── model_comparison_table.csv
│       ├── model_comparison_table.json
│       └── cross_model_error_analysis.json
│
├── report_assets/                          # Figures formatted for final PDF report
│   ├── methodology_flowchart.png
│   └── methodology_flowchart.svg
│
├── notebooks/                              # Jupyter analysis artifacts
├── main.ipynb                              # Master reproducible submission notebook
├── requirements.txt                        # Pinned dependencies
└── README.md                               # Project documentation
```

---

## 4. Environment & Installation

### Local Environment Specs
- **OS**: Windows 11 Pro 64-bit (Build 10.0.26200)
- **Host GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM)
- **Host RAM**: 32 GB RAM
- **Python**: Python 3.12.14 in an isolated virtual environment (`.venv`)
- **Deep Learning Framework**: TensorFlow 2.21.0 & Keras 3.15.1 with oneDNN optimization

### Setup Instructions

1. Clone or navigate to the project directory:
   ```bash
   cd "skin_disease_cnn_assignment"
   ```

2. Create an isolated virtual environment and activate it:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. Install all dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 5. How to Run the Experiments

All experimental phases can be run sequentially via command line or explored interactively inside `main.ipynb`.

### Step 1: Inspect Dataset & Generate Distribution Figures
```powershell
python src/dataset_inspector.py
```

### Step 2: Prepare Splits & Verify Data Pipeline
```powershell
python src/dataset.py
```

### Step 3: Train All Three CNN Architectures
Executes training for MobileNetV2, ResNet50, and DenseNet121 with `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`:
```powershell
python src/train.py
```

### Step 4: Evaluate Checkpoints on Untouched Test Set
Evaluates models on test data, calculates multiclass metrics, computes inference latency, and plots confusion matrices:
```powershell
python src/evaluate.py
```

### Step 5: Perform Dermatological Error Analysis
Extracts false predictions, computes top confounding disease pairs, and plots sample misclassification grids:
```powershell
python src/error_analysis.py
```

### Step 6: Generate Methodology Flowchart
Produces the high-resolution PNG and vector SVG flowchart:
```powershell
python src/flowchart.py
```

---

## 6. Experimental Results & Model Comparison

Every metric reported below is derived directly from the real experimental execution on the untouched 127-image Test Set.

### Comparative Performance Table
| Architecture | Total Params | Trainable Params | Best Epoch | Train Time | Model Size | Inference Latency | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2** | **2,428,233** | 167,689 | 9 | **92.9 s** | **11.1 MB** | **16.55 ms** | **70.08%** | 73.11% | **71.11%** | **71.12%** | **69.98%** |
| **ResNet50** | 23,859,337 | 267,529 | 12 | 251.5 s | 93.7 MB | 36.86 ms | 69.29% | **73.20%** | 70.74% | 70.86% | 69.33% |
| **DenseNet121** | 7,173,961 | **134,409** | 13 | 268.2 s | 29.9 MB | 50.80 ms | 65.35% | 71.66% | 67.22% | 65.91% | 63.86% |

### Key Experimental Insights
1. **Strongest Overall Performance-Efficiency Balance**: **MobileNetV2** achieved the strongest overall trade-off among the three architectures under the current experimental setup, with highest test accuracy (**70.08%**), highest Macro F1 (**71.12%**), and highest Weighted F1 (**69.98%**), while requiring only **2.43M parameters**, an **11.12 MB** checkpoint, and **16.55 ms** latency per image on the host CPU.
2. **Analysis of the ResNet50 Validation-to-Test Performance Gap**: ResNet50 attained the highest validation accuracy (**83.33%** at Epoch 12) but dropped to **69.29%** on the independent test set—a decline of approximately **14.04 percentage points**. This relatively large gap suggests sensitivity to the specific validation partition and indicates that ResNet50's substantially higher parameter capacity (23.86M) may have led to mild overfitting or higher sampling variance on this compact dataset.
3. **DenseNet Stabilization Dynamics**: **DenseNet121** achieved **65.35%** test accuracy. Dense concatenation across layers required more epochs to stabilize, exhibiting sensitivity to small sample sizes per class.
4. **Clinical Confounders**: Error analysis revealed persistent diagnostic ambiguities between biologically related entities: **Actinic keratosis** vs. **Squamous cell carcinoma** (pre-malignant to malignant progression continuum) and **Melanoma** vs. **Melanocytic nevus** (overlapping pigment networks).
5. **Practical Deployment Scope**: The 16.55 ms/image inference latency was benchmarked on a workstation CPU; on-device edge deployment performance would require dedicated testing on mobile target hardware.

---

## 7. Authors & Academic Integrity Statement

- **Student Name**: THARIK AKBAR
- **Student ID (NIM)**: 001202607005
- **Study Program**: S2 Informatics
- **Course**: Deep Learning (Assignment 1)

All metrics, histories, confusion matrices, and figures in this repository were generated through verifiable local code execution. No numbers or figures were fabricated.

**Source Code Repository:** [https://github.com/aricksss/skin-disease-cnn-assignment-tharik-akbar](https://github.com/aricksss/skin-disease-cnn-assignment-tharik-akbar)
