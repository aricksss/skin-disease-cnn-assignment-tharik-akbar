"""
Visualization module for Skin Disease CNN Classification.
Generates publication-quality figures:
1. Training vs Validation Accuracy curves per model
2. Training vs Validation Loss curves per model
3. Combined multi-model comparison curves
4. Confusion Matrix heatmaps (counts and normalized percentages)
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, CONFUSION_DIR, METRICS_DIR, CLASS_NAMES

# Use clean styling for academic papers
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_learning_curves(model_name: str):
    """Plot Loss and Accuracy curves for a specific model using its real Keras history."""
    json_path = METRICS_DIR / f"{model_name}_history.json"
    if not json_path.exists():
        print(f"Warning: History not found at {json_path}")
        return
        
    with open(json_path) as f:
        data = json.load(f)
        
    history = data["history"]
    summary = data["summary"]
    epochs = range(1, len(history["loss"]) + 1)
    best_epoch = summary["best_epoch"]
    
    # 1. Combined Figure (2 subplots: Accuracy & Loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Accuracy Subplot
    ax1.plot(epochs, history["accuracy"], 'o-', color='#1f77b4', linewidth=2, label='Training Accuracy')
    ax1.plot(epochs, history["val_accuracy"], 's-', color='#ff7f0e', linewidth=2, label='Validation Accuracy')
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_title(f"{model_name.upper()} - Classification Accuracy", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', frameon=True)
    
    # Loss Subplot
    ax2.plot(epochs, history["loss"], 'o-', color='#1f77b4', linewidth=2, label='Training Loss')
    ax2.plot(epochs, history["val_loss"], 's-', color='#d62728', linewidth=2, label='Validation Loss')
    ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax2.set_title(f"{model_name.upper()} - Categorical Cross-Entropy Loss", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    combined_path = FIGURES_DIR / f"{model_name}_learning_curves.png"
    plt.savefig(combined_path, bbox_inches='tight')
    plt.close()
    
    # 2. Individual Accuracy Curve
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(epochs, history["accuracy"], 'o-', color='#1f77b4', linewidth=2, label='Training Accuracy')
    plt.plot(epochs, history["val_accuracy"], 's-', color='#ff7f0e', linewidth=2, label='Validation Accuracy')
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Checkpoint (Epoch {best_epoch})')
    plt.title(f"{model_name.upper()}: Training vs Validation Accuracy", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0, 1.05])
    plt.legend(loc='lower right', frameon=True)
    acc_path = FIGURES_DIR / f"{model_name}_accuracy_curve.png"
    plt.savefig(acc_path, bbox_inches='tight')
    plt.close()
    
    # 3. Individual Loss Curve
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(epochs, history["loss"], 'o-', color='#1f77b4', linewidth=2, label='Training Loss')
    plt.plot(epochs, history["val_loss"], 's-', color='#d62728', linewidth=2, label='Validation Loss')
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Checkpoint (Epoch {best_epoch})')
    plt.title(f"{model_name.upper()}: Training vs Validation Loss", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.legend(loc='upper right', frameon=True)
    loss_path = FIGURES_DIR / f"{model_name}_loss_curve.png"
    plt.savefig(loss_path, bbox_inches='tight')
    plt.close()
    
    print(f"Generated learning curve figures for {model_name} in {FIGURES_DIR}")

def plot_all_models_comparison():
    """Plot overlay comparison of all three architectures."""
    models = ["mobilenetv2", "resnet50", "densenet121"]
    colors = {"mobilenetv2": "#1f77b4", "resnet50": "#2ca02c", "densenet121": "#d62728"}
    labels = {"mobilenetv2": "MobileNetV2", "resnet50": "ResNet50", "densenet121": "DenseNet121"}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    for m in models:
        json_path = METRICS_DIR / f"{m}_history.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            h = json.load(f)["history"]
        epochs = range(1, len(h["loss"]) + 1)
        ax1.plot(epochs, h["val_accuracy"], marker='o', color=colors[m], label=labels[m], linewidth=2)
        ax2.plot(epochs, h["val_loss"], marker='s', color=colors[m], label=labels[m], linewidth=2)
        
    ax1.set_title("Validation Accuracy Across Epochs", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Validation Accuracy", fontsize=11)
    ax1.set_ylim([0, 1.0])
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.set_title("Validation Loss Across Epochs", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    comparison_path = FIGURES_DIR / "models_training_comparison.png"
    plt.savefig(comparison_path, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-model training comparison to: {comparison_path}")

def plot_confusion_matrix_heatmap(cm: np.ndarray, model_name: str, class_names=CLASS_NAMES):
    """Plot high-res annotated confusion matrix heatmap with both counts and percentages."""
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Calculate row-normalized percentages for annotations
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # handle 0/0
    
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)"
            
    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title(f"Confusion Matrix: {model_name.upper()}\n(Counts & Row-Normalized Recall %)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=10)
    plt.ylabel("True Class", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    out_path = CONFUSION_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix figure to: {out_path}")
