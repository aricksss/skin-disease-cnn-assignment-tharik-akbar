"""
Test Set Evaluation and Model Comparison Module.
Evaluates the best checkpoint of MobileNetV2, ResNet50, and DenseNet121
on the untouched Test Set. Calculates overall and per-class metrics,
measures inference latency, model file size, and generates comparison tables.
"""

import sys
import os
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR, RESULTS_DIR, CONFUSION_DIR, REPORTS_DIR,
    METRICS_DIR, CLASS_NAMES
)
from src.dataset import prepare_data_splits, create_tf_dataset
from src.visualize import plot_learning_curves, plot_all_models_comparison, plot_confusion_matrix_heatmap

def evaluate_model(model_name: str, test_ds, y_true_classes, test_num_images: int):
    """Evaluate a single saved best model on the test dataset."""
    model_path = MODELS_DIR / f"{model_name}_best.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    print(f"\n--- Evaluating {model_name.upper()} ---")
    model = tf.keras.models.load_model(str(model_path))
    
    # 1. Model file size in MB
    file_size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)
    
    # 2. Warmup & Inference Latency Measurement
    # Warmup
    for x_batch, _ in test_ds.take(1):
        _ = model.predict(x_batch, verbose=0)
        
    start_time = time.time()
    y_pred_probs = model.predict(test_ds, verbose=0)
    inference_time_total = time.time() - start_time
    latency_ms_per_image = round((inference_time_total / test_num_images) * 1000, 2)
    
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # 3. Overall Multiclass Metrics
    acc = accuracy_score(y_true_classes, y_pred_classes)
    macro_prec = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    macro_rec = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    
    # 4. Classification Report
    clf_report_dict = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    clf_report_text = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    
    # Save classification reports
    report_txt_file = REPORTS_DIR / f"{model_name}_classification_report.txt"
    report_json_file = REPORTS_DIR / f"{model_name}_classification_report.json"
    with open(report_txt_file, "w") as f:
        f.write(f"CLASSIFICATION REPORT: {model_name.upper()}\n")
        f.write("="*70 + "\n")
        f.write(clf_report_text)
    with open(report_json_file, "w") as f:
        json.dump(clf_report_dict, f, indent=2)
        
    # 5. Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_csv_file = CONFUSION_DIR / f"{model_name}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_file)
    
    # Plot Confusion Matrix Heatmap
    plot_confusion_matrix_heatmap(cm, model_name, CLASS_NAMES)
    
    # Load training history summary for this model
    history_json = METRICS_DIR / f"{model_name}_history.json"
    best_epoch = None
    train_time = None
    if history_json.exists():
        with open(history_json) as f:
            hist_data = json.load(f)["summary"]
            best_epoch = hist_data["best_epoch"]
            train_time = hist_data["training_time_seconds"]
            total_params = hist_data["total_parameters"]
            trainable_params = hist_data["trainable_parameters"]
    else:
        total_params = model.count_params()
        trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        
    metrics_summary = {
        "model": model_name,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "best_epoch": best_epoch,
        "training_time_seconds": train_time,
        "model_file_size_mb": file_size_mb,
        "inference_latency_ms": latency_ms_per_image,
        "test_accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4)
    }
    
    print(f"Test Accuracy:    {acc*100:.2f}%")
    print(f"Macro Precision:  {macro_prec*100:.2f}%")
    print(f"Macro Recall:     {macro_rec*100:.2f}%")
    print(f"Macro F1-Score:   {macro_f1*100:.2f}%")
    print(f"Weighted F1-Score:{weighted_f1*100:.2f}%")
    print(f"Inference Speed:  {latency_ms_per_image} ms/image")
    print(f"Model File Size:  {file_size_mb} MB")
    
    return metrics_summary, y_pred_probs, y_pred_classes

def run_all_evaluations():
    """Run evaluation for all 3 models on the untouched test set and generate comparison table."""
    _, _, test_df = prepare_data_splits()
    test_ds = create_tf_dataset(test_df, is_training=False, batch_size=32)
    y_true_classes = test_df['label'].values
    
    models = ["mobilenetv2", "resnet50", "densenet121"]
    all_summaries = []
    
    for m in models:
        # Generate learning curves
        plot_learning_curves(m)
        
        # Evaluate on test set
        summary, _, _ = evaluate_model(m, test_ds, y_true_classes, len(test_df))
        all_summaries.append(summary)
        
    # Multi-model training curves overlay
    plot_all_models_comparison()
    
    # Build Comparison Table
    df_comparison = pd.DataFrame(all_summaries)
    
    # Save comparison table in CSV and JSON
    comparison_csv = METRICS_DIR / "model_comparison_table.csv"
    comparison_json = METRICS_DIR / "model_comparison_table.json"
    df_comparison.to_csv(comparison_csv, index=False)
    with open(comparison_json, "w") as f:
        json.dump(all_summaries, f, indent=2)
        
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON TABLE (UNTOUCHED TEST SET)")
    print("="*80)
    print(df_comparison.to_string(index=False))
    print(f"\nSaved comparison table to {comparison_csv}")
    
    return df_comparison

if __name__ == "__main__":
    run_all_evaluations()
