"""
Error Analysis and Misclassification Visualization Module.
Identifies:
- Strongest and weakest classes per architecture
- Most frequently confused class pairs
- Misclassification patterns and dermatological confounding factors
- Visualizes representative misclassified samples with True, Pred, and Confidence
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR, FIGURES_DIR, METRICS_DIR, REPORTS_DIR,
    CONFUSION_DIR, CLASS_NAMES
)
from src.dataset import prepare_data_splits, create_tf_dataset

def analyze_model_errors(model_name: str, test_df: pd.DataFrame, test_ds):
    """Perform in-depth error analysis on a specific model."""
    model_path = MODELS_DIR / f"{model_name}_best.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
    print(f"\n[Error Analysis: {model_name.upper()}]")
    model = tf.keras.models.load_model(str(model_path))
    
    # Run predictions
    y_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = test_df['label'].values
    confidences = np.max(y_probs, axis=1)
    
    test_df_copy = test_df.copy()
    test_df_copy['predicted_label'] = y_pred
    test_df_copy['predicted_class'] = [CLASS_NAMES[i] for i in y_pred]
    test_df_copy['confidence'] = confidences
    test_df_copy['is_correct'] = (y_true == y_pred)
    
    # 1. Misclassified samples
    errors_df = test_df_copy[~test_df_copy['is_correct']].copy()
    print(f"Total Test Images: {len(test_df)} | Total Errors: {len(errors_df)} | Accuracy: {(len(test_df)-len(errors_df))/len(test_df)*100:.2f}%")
    
    # 2. Confusion frequency pairs
    confusion_pairs = (
        errors_df.groupby(['class_name', 'predicted_class'])
        .size()
        .reset_index(name='count')
        .sort_values(by='count', ascending=False)
    )
    
    # 3. Class-level performance from classification report
    report_file = REPORTS_DIR / f"{model_name}_classification_report.json"
    strongest_classes = []
    weakest_classes = []
    if report_file.exists():
        with open(report_file) as f:
            rep = json.load(f)
        class_f1s = {cls: rep[cls]['f1-score'] for cls in CLASS_NAMES if cls in rep}
        sorted_f1s = sorted(class_f1s.items(), key=lambda x: x[1], reverse=True)
        strongest_classes = sorted_f1s[:3]
        weakest_classes = sorted_f1s[-3:]
        
    error_summary = {
        "model_name": model_name,
        "total_test_samples": len(test_df),
        "total_misclassified": len(errors_df),
        "error_rate": round(len(errors_df) / len(test_df), 4),
        "strongest_classes_f1": strongest_classes,
        "weakest_classes_f1": weakest_classes,
        "top_confusion_pairs": confusion_pairs.head(5).to_dict(orient="records")
    }
    
    # Save error analysis json
    err_json_path = METRICS_DIR / f"{model_name}_error_analysis.json"
    with open(err_json_path, "w") as f:
        json.dump(error_summary, f, indent=2)
        
    # 4. Generate Misclassification Sample Grid Figure
    if len(errors_df) > 0:
        num_samples = min(6, len(errors_df))
        # Select representative errors with highest confidence (high-confidence mistakes)
        sample_errors = errors_df.sort_values(by='confidence', ascending=False).head(num_samples)
        
        cols = 3
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows), dpi=300)
        axes = np.array(axes).flatten()
        
        for idx, (_, row) in enumerate(sample_errors.iterrows()):
            ax = axes[idx]
            try:
                with Image.open(row['filepath']) as img:
                    ax.imshow(img.convert('RGB'))
            except Exception as e:
                ax.text(0.5, 0.5, f"Image Load Error\n{e}", ha='center', va='center')
                
            title_text = (
                f"True: {row['class_name']}\n"
                f"Pred: {row['predicted_class']}\n"
                f"Confidence: {row['confidence']*100:.1f}%"
            )
            ax.set_title(title_text, fontsize=10, fontweight='bold', color='darkred', pad=6)
            ax.axis('off')
            
        # Hide any unused axes
        for j in range(num_samples, len(axes)):
            axes[j].axis('off')
            
        plt.suptitle(f"Sample Misclassifications: {model_name.upper()} on Test Set", fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = FIGURES_DIR / f"{model_name}_misclassifications.png"
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        print(f"Saved misclassification visual figure to: {fig_path}")
        
    return error_summary

def run_all_error_analyses():
    """Run error analysis across all 3 models."""
    _, _, test_df = prepare_data_splits()
    test_ds = create_tf_dataset(test_df, is_training=False, batch_size=32)
    
    models = ["mobilenetv2", "resnet50", "densenet121"]
    all_errors = {}
    
    for m in models:
        summary = analyze_model_errors(m, test_df, test_ds)
        all_errors[m] = summary
        
    overall_path = METRICS_DIR / "cross_model_error_analysis.json"
    with open(overall_path, "w") as f:
        json.dump(all_errors, f, indent=2)
        
    print("\n" + "="*70)
    print("CROSS-MODEL ERROR ANALYSIS COMPLETED")
    print("="*70)
    for m, data in all_errors.items():
        print(f"\n[{m.upper()}] Error Count: {data['total_misclassified']} / {data['total_test_samples']}")
        print(f"  Top Confusions:")
        for pair in data['top_confusion_pairs'][:3]:
            print(f"   * True '{pair['class_name']}' -> Predicted '{pair['predicted_class']}' ({pair['count']} instances)")

if __name__ == "__main__":
    run_all_error_analyses()
