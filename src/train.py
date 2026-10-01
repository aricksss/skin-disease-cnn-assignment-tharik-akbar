"""
Unified Training Pipeline for Skin Disease CNN Classification.
Trains MobileNetV2, ResNet50, and DenseNet121 using identical configurations,
callbacks, and hyperparameter policies.
"""

import sys
import os
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR, METRICS_DIR, LOGS_DIR, INITIAL_EPOCHS,
    BATCH_SIZE, RANDOM_SEED
)
from src.dataset import prepare_data_splits, create_tf_dataset
from src.models import (
    build_mobilenetv2, build_resnet50, build_densenet121,
    get_model_summary_info
)

# Set random seeds for reproducibility
tf.keras.utils.set_random_seed(RANDOM_SEED)

def train_single_model(model_builder, model_name: str, train_ds, val_ds, epochs: int = INITIAL_EPOCHS):
    """Train a single model with ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau."""
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING: {model_name.upper()}")
    print(f"{'='*70}")
    
    # 1. Instantiate Model
    model = model_builder()
    info = get_model_summary_info(model)
    print(f"Architecture Parameters:")
    print(f" - Total Parameters:         {info['total_parameters']:,}")
    print(f" - Trainable Parameters:     {info['trainable_parameters']:,}")
    print(f" - Non-trainable Parameters: {info['non_trainable_parameters']:,}")
    
    # 2. Checkpoint and Log Paths
    best_model_path = MODELS_DIR / f"{model_name}_best.keras"
    history_json_path = METRICS_DIR / f"{model_name}_history.json"
    history_csv_path = METRICS_DIR / f"{model_name}_history.csv"
    
    # 3. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode="min",
            verbose=1
        )
    ]
    
    # 4. Execute Training with Precise Timing
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    
    # 5. Extract and Analyze Actual History
    hist_dict = history.history
    # Convert any float32 to float for clean JSON serialization
    hist_serializable = {k: [float(val) for val in v] for k, v in hist_dict.items()}
    
    # Find best epoch (minimum validation loss)
    val_losses = hist_serializable["val_loss"]
    best_epoch_idx = int(np.argmin(val_losses))
    best_epoch_num = best_epoch_idx + 1  # 1-indexed
    
    best_metrics = {
        "model_name": model_name,
        "best_epoch": best_epoch_num,
        "total_epochs_trained": len(val_losses),
        "training_time_seconds": round(training_time, 2),
        "best_val_loss": round(hist_serializable["val_loss"][best_epoch_idx], 4),
        "best_val_accuracy": round(hist_serializable["val_accuracy"][best_epoch_idx], 4),
        "best_train_loss": round(hist_serializable["loss"][best_epoch_idx], 4),
        "best_train_accuracy": round(hist_serializable["accuracy"][best_epoch_idx], 4),
        "total_parameters": info["total_parameters"],
        "trainable_parameters": info["trainable_parameters"]
    }
    
    # 6. Save History and Best Epoch Summary
    with open(history_json_path, "w") as f:
        json.dump({"summary": best_metrics, "history": hist_serializable}, f, indent=2)
        
    df_hist = pd.DataFrame(hist_serializable)
    df_hist["epoch"] = range(1, len(df_hist) + 1)
    df_hist.to_csv(history_csv_path, index=False)
    
    print(f"\n[Training Complete: {model_name}]")
    print(f" - Wall Time:       {training_time:.2f} seconds ({training_time/60:.2f} min)")
    print(f" - Best Epoch:      {best_epoch_num} / {len(val_losses)}")
    print(f" - Train Accuracy:  {best_metrics['best_train_accuracy']*100:.2f}%")
    print(f" - Val Accuracy:    {best_metrics['best_val_accuracy']*100:.2f}%")
    print(f" - Train Loss:      {best_metrics['best_train_loss']:.4f}")
    print(f" - Val Loss:        {best_metrics['best_val_loss']:.4f}")
    print(f" - Checkpoint:      {best_model_path}")
    
    return best_metrics, hist_serializable

def run_all_training():
    """Train all 3 architectures sequentially under identical conditions."""
    train_df, val_df, test_df = prepare_data_splits()
    
    train_ds = create_tf_dataset(train_df, is_training=True)
    val_ds = create_tf_dataset(val_df, is_training=False)
    
    models_to_train = [
        ("mobilenetv2", build_mobilenetv2),
        ("resnet50", build_resnet50),
        ("densenet121", build_densenet121)
    ]
    
    training_summaries = {}
    for name, builder in models_to_train:
        summary, _ = train_single_model(builder, name, train_ds, val_ds, epochs=INITIAL_EPOCHS)
        training_summaries[name] = summary
        
    summary_file = METRICS_DIR / "training_summary_all_models.json"
    with open(summary_file, "w") as f:
        json.dump(training_summaries, f, indent=2)
        
    print("\n" + "="*70)
    print("ALL THREE MODELS SUCCESSFULLY TRAINED!")
    print("="*70)
    summary_table = pd.DataFrame.from_dict(training_summaries, orient="index")
    print(summary_table[["total_parameters", "trainable_parameters", "best_epoch", "training_time_seconds", "best_train_accuracy", "best_val_accuracy"]].to_string())

if __name__ == "__main__":
    run_all_training()
