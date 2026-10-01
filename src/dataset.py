"""
Dataset loading, deduplication, stratified splitting, and tf.data pipeline creation.
Ensures zero data leakage between train, validation, and test splits.
"""

import os
import hashlib
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR, METRICS_DIR, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED,
    CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES
)

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash to detect bit-identical duplicate images."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def prepare_data_splits():
    """
    Scans raw data, deduplicates identical images, and creates stratified
    70% Train / 15% Validation / 15% Test splits with fixed random seed.
    Saves split metadata to CSV for complete reproducibility.
    """
    train_csv = METRICS_DIR / "train_split.csv"
    val_csv = METRICS_DIR / "val_split.csv"
    test_csv = METRICS_DIR / "test_split.csv"
    
    # If already prepared, reload
    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        return train_df, val_df, test_df
    
    # Otherwise scan files
    records = []
    for filepath in DATA_DIR.rglob('*.*'):
        if filepath.is_file() and filepath.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            rel_parts = filepath.relative_to(DATA_DIR).parts
            if len(rel_parts) >= 2:
                split_folder = rel_parts[0]
                class_name = rel_parts[1]
                if class_name in CLASS_TO_IDX:
                    records.append({
                        "filepath": str(filepath),
                        "filename": filepath.name,
                        "split_folder": split_folder,
                        "class_name": class_name,
                        "label": CLASS_TO_IDX[class_name],
                        "sha256": compute_file_hash(filepath)
                    })
                    
    df = pd.DataFrame(records)
    print(f"[Dataset] Scanned {len(df)} total image files across {df['class_name'].nunique()} classes.")
    
    # Deduplication to avoid data leakage
    unique_df = df.drop_duplicates(subset=['sha256']).copy().reset_index(drop=True)
    num_duplicates_dropped = len(df) - len(unique_df)
    print(f"[Dataset] Identified and removed {num_duplicates_dropped} duplicate images.")
    print(f"[Dataset] Total unique images retained for experiment: {len(unique_df)}")
    
    # Stratified 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        unique_df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=unique_df['class_name']
    )
    
    # Stratified 15% Val, 15% Test (50% of Temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df['class_name']
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Save splits
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    # Generate split summary table
    summary_df = pd.DataFrame({
        "Train (70%)": train_df['class_name'].value_counts(),
        "Val (15%)": val_df['class_name'].value_counts(),
        "Test (15%)": test_df['class_name'].value_counts(),
        "Total (100%)": unique_df['class_name'].value_counts()
    })
    summary_df.to_csv(METRICS_DIR / "split_distribution_summary.csv")
    
    print("\n--- DATA SPLIT SUMMARY ---")
    print(f"Training set:   {len(train_df)} images ({len(train_df)/len(unique_df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} images ({len(val_df)/len(unique_df)*100:.1f}%)")
    print(f"Test set:       {len(test_df)} images ({len(test_df)/len(unique_df)*100:.1f}%)")
    print(f"Total:          {len(unique_df)} images (100.0%)")
    
    return train_df, val_df, test_df

# Preprocessing & Data Augmentation Layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
    tf.keras.layers.RandomRotation(0.05, fill_mode="reflect", seed=RANDOM_SEED),
    tf.keras.layers.RandomZoom(0.05, fill_mode="reflect", seed=RANDOM_SEED),
    tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="reflect", seed=RANDOM_SEED)
], name="data_augmentation")

def load_and_preprocess_image(path, label, is_training=False):
    """Load image from disk, decode, resize, and optionally augment."""
    raw_img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw_img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    
    if is_training:
        img = data_augmentation(img, training=True)
        
    label_one_hot = tf.one_hot(label, depth=NUM_CLASSES)
    return img, label_one_hot

def create_tf_dataset(df: pd.DataFrame, is_training: bool = False, batch_size: int = BATCH_SIZE):
    """Construct an optimized tf.data.Dataset pipeline."""
    filepaths = df['filepath'].values
    labels = df['label'].values
    
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    if is_training:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        
    ds = ds.map(
        lambda p, l: load_and_preprocess_image(p, l, is_training=is_training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

if __name__ == "__main__":
    train_df, val_df, test_df = prepare_data_splits()
    train_ds = create_tf_dataset(train_df, is_training=True)
    val_ds = create_tf_dataset(val_df, is_training=False)
    test_ds = create_tf_dataset(test_df, is_training=False)
    
    for batch_x, batch_y in train_ds.take(1):
        print(f"Batch X shape: {batch_x.shape}, Batch Y shape: {batch_y.shape}")
        print(f"Pixel range: [{tf.reduce_min(batch_x):.1f}, {tf.reduce_max(batch_x):.1f}]")
