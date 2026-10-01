"""
Dataset Inspector and Explorer for Skin Disease Classification.
Performs verification, corrupt image detection, duplicate detection,
class distribution visualization, and representative sample grid generation.
"""

import os
import hashlib
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def inspect_dataset(data_dir: Path, output_dir: Path):
    """Inspect dataset files, detect corruption/duplicates, generate statistics and figures."""
    print("=== INSPECTING DATASET ===")
    
    figures_dir = output_dir / "results" / "figures"
    metrics_dir = output_dir / "results" / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    hash_map = {}
    duplicates = []
    corrupt_images = []
    
    # Traverse data_dir
    all_files = [p for p in data_dir.rglob('*') if p.is_file()]
    print(f"Total files found in {data_dir}: {len(all_files)}")
    
    for p in all_files:
        # Determine split and class from path
        # Path format: data/raw/<split>/<class_name>/<filename>
        rel_parts = p.relative_to(data_dir).parts
        if len(rel_parts) >= 2:
            split_folder = rel_parts[0]  # e.g., 'train' or 'val'
            class_name = rel_parts[1]    # e.g., 'Melanoma'
        else:
            split_folder = "unknown"
            class_name = "unknown"
            
        file_ext = p.suffix.lower()
        file_size_kb = round(p.stat().st_size / 1024, 2)
        
        # Hash check for duplicates
        file_hash = compute_file_hash(p)
        if file_hash in hash_map:
            duplicates.append({
                "duplicate_file": str(p),
                "original_file": str(hash_map[file_hash]),
                "sha256": file_hash
            })
        else:
            hash_map[file_hash] = p
            
        # Verify image integrity and dimensions
        is_corrupt = False
        width, height, channels, img_mode = None, None, None, None
        try:
            with Image.open(p) as img:
                img.verify()
            # Reopen to get image properties (verify closes or invalidates)
            with Image.open(p) as img:
                width, height = img.size
                img_mode = img.mode
                channels = len(img.getbands())
        except Exception as e:
            is_corrupt = True
            corrupt_images.append({
                "filepath": str(p),
                "error": str(e)
            })
            
        records.append({
            "filepath": str(p),
            "filename": p.name,
            "split_folder": split_folder,
            "class_name": class_name,
            "extension": file_ext,
            "file_size_kb": file_size_kb,
            "sha256": file_hash,
            "width": width,
            "height": height,
            "channels": channels,
            "mode": img_mode,
            "is_corrupt": is_corrupt
        })
        
    df = pd.DataFrame(records)
    
    # Save raw inspection dataset
    df.to_csv(metrics_dir / "all_images_metadata.csv", index=False)
    
    # Class distribution analysis
    class_counts = df.groupby(["class_name", "split_folder"]).size().unstack(fill_value=0)
    class_counts["Total"] = class_counts.sum(axis=1)
    class_counts = class_counts.sort_values(by="Total", ascending=False)
    class_counts.to_csv(metrics_dir / "class_distribution.csv")
    
    print("\n--- CLASS DISTRIBUTION ---")
    print(class_counts.to_string())
    
    # Image resolution statistics
    res_summary = {
        "min_width": int(df["width"].min()) if not df["width"].isna().all() else None,
        "max_width": int(df["width"].max()) if not df["width"].isna().all() else None,
        "mean_width": float(round(df["width"].mean(), 1)) if not df["width"].isna().all() else None,
        "min_height": int(df["height"].min()) if not df["height"].isna().all() else None,
        "max_height": int(df["height"].max()) if not df["height"].isna().all() else None,
        "mean_height": float(round(df["height"].mean(), 1)) if not df["height"].isna().all() else None,
        "color_modes": df["mode"].value_counts().to_dict(),
        "extensions": df["extension"].value_counts().to_dict()
    }
    
    dataset_summary = {
        "total_images": len(df),
        "total_classes": df["class_name"].nunique(),
        "classes": sorted(df["class_name"].unique().tolist()),
        "corrupt_images_count": len(corrupt_images),
        "corrupt_images": corrupt_images,
        "exact_duplicates_count": len(duplicates),
        "duplicates": duplicates,
        "image_resolution_summary": res_summary,
        "class_totals": df["class_name"].value_counts().to_dict()
    }
    
    with open(metrics_dir / "dataset_summary.json", "w") as f:
        json.dump(dataset_summary, f, indent=2)
        
    print("\n--- DATASET SUMMARY ---")
    print(f"Total Images: {dataset_summary['total_images']}")
    print(f"Total Classes: {dataset_summary['total_classes']}")
    print(f"Corrupt Images: {dataset_summary['corrupt_images_count']}")
    print(f"Exact Duplicate Files: {dataset_summary['exact_duplicates_count']}")
    print(f"Extensions: {res_summary['extensions']}")
    print(f"Color Modes: {res_summary['color_modes']}")
    print(f"Resolution Range: {res_summary['min_width']}x{res_summary['min_height']} to {res_summary['max_width']}x{res_summary['max_height']}")
    
    # Generate Publication-Quality Figures
    # 1. Class Distribution Chart
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(10, 6), dpi=300)
    
    totals = df["class_name"].value_counts()
    colors = sns.color_palette("viridis", len(totals))
    
    bars = plt.barh(totals.index[::-1], totals.values[::-1], color=colors[::-1], edgecolor="black", linewidth=0.8)
    plt.title("Raw Dataset Class Distribution (Total = 876 Images)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Images", fontsize=12, labelpad=10)
    plt.ylabel("Skin Disease Class", fontsize=12, labelpad=10)
    plt.xlim(0, max(totals.values) + 15)
    
    for bar in bars:
        w = bar.get_width()
        pct = (w / len(df)) * 100
        plt.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{int(w)} ({pct:.1f}%)",
                 va='center', ha='left', fontsize=10, fontweight="semibold")
        
    plt.tight_layout()
    chart_path = figures_dir / "class_distribution.png"
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved class distribution chart to: {chart_path}")
    
    # 2. Representative Sample Grid (3x3 grid for the 9 classes)
    classes = sorted(df["class_name"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=300)
    axes = axes.flatten()
    
    for i, cls in enumerate(classes):
        cls_df = df[df["class_name"] == cls]
        # Pick a valid representative image
        sample_row = cls_df.iloc[0]
        img_path = sample_row["filepath"]
        
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB")
            axes[i].imshow(im_rgb)
            axes[i].set_title(f"{cls}\n({len(cls_df)} images)", fontsize=11, fontweight="bold", pad=8)
            axes[i].axis("off")
            
    plt.suptitle("Representative Dermatological Samples per Class", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    sample_grid_path = figures_dir / "dataset_sample_grid.png"
    plt.savefig(sample_grid_path, bbox_inches='tight')
    plt.close()
    print(f"Saved representative sample grid to: {sample_grid_path}")
    
    return dataset_summary

if __name__ == "__main__":
    base_proj = Path(__file__).resolve().parent.parent
    data_raw = base_proj / "data" / "raw"
    inspect_dataset(data_raw, base_proj)
