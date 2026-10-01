"""
Academic Methodology Flowchart Generator.
Creates a publication-quality diagram representing the end-to-end experimental pipeline:
Dataset -> Inspection -> Preprocessing -> Stratified Split -> CNN Architectures -> Training -> Checkpoints -> Evaluation -> Metrics -> Comparison -> Conclusion.
Saves high-resolution PNG and SVG formats.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, REPORT_ASSETS_DIR

def draw_flowchart():
    """Render and save an academic flowchart diagram in PNG and SVG formats."""
    fig, ax = plt.subplots(figsize=(10, 14), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define style parameters
    box_props = dict(boxstyle="round,pad=0.5,rounding_size=0.15", ec="#2C3E50", lw=1.5)
    model_box_props = dict(boxstyle="round,pad=0.4,rounding_size=0.15", ec="#2980B9", lw=1.5)
    decision_props = dict(boxstyle="round,pad=0.5,rounding_size=0.15", ec="#27AE60", lw=1.5)
    
    # Nodes configuration: (y, text, fill_color, text_color, props)
    nodes = [
        (13.2, "Skin Disease Image Dataset\n(876 Images across 9 Classes)", "#EBF5FB", "#1B4F72", box_props),
        (11.8, "Dataset Inspection & Deduplication\n(SHA-256 Hashing, Corrupt Image Check, 33 Duplicates Removed -> 843 Unique)", "#E8F8F5", "#0E6251", box_props),
        (10.4, "Image Preprocessing & Augmentation\n(224×224 Resizing, RGB Normalization, Training-Only Flipping/Rotation/Zoom)", "#FEF9E7", "#7D6608", box_props),
        (9.0, "Reproducible Stratified Split (Seed=42)\n• Training: 70% (590 imgs)   • Validation: 15% (126 imgs)   • Test: 15% (127 imgs)", "#F4ECF7", "#512E5F", box_props),
    ]
    
    # Draw linear nodes from top down to split
    for i, (y, text, fill, text_col, props) in enumerate(nodes):
        bbox = dict(**props, fc=fill)
        ax.text(5.0, y, text, ha="center", va="center", fontsize=9.5, fontweight="bold",
                color=text_col, bbox=bbox, multialignment="center")
        
        # Arrow to next node
        if i < len(nodes) - 1:
            ax.annotate("", xy=(5.0, nodes[i+1][0] + 0.55), xytext=(5.0, y - 0.55),
                        arrowprops=dict(arrowstyle="->", color="#34495E", lw=1.5, mutation_scale=15))
            
    # Branching Models at y = 7.4
    model_y = 7.4
    models = [
        (2.0, "MobileNetV2\n(2.4M Params)\nLightweight Depthwise", "#E8F4F8", "#1A5276"),
        (5.0, "ResNet50\n(23.9M Params)\nResidual Bottlenecks", "#E8F8F5", "#145A32"),
        (8.0, "DenseNet121\n(7.2M Params)\nDense Feature Reuse", "#FDF2E9", "#78281F")
    ]
    
    split_y = nodes[-1][0] - 0.55
    for x, text, fill, text_col in models:
        bbox = dict(**model_box_props, fc=fill)
        ax.text(x, model_y, text, ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=text_col, bbox=bbox, multialignment="center")
        # Arrow from Split to each Model
        ax.annotate("", xy=(x, model_y + 0.65), xytext=(5.0, split_y),
                    arrowprops=dict(arrowstyle="->", color="#5D6D7E", lw=1.4, mutation_scale=13))
        
    # Reconvergence at Training at y = 5.8
    train_y = 5.8
    train_text = "Controlled Transfer Learning & Training\n(Frozen Backbones, Adam Optimizer, Cross-Entropy Loss, Batch Size 32,\nEarlyStopping, ReduceLROnPlateau, Best ModelCheckpoint)"
    bbox = dict(**box_props, fc="#FEF5E7")
    ax.text(5.0, train_y, train_text, ha="center", va="center", fontsize=9, fontweight="bold",
            color="#7E5109", bbox=bbox, multialignment="center")
    
    for x, _, _, _ in models:
        ax.annotate("", xy=(5.0, train_y + 0.65), xytext=(x, model_y - 0.65),
                    arrowprops=dict(arrowstyle="->", color="#5D6D7E", lw=1.4, mutation_scale=13))
        
    # Subsequent linear stages
    post_nodes = [
        (4.4, "Saved Best Checkpoints (.keras)\n(mobilenetv2_best.keras, resnet50_best.keras, densenet121_best.keras)", "#EAF2F8", "#1B4F72", box_props),
        (3.1, "Evaluation on Untouched Test Set (127 images)\n(Strict Separation: Test Set Never Seen in Training or Tuning)", "#FDEDEC", "#78281F", box_props),
        (1.9, "Comprehensive Performance Metrics\n• Accuracy, Macro & Weighted Precision, Recall, F1-Score\n• Detailed Confusion Matrices & Per-Class Breakdown\n• Inference Latency (ms/img) & Model Footprint (MB)", "#F5EEF8", "#4A235A", box_props),
        (0.6, "Cross-Architectural Comparison, Error Analysis & Synthesis\n(Trade-Offs Analysis: Accuracy vs. Efficiency for Clinical Decision-Support)", "#E8F8F5", "#0E6251", decision_props)
    ]
    
    prev_y = train_y - 0.65
    for y, text, fill, text_col, props in post_nodes:
        bbox = dict(**props, fc=fill)
        ax.text(5.0, y, text, ha="center", va="center", fontsize=9, fontweight="bold",
                color=text_col, bbox=bbox, multialignment="center")
        ax.annotate("", xy=(5.0, y + 0.55), xytext=(5.0, prev_y),
                    arrowprops=dict(arrowstyle="->", color="#34495E", lw=1.5, mutation_scale=15))
        prev_y = y - 0.55
        
    plt.tight_layout()
    
    # Save in both FIGURES_DIR and REPORT_ASSETS_DIR
    png_path_1 = FIGURES_DIR / "methodology_flowchart.png"
    svg_path_1 = FIGURES_DIR / "methodology_flowchart.svg"
    png_path_2 = REPORT_ASSETS_DIR / "methodology_flowchart.png"
    svg_path_2 = REPORT_ASSETS_DIR / "methodology_flowchart.svg"
    
    plt.savefig(png_path_1, bbox_inches="tight")
    plt.savefig(svg_path_1, bbox_inches="tight")
    plt.savefig(png_path_2, bbox_inches="tight")
    plt.savefig(svg_path_2, bbox_inches="tight")
    plt.close()
    
    print(f"Generated flowchart saved to:\n {png_path_1}\n {svg_path_1}\n {png_path_2}\n {svg_path_2}")

if __name__ == "__main__":
    draw_flowchart()
