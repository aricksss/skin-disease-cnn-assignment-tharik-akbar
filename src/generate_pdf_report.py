"""
Academic PDF Report Generator for Skin Disease Classification using Deep Convolutional Neural Networks.
Master's Program in Informatics (S2 Informatics) - Deep Learning Assignment 1.

Student: THARIK AKBAR (NIM: 001202607005)
Course: Deep Learning - Assignment 1

Complies with all academic reporting standards, quality control criteria,
and rigorous scientific documentation guidelines.
"""

import sys
import os
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print running headers and total page numbers."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        
        # Suppress header on Cover Page (Page 1)
        if self._pageNumber > 1:
            self.drawString(54, 752, "Skin Disease Classification Using Deep CNN — Deep Learning Assignment 1")
            self.drawRightString(612 - 54, 752, "THARIK AKBAR | 001202607005")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 746, 612 - 54, 746)

        # Running Footer on all pages
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 612 - 54, 48)
        self.drawString(54, 36, "S2 Informatics — Verifiable Academic Experimental Report")
        self.drawRightString(612 - 54, 36, page_str)
        self.restoreState()

def build_pdf_report():
    pdf_path = PROJECT_ROOT / "Skin_Disease_CNN_Assignment_Report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Academic Color Palette
    c_primary = colors.HexColor("#1A365D")    # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0")  # Slate Blue
    c_dark = colors.HexColor("#2D3748")       # Body text charcoal
    c_light_bg = colors.HexColor("#F7FAFC")   # Table row light bg
    c_border = colors.HexColor("#CBD5E0")     # Light gray border
    c_highlight = colors.HexColor("#EBF8FF")  # Highlight blue
    c_alert_bg = colors.HexColor("#FFF5F5")   # Alert callout bg
    c_alert_border = colors.HexColor("#E53E3E")

    cover_inst_style = ParagraphStyle(
        'CoverInst', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=12, leading=16,
        textColor=c_secondary, alignment=1, spaceAfter=8
    )

    cover_title_style = ParagraphStyle(
        'CoverTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=22, leading=27,
        textColor=c_primary, alignment=1, spaceAfter=10
    )

    cover_subtitle_style = ParagraphStyle(
        'CoverSubtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=12, leading=16,
        textColor=c_secondary, alignment=1, spaceAfter=20
    )

    h1_style = ParagraphStyle(
        'AcademicH1', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=13, leading=17,
        textColor=c_primary, spaceBefore=14, spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'AcademicH2', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=10.5, leading=14,
        textColor=c_secondary, spaceBefore=10, spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'AcademicBody', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=13.5,
        textColor=c_dark, spaceAfter=6, alignment=4 # Justify
    )

    bullet_style = ParagraphStyle(
        'AcademicBullet', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=13.5,
        textColor=c_dark, leftIndent=14, spaceAfter=3
    )

    caption_style = ParagraphStyle(
        'FigCaption', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=8, leading=11,
        textColor=colors.HexColor("#4A5568"), alignment=1,
        spaceBefore=4, spaceAfter=10
    )

    table_hdr_style = ParagraphStyle(
        'TableHdr', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=10,
        textColor=colors.white, alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8, leading=10.5,
        textColor=c_dark, alignment=1
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8, leading=10.5,
        textColor=c_dark, alignment=0
    )

    callout_style = ParagraphStyle(
        'CalloutText', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=colors.HexColor("#742A2A")
    )

    story = []

    # =============================================================
    # 1. COVER PAGE
    # =============================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("PROGRAM MAGISTER INFORMATIKA (S2 INFORMATICS)", cover_inst_style))
    story.append(Paragraph("COURSE: DEEP LEARNING &bull; ASSIGNMENT 1", ParagraphStyle('SubSub', parent=cover_inst_style, fontSize=10, textColor=c_dark)))
    story.append(Spacer(1, 30))

    story.append(Paragraph("Skin Disease Classification Using Deep Convolutional Neural Networks", cover_title_style))
    story.append(Paragraph("A Comparative Study of MobileNetV2, ResNet50, and DenseNet121 in Dermatological Image Analysis", cover_subtitle_style))
    story.append(HRFlowable(width="80%", thickness=1.5, color=c_primary, spaceAfter=35))

    # Student Identity Card
    student_info_data = [
        [Paragraph("<b>Student Identity</b>", ParagraphStyle('HCard', parent=table_hdr_style, fontSize=9)), ""],
        [Paragraph("<b>Student Name</b>", table_cell_left), Paragraph("<b>THARIK AKBAR</b>", table_cell_left)],
        [Paragraph("<b>Student ID (NIM)</b>", table_cell_left), Paragraph("<b>001202607005</b>", table_cell_left)],
        [Paragraph("<b>Study Program</b>", table_cell_left), Paragraph("S2 Informatics", table_cell_left)],
        [Paragraph("<b>Course</b>", table_cell_left), Paragraph("Deep Learning", table_cell_left)],
        [Paragraph("<b>Assignment</b>", table_cell_left), Paragraph("Assignment 1", table_cell_left)],
        [Paragraph("<b>Academic Term</b>", table_cell_left), Paragraph("Odd Semester 2026/2027", table_cell_left)],
    ]
    t_id = Table(student_info_data, colWidths=[150, 230])
    t_id.setStyle(TableStyle([
        ('SPAN', (0,0), (1,0)),
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BACKGROUND', (0,1), (-1,-1), c_light_bg),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('BOX', (0,0), (-1,-1), 1, c_secondary),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_id)
    story.append(Spacer(1, 25))

    # Technical Execution Metadata Card
    tech_meta_data = [
        [Paragraph("<b>Experimental Runtime & Hardware Specification</b>", ParagraphStyle('HCard2', parent=table_hdr_style, fontSize=9)), ""],
        [Paragraph("<b>Hardware Present</b>", table_cell_left), Paragraph("NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM)", table_cell_left)],
        [Paragraph("<b>TensorFlow Execution Backend</b>", table_cell_left), Paragraph("CPU execution under native Windows 64-bit (oneDNN AVX2 optimizations active)", table_cell_left)],
        [Paragraph("<b>Python Framework</b>", table_cell_left), Paragraph("Python 3.12.14 &bull; TensorFlow 2.21.0 &bull; Keras 3.15.1", table_cell_left)],
        [Paragraph("<b>Evaluated Dataset Scope</b>", table_cell_left), Paragraph("876 Raw Images &bull; 843 Unique Images (post-deduplication) &bull; 9 Classes", table_cell_left)],
        [Paragraph("<b>Code Reproducibility</b>", table_cell_left), Paragraph("Master Notebook (<code>main.ipynb</code>) &bull; Verifiable Local Execution", table_cell_left)]
    ]
    t_tech = Table(tech_meta_data, colWidths=[150, 230])
    t_tech.setStyle(TableStyle([
        ('SPAN', (0,0), (1,0)),
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('BACKGROUND', (0,1), (-1,-1), c_light_bg),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_tech)

    # =============================================================
    # 2. EXECUTIVE SUMMARY / ABSTRACT
    # =============================================================
    story.append(PageBreak())
    story.append(Paragraph("Executive Summary", h1_style))
    exec_summary_text = (
        "This experimental report presents an end-to-end deep learning framework for automated skin disease "
        "classification across nine diagnostic dermatological categories using transfer learning with ImageNet-pretrained "
        "Convolutional Neural Networks (CNNs). Three distinct structural paradigms were investigated: <b>MobileNetV2</b> "
        "(lightweight depthwise separable convolutions), <b>ResNet50</b> (deep residual skip connections), and "
        "<b>DenseNet121</b> (dense layer-to-layer feature reuse). Prior to model training, a systematic cryptographic SHA-256 "
        "audit of the public Kaggle dataset uncovered <b>33 exact duplicate images</b>, of which 32 occurred across the "
        "author's predefined train and validation folders. To eliminate direct train-to-validation data leakage, all exact "
        "duplicates were removed, leaving <b>843 unique images</b> that were partitioned via stratified sampling into a "
        "reproducible 70% / 15% / 15% split (590 training, 126 validation, and 127 independent test samples). "
        "Under standardized experimental conditions (Adam, categorical cross-entropy, EarlyStopping, ReduceLROnPlateau), "
        "<b>MobileNetV2 achieved the strongest overall performance-efficiency trade-off among the three evaluated architectures "
        "under the current experimental setup</b>, attaining <b>70.08% Test Accuracy</b> and <b>71.12% Macro F1-Score</b> with only "
        "2.43M parameters, an 11.12 MB checkpoint size, and 16.55 ms/image inference latency on the test hardware. ResNet50 achieved "
        "69.29% test accuracy (70.86% Macro F1) but required nearly 10× more parameters (23.86M, 93.71 MB footprint), while DenseNet121 "
        "attained 65.35% test accuracy (65.91% Macro F1). A notable validation-to-test performance gap was observed in ResNet50 (83.33% "
        "validation accuracy vs. 69.29% test accuracy), suggesting sensitivity to validation sampling or mild overfitting on this compact "
        "dataset. In-depth error analysis revealed that classification errors were predominantly concentrated between clinically "
        "related entities—specifically premalignant Actinic Keratosis vs. invasive Squamous Cell Carcinoma, and benign Melanocytic Nevi "
        "vs. malignant Melanoma—mirroring recognized challenges in clinical dermatopathology."
    )
    story.append(Paragraph(exec_summary_text, body_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 3. INTRODUCTION
    # =============================================================
    story.append(Paragraph("3. Introduction & Clinical Motivation", h1_style))
    intro_p1 = (
        "Skin diseases represent one of the most common causes of clinical consultations worldwide, ranging from benign superficial "
        "lesions to life-threatening cutaneous malignancies such as malignant melanoma and squamous cell carcinoma. Timely detection "
        "is critical; for instance, localized melanoma treated early yields a 5-year survival rate exceeding 99%, whereas advanced regional "
        "or distant metastasis reduces survival drastically. However, visual evaluation requires specialized dermatoscopic expertise that "
        "is often unavailable in rural, primary-care, or resource-constrained healthcare environments."
    )
    intro_p2 = (
        "Automated Computer-Aided Diagnosis (CAD) systems powered by Convolutional Neural Networks (CNNs) offer substantial promise as "
        "<b>potential decision-support tools</b> to assist clinicians during initial triaging. Importantly, this project is an academic "
        "deep learning investigation rather than a certified medical device; automated image classification cannot replace comprehensive "
        "dermatological examination, histopathological biopsy, or clinical judgment, and future clinical validation would be required "
        "prior to any real-world deployment. In this work, we evaluate the comparative utility and operational trade-offs of MobileNetV2, "
        "ResNet50, and DenseNet121 on a 9-class dermatological dataset."
    )
    story.append(Paragraph(intro_p1, body_style))
    story.append(Paragraph(intro_p2, body_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 4. DATASET EXPLORATION & LEAKAGE AUDIT
    # =============================================================
    story.append(Paragraph("4. Dataset Exploration and Data Leakage Audit", h1_style))
    data_intro = (
        "The experimental dataset is sourced from the public Kaggle repository published by Riya Eliza Shaju. "
        "The raw archive contains <b>876 image files</b> categorized into 9 distinct skin disease classes. "
        "Before proceeding with model training, a full cryptographic SHA-256 hash audit was performed across all images."
    )
    story.append(Paragraph(data_intro, body_style))

    # Callout Box: Leakage Removal & Limitation
    leak_box_text = (
        "<b>DATA INTEGRITY AUDIT & LEAKAGE REMOVAL</b><br/>"
        "Cryptographic SHA-256 analysis identified <b>33 exact bit-for-bit duplicate files</b> in the raw Kaggle dataset. "
        "Crucially, <b>32 of these duplicates occurred across the author's predefined <code>Split_smol/train</code> and "
        "<code>Split_smol/val</code> folders</b> (e.g., duplicate Atopic Dermatitis lesions). Evaluating models on the author's "
        "predefined validation folder would have introduced severe data leakage. "
        "<b>All exact duplicate leakage identified through SHA-256 hashing was removed before dataset splitting</b>, "
        "leaving exactly <b>843 unique images</b>.<br/>"
        "<i>Methodological Limitation</i>: Patient-level duplication and visually near-duplicate images could not be completely "
        "excluded because patient identifiers and clinical case metadata were unavailable in the public dataset."
    )
    t_leak = Table([[Paragraph(leak_box_text, callout_style)]], colWidths=[504])
    t_leak.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_alert_bg),
        ('BOX', (0,0), (-1,-1), 1, c_alert_border),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_leak)
    story.append(Spacer(1, 8))

    # Table 1: Raw vs Deduplicated Class Counts
    raw_class_counts = [
        ("Actinic keratosis", 100, 0, 100, "11.86%"),
        ("Benign keratosis", 100, 0, 100, "11.86%"),
        ("Dermatofibroma", 100, 0, 100, "11.86%"),
        ("Melanocytic nevus", 100, 0, 100, "11.86%"),
        ("Melanoma", 100, 0, 100, "11.86%"),
        ("Squamous cell carcinoma", 100, 0, 100, "11.86%"),
        ("Vascular lesion", 100, 0, 100, "11.86%"),
        ("Atopic Dermatitis", 102, 22, 80, "9.49%"),
        ("Tinea Ringworm Candidiasis", 74, 11, 63, "7.47%"),
        ("TOTAL DATASET", 876, 33, 843, "100.00%")
    ]
    t1_data = [
        [Paragraph("Skin Disease Class", table_hdr_style),
         Paragraph("Raw Dataset Count", table_hdr_style),
         Paragraph("Duplicate Files", table_hdr_style),
         Paragraph("Unique Images", table_hdr_style),
         Paragraph("Proportion of Unique", table_hdr_style)]
    ]
    for row in raw_class_counts:
        is_tot = (row[0] == "TOTAL DATASET")
        c_font = 'Helvetica-Bold' if is_tot else 'Helvetica'
        c_st = ParagraphStyle('T1C', parent=table_cell_style, fontName=c_font)
        t1_data.append([
            Paragraph(row[0], ParagraphStyle('T1L', parent=c_st, alignment=0)),
            Paragraph(str(row[1]), c_st),
            Paragraph(str(row[2]), c_st),
            Paragraph(str(row[3]), c_st),
            Paragraph(row[4], c_st),
        ])
    t1 = Table(t1_data, colWidths=[174, 85, 80, 80, 85])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, c_light_bg]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#EDF2F7")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t1)
    story.append(Paragraph("Table 1: Raw vs. deduplicated class counts across the nine dermatological categories.", caption_style))
    story.append(Spacer(1, 8))

    # Figure 1: Class Distribution
    f1_path = PROJECT_ROOT / "results" / "figures" / "class_distribution.png"
    if f1_path.exists():
        story.append(Image(str(f1_path), width=5.8*inch, height=3.2*inch))
        story.append(Paragraph("Figure 1: Raw Dataset Class Distribution — Total 876 Images. (After SHA-256 deduplication, 843 unique images remained).", caption_style))
    story.append(Spacer(1, 8))

    # Figure 2: Sample Grid
    f2_path = PROJECT_ROOT / "results" / "figures" / "dataset_sample_grid.png"
    if f2_path.exists():
        story.append(Image(str(f2_path), width=5.6*inch, height=5.0*inch))
        story.append(Paragraph("Figure 2: Representative dermatological samples from each of the nine disease classes.", caption_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 5. PREPROCESSING & DATA AUGMENTATION
    # =============================================================
    story.append(Paragraph("5. Preprocessing and Data Augmentation", h1_style))
    prep_text = (
        "To ensure rigorous, standardized inputs across all three architectures:<br/>"
        "&bull; <b>Spatial Standardization</b>: All images were bilinearly resized to a uniform <b>224 × 224 pixels</b> with 3 RGB color channels.<br/>"
        "&bull; <b>Architectural Normalization</b>: Each CNN backbone requires specific ImageNet channel calibration. "
        "MobileNetV2 scales raw $[0, 255]$ values to $[-1, 1]$; ResNet50 applies zero-centering using ImageNet BGR mean subtraction; "
        "and DenseNet121 standardizes channels using ImageNet mean and variance scaling. Each model's native <code>preprocess_input</code> "
        "transformation was integrated as the first layer in the respective network.<br/>"
        "&bull; <b>Training-Only Augmentation</b>: To mitigate overfitting on 590 training images without distorting diagnostic criteria, "
        "conservative augmentation was applied to the training set only: random horizontal flipping, small rotations (&plusmn;5%), "
        "slight zooming (&plusmn;5%), and small spatial translations (&plusmn;4%). <i>Validation and test sets remained strictly unaugmented</i>."
    )
    story.append(Paragraph(prep_text, body_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 6. DATASET SPLITTING
    # =============================================================
    story.append(Paragraph("6. Dataset Splitting Strategy", h1_style))
    split_p = (
        "<b>Stratified sampling was used to preserve class proportions as closely as possible across the training, validation, "
        "and test subsets</b> under a fixed random seed (<code>seed=42</code>). Integer sample allocation prevents exact mathematical "
        "proportionality for smaller classes, but stratified splitting provides the closest valid approximation. "
        "The resulting subsets comprise <b>70% Training (590 images)</b>, <b>15% Validation (126 images)</b>, and "
        "<b>15% Held-Out Test (127 images)</b>, as detailed in Table 2."
    )
    story.append(Paragraph(split_p, body_style))

    per_class_splits = [
        ("Actinic keratosis", 70, 15, 15, 100),
        ("Atopic Dermatitis", 56, 12, 12, 80),
        ("Benign keratosis", 70, 15, 15, 100),
        ("Dermatofibroma", 70, 15, 15, 100),
        ("Melanocytic nevus", 70, 15, 15, 100),
        ("Melanoma", 70, 15, 15, 100),
        ("Squamous cell carcinoma", 70, 15, 15, 100),
        ("Tinea Ringworm Candidiasis", 44, 9, 10, 63),
        ("Vascular lesion", 70, 15, 15, 100),
        ("TOTAL SUBSETS", 590, 126, 127, 843)
    ]
    t2_data = [
        [Paragraph("Skin Disease Category", table_hdr_style),
         Paragraph("Training (70%)", table_hdr_style),
         Paragraph("Validation (15%)", table_hdr_style),
         Paragraph("Held-Out Test (15%)", table_hdr_style),
         Paragraph("Total Unique Images", table_hdr_style)]
    ]
    for row in per_class_splits:
        is_t = (row[0] == "TOTAL SUBSETS")
        f_st = 'Helvetica-Bold' if is_t else 'Helvetica'
        c_st = ParagraphStyle('T2C', parent=table_cell_style, fontName=f_st)
        t2_data.append([
            Paragraph(row[0], ParagraphStyle('T2L', parent=c_st, alignment=0)),
            Paragraph(str(row[1]), c_st),
            Paragraph(str(row[2]), c_st),
            Paragraph(str(row[3]), c_st),
            Paragraph(str(row[4]), c_st),
        ])
    t2 = Table(t2_data, colWidths=[174, 85, 80, 80, 85])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, c_light_bg]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#EDF2F7")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t2)
    story.append(Paragraph("Table 2: Sample allocation across training, validation, and test subsets.", caption_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 7. METHODOLOGY FLOWCHART
    # =============================================================
    story.append(PageBreak())
    story.append(Paragraph("7. Methodology Flowchart", h1_style))
    flow_desc = (
        "Figure 3 presents the academic methodology flowchart depicting the full pipeline: raw data inspection, "
        "deduplication, spatial resizing, architecture normalization, stratified partitioning, transfer learning, "
        "training with dynamic callbacks, best checkpoint saving, and independent test-set evaluation."
    )
    story.append(Paragraph(flow_desc, body_style))
    f3_path = PROJECT_ROOT / "results" / "figures" / "methodology_flowchart.png"
    if f3_path.exists():
        story.append(Image(str(f3_path), width=5.4*inch, height=7.2*inch))
        story.append(Paragraph("Figure 3: Academic methodology flowchart of the complete skin disease classification pipeline.", caption_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 8. CNN ARCHITECTURES
    # =============================================================
    story.append(Paragraph("8. CNN Architectures & Transfer Learning Strategy", h1_style))
    arch_text = (
        "Three distinct CNN backbones were implemented using transfer learning with ImageNet weights:<br/>"
        "1. <b>MobileNetV2</b>: Utilizes inverted residuals with linear bottlenecks and 3×3 depthwise separable convolutions, "
        "drastically reducing parameter volume and FLOPs.<br/>"
        "2. <b>ResNet50</b>: Features 50 deep layers arranged in residual bottleneck blocks with identity skip connections that "
        "mitigate vanishing gradients.<br/>"
        "3. <b>DenseNet121</b>: Connects every layer to every subsequent layer within dense blocks, promoting maximum feature reuse.<br/><br/>"
        "<b>Standardized Classification Head</b>: To ensure fairness, all backbones were frozen and equipped with an identical classification head: "
        "<code>GlobalAveragePooling2D</code> &rarr; <code>BatchNormalization</code> &rarr; <code>Dropout(0.3)</code> &rarr; "
        "<code>Dense(128, ReLU)</code> &rarr; <code>Dropout(0.2)</code> &rarr; <code>Dense(9, Softmax)</code>."
    )
    story.append(Paragraph(arch_text, body_style))

    arch_params = [
        ("MobileNetV2", "2,428,233", "167,689", "2,260,544", "11.12 MB"),
        ("ResNet50", "23,859,337", "267,529", "23,591,808", "93.71 MB"),
        ("DenseNet121", "7,173,961", "134,409", "7,039,552", "29.86 MB")
    ]
    t3_data = [
        [Paragraph("Architecture", table_hdr_style),
         Paragraph("Total Parameters", table_hdr_style),
         Paragraph("Trainable Parameters", table_hdr_style),
         Paragraph("Frozen Parameters", table_hdr_style),
         Paragraph("Checkpoint Size", table_hdr_style)]
    ]
    for row in arch_params:
        t3_data.append([
            Paragraph(row[0], ParagraphStyle('T3L', parent=table_cell_style, fontName='Helvetica-Bold', alignment=0)),
            Paragraph(row[1], table_cell_style),
            Paragraph(row[2], table_cell_style),
            Paragraph(row[3], table_cell_style),
            Paragraph(row[4], table_cell_style),
        ])
    t3 = Table(t3_data, colWidths=[124, 95, 95, 95, 95])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t3)
    story.append(Paragraph("Table 3: Structural parameter breakdown and disk checkpoint sizes.", caption_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 9. EXPERIMENTAL SETUP
    # =============================================================
    story.append(Paragraph("9. Experimental Setup & Hardware vs. Backend Clarification", h1_style))
    setup_text = (
        "<b>Hardware vs. Execution Backend Clarification</b>: An NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM) was present on the "
        "host computer. However, because TensorFlow officially dropped native Windows CUDA support starting in version 2.11, "
        "<b>the TensorFlow execution backend ran in CPU mode under the native Windows 64-bit environment</b> using optimized "
        "oneDNN AVX2 instruction sets. GPU acceleration was therefore not utilized during model execution.<br/><br/>"
        "<b>Hyperparameter Configuration</b>:<br/>"
        "&bull; <b>Optimizer</b>: Adam with initial learning rate $\\eta = 10^{-3}$<br/>"
        "&bull; <b>Loss Function</b>: Categorical Cross-Entropy<br/>"
        "&bull; <b>Batch Size</b>: 32 (19 iterations per epoch on 590 training samples)<br/>"
        "&bull; <b>Callbacks</b>: <code>ModelCheckpoint</code> (saving best weights based on minimum <code>val_loss</code>), "
        "<code>EarlyStopping</code> (patience = 5 epochs), and <code>ReduceLROnPlateau</code> (factor = 0.5, patience = 2 epochs)."
    )
    story.append(Paragraph(setup_text, body_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 10. TRAINING RESULTS
    # =============================================================
    story.append(PageBreak())
    story.append(Paragraph("10. Training Results & Convergence Curves", h1_style))
    train_res_text = (
        "Training convergence across the three architectures under identical hyperparameter policies yielded the following observations:<br/>"
        "&bull; <b>MobileNetV2</b>: Trained for 14 epochs before EarlyStopping triggered, restoring the optimal model from <b>Epoch 9</b> "
        "(validation accuracy: 73.02%, validation loss: 0.7124; training wall time: 92.90 seconds).<br/>"
        "&bull; <b>ResNet50</b>: Completed 15 epochs in 251.50 seconds. Its best checkpoint was achieved at <b>Epoch 12</b> "
        "(validation accuracy: 83.33%, validation loss: 0.6305).<br/>"
        "&bull; <b>DenseNet121</b>: Completed 15 epochs in 268.24 seconds, reaching its best checkpoint at <b>Epoch 13</b> "
        "(validation accuracy: 69.84%, validation loss: 0.7362)."
    )
    story.append(Paragraph(train_res_text, body_style))

    f4_path = PROJECT_ROOT / "results" / "figures" / "models_training_comparison.png"
    if f4_path.exists():
        story.append(Image(str(f4_path), width=6.0*inch, height=2.5*inch))
        story.append(Paragraph("Figure 4: Comparative validation accuracy and loss trajectories across training epochs.", caption_style))
    story.append(Spacer(1, 6))

    f5_path = PROJECT_ROOT / "results" / "figures" / "mobilenetv2_learning_curves.png"
    f6_path = PROJECT_ROOT / "results" / "figures" / "resnet50_learning_curves.png"
    f7_path = PROJECT_ROOT / "results" / "figures" / "densenet121_learning_curves.png"
    if f5_path.exists():
        story.append(Image(str(f5_path), width=6.0*inch, height=2.1*inch))
        story.append(Paragraph("Figure 5: MobileNetV2 learning curves (Accuracy & Loss).", caption_style))
    if f6_path.exists():
        story.append(Image(str(f6_path), width=6.0*inch, height=2.1*inch))
        story.append(Paragraph("Figure 6: ResNet50 learning curves (Accuracy & Loss).", caption_style))
    if f7_path.exists():
        story.append(Image(str(f7_path), width=6.0*inch, height=2.1*inch))
        story.append(Paragraph("Figure 7: DenseNet121 learning curves (Accuracy & Loss).", caption_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 11. TEST EVALUATION & 12. MODEL COMPARISON
    # =============================================================
    story.append(PageBreak())
    story.append(Paragraph("11. Test Evaluation & 12. Model Comparison", h1_style))
    eval_text = (
        "The best checkpoint of each model was evaluated on the independent, held-out Test Set (127 images). "
        "To ensure high readability and avoid text compression, the comparison is organized into two distinct tables: "
        "<b>Table 4a</b> presents computational and structural efficiency metrics, while <b>Table 4b</b> details held-out "
        "classification performance."
    )
    story.append(Paragraph(eval_text, body_style))
    story.append(Spacer(1, 4))

    # Table 4a: Efficiency Metrics
    story.append(Paragraph("Table 4a: Model Architectural Characteristics & Computational Efficiency", h2_style))
    t4a_data = [
        [Paragraph("Architecture", table_hdr_style),
         Paragraph("Total Params", table_hdr_style),
         Paragraph("Trainable Params", table_hdr_style),
         Paragraph("Checkpoint Size", table_hdr_style),
         Paragraph("Inference Latency*", table_hdr_style),
         Paragraph("Training Time", table_hdr_style),
         Paragraph("Best Epoch", table_hdr_style)]
    ]
    t4a_rows = [
        ("MobileNetV2", "2,428,233", "167,689", "11.12 MB", "16.55 ms/image", "92.90 s", "Epoch 9 / 14"),
        ("ResNet50", "23,859,337", "267,529", "93.71 MB", "36.86 ms/image", "251.50 s", "Epoch 12 / 15"),
        ("DenseNet121", "7,173,961", "134,409", "29.86 MB", "50.80 ms/image", "268.24 s", "Epoch 13 / 15")
    ]
    for row in t4a_rows:
        is_mob = (row[0] == "MobileNetV2")
        f_w = 'Helvetica-Bold' if is_mob else 'Helvetica'
        c_st = ParagraphStyle('T4aC', parent=table_cell_style, fontName=f_w)
        t4a_data.append([
            Paragraph(row[0], ParagraphStyle('T4aL', parent=c_st, alignment=0)),
            Paragraph(row[1], c_st),
            Paragraph(row[2], c_st),
            Paragraph(row[3], c_st),
            Paragraph(row[4], c_st),
            Paragraph(row[5], c_st),
            Paragraph(row[6], c_st),
        ])
    t4a = Table(t4a_data, colWidths=[94, 75, 75, 68, 72, 60, 60])
    t4a.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('BACKGROUND', (0,1), (-1,1), c_highlight),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t4a)
    story.append(Paragraph("*Note: Inference latency measured on local test PC hardware; mobile hardware was not directly benchmarked.", caption_style))
    story.append(Spacer(1, 8))

    # Table 4b: Classification Metrics
    story.append(Paragraph("Table 4b: Classification Performance on Held-Out Test Set (127 Untouched Samples)", h2_style))
    t4b_data = [
        [Paragraph("Architecture", table_hdr_style),
         Paragraph("Test Accuracy", table_hdr_style),
         Paragraph("Macro Precision", table_hdr_style),
         Paragraph("Macro Recall", table_hdr_style),
         Paragraph("Macro F1-Score", table_hdr_style),
         Paragraph("Weighted F1-Score", table_hdr_style),
         Paragraph("Best Val Accuracy", table_hdr_style)]
    ]
    t4b_rows = [
        ("MobileNetV2", "70.08%", "73.11%", "71.11%", "71.12%", "69.98%", "73.02%"),
        ("ResNet50", "69.29%", "73.20%", "70.74%", "70.86%", "69.33%", "83.33%"),
        ("DenseNet121", "65.35%", "71.66%", "67.22%", "65.91%", "63.86%", "69.84%")
    ]
    for row in t4b_rows:
        is_mob = (row[0] == "MobileNetV2")
        f_w = 'Helvetica-Bold' if is_mob else 'Helvetica'
        c_st = ParagraphStyle('T4bC', parent=table_cell_style, fontName=f_w)
        t4b_data.append([
            Paragraph(row[0], ParagraphStyle('T4bL', parent=c_st, alignment=0)),
            Paragraph(row[1], c_st),
            Paragraph(row[2], c_st),
            Paragraph(row[3], c_st),
            Paragraph(row[4], c_st),
            Paragraph(row[5], c_st),
            Paragraph(row[6], c_st),
        ])
    t4b = Table(t4b_data, colWidths=[104, 70, 70, 70, 70, 70, 50])
    t4b.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('BACKGROUND', (0,1), (-1,1), c_highlight),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t4b)
    story.append(Paragraph("Table 4b: Evaluated multi-class classification metrics on the independent held-out test partition.", caption_style))
    story.append(Spacer(1, 10))

    # =============================================================
    # 13. CONFUSION MATRIX ANALYSIS
    # =============================================================
    story.append(PageBreak())
    story.append(Paragraph("13. Confusion Matrix Analysis", h1_style))
    cm_text = (
        "To inspect per-class diagnostic sensitivity and specific error patterns, Figures 8a, 8b, and 8c present the confusion matrices "
        "for each model on the 127 test samples. Each matrix displays both integer misclassification counts and row-normalized true-positive recall percentages."
    )
    story.append(Paragraph(cm_text, body_style))

    cm_m = PROJECT_ROOT / "results" / "confusion_matrix" / "mobilenetv2_confusion_matrix.png"
    cm_r = PROJECT_ROOT / "results" / "confusion_matrix" / "resnet50_confusion_matrix.png"
    cm_d = PROJECT_ROOT / "results" / "confusion_matrix" / "densenet121_confusion_matrix.png"

    if cm_m.exists():
        story.append(Image(str(cm_m), width=5.6*inch, height=4.5*inch))
        story.append(Paragraph("Figure 8a: Confusion Matrix for MobileNetV2 (70.08% Test Accuracy).", caption_style))
    story.append(Spacer(1, 8))

    if cm_r.exists():
        story.append(PageBreak())
        story.append(Image(str(cm_r), width=5.6*inch, height=4.5*inch))
        story.append(Paragraph("Figure 8b: Confusion Matrix for ResNet50 (69.29% Test Accuracy).", caption_style))
        story.append(Spacer(1, 8))

    if cm_d.exists():
        story.append(Image(str(cm_d), width=5.6*inch, height=4.5*inch))
        story.append(Paragraph("Figure 8c: Confusion Matrix for DenseNet121 (65.35% Test Accuracy).", caption_style))
        story.append(Spacer(1, 8))

    # =============================================================
    # 14. ERROR ANALYSIS
    # =============================================================
    story.append(PageBreak())
    story.append(Paragraph("14. Dermatological Error Analysis", h1_style))
    err_text = (
        "Analysis of misclassified samples revealed that errors were not randomly distributed across classes; rather, "
        "they clustered around biologically and morphologically related dermatological conditions:<br/>"
        "&bull; <b>Actinic Keratosis (AK) &harr; Squamous Cell Carcinoma (SCC)</b>: "
        "This represented the most frequent confusion pair across all models (MobileNetV2: 6 instances; ResNet50: 4 instances; "
        "DenseNet121: 5 instances). In clinical pathology, Actinic Keratosis is the premalignant precursor to Cutaneous Squamous Cell Carcinoma. "
        "Both lesions exhibit hyperkeratosis, erythematous scaling, and crusting; distinguishing early non-invasive AK from micro-invasive SCC "
        "frequently requires histopathological biopsy even for experienced dermatologists.<br/>"
        "&bull; <b>Melanoma &harr; Melanocytic Nevus</b>: "
        "Cross-confusion between benign moles (Melanocytic Nevi) and malignant Melanoma occurred across all architectures (MobileNetV2: 3 cases; "
        "ResNet50: 4 cases; DenseNet121: 7 cases). Both lesions originate from melanocytic proliferation and share overlapping pigmentary networks and "
        "irregular border profiles.<br/>"
        "&bull; <b>Squamous Cell Carcinoma &harr; Dermatofibroma</b>: "
        "DenseNet121 misclassified SCC as Dermatofibroma 8 times. Both present as firm, dome-shaped nodular lesions with central fibrous or keratotic whitening.<br/>"
        "&bull; <b>Distinctive Classes</b>: <b>Vascular lesion</b> achieved the highest recognition (Precision: 93.3%–100%, F1: 90.3%–93.3%) "
        "due to distinct red-purple lacunae, while <b>Benign keratosis</b> also attained high precision (86.7%–92.3%) from its characteristic 'stuck-on' warty texture."
    )
    story.append(Paragraph(err_text, body_style))

    misc_m = PROJECT_ROOT / "results" / "figures" / "mobilenetv2_misclassifications.png"
    misc_r = PROJECT_ROOT / "results" / "figures" / "resnet50_misclassifications.png"
    misc_d = PROJECT_ROOT / "results" / "figures" / "densenet121_misclassifications.png"

    if misc_m.exists():
        story.append(Image(str(misc_m), width=5.8*inch, height=2.8*inch))
        story.append(Paragraph("Figure 9: High-confidence misclassified samples for MobileNetV2 with True Class, Predicted Class, and Confidence.", caption_style))
    if misc_r.exists():
        story.append(Image(str(misc_r), width=5.8*inch, height=2.8*inch))
        story.append(Paragraph("Figure 10: High-confidence misclassified samples for ResNet50 with True Class, Predicted Class, and Confidence.", caption_style))
    if misc_d.exists():
        story.append(Image(str(misc_d), width=5.8*inch, height=2.8*inch))
        story.append(Paragraph("Figure 11: High-confidence misclassified samples for DenseNet121 with True Class, Predicted Class, and Confidence.", caption_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 15. DISCUSSION
    # =============================================================
    story.append(PageBreak())
    story.append(Paragraph("15. Discussion", h1_style))
    
    # ResNet50 Validation-Test Gap Discussion
    disc_gap_title = "Analysis of the ResNet50 Validation-to-Test Performance Gap"
    story.append(Paragraph(disc_gap_title, h2_style))
    disc_gap_text = (
        "An important experimental observation in this study is the marked performance divergence exhibited by ResNet50 between "
        "the validation and held-out test partitions. During training, ResNet50 attained a peak validation accuracy of <b>83.33%</b> at Epoch 12 "
        "(validation loss: 0.6305). However, when evaluated on the independent test set, its accuracy dropped to <b>69.29%</b>—a decline of "
        "approximately <b>14.04 percentage points</b>. In contrast, MobileNetV2 demonstrated substantially tighter generalization, "
        "moving from 73.02% validation accuracy to 70.08% test accuracy (a modest 2.94 percentage-point variance), while DenseNet121 dropped "
        "from 69.84% validation accuracy to 65.35% test accuracy (a 4.49 percentage-point drop).<br/><br/>"
        "Under scientific interpretation, this relatively large gap in ResNet50 suggests sensitivity to the specific validation partition "
        "and indicates that the model's substantially higher parameter capacity (<b>23.86M parameters</b>, compared to 2.43M in MobileNetV2) "
        "may have led to mild overfitting or higher sampling variance within this compact dataset of 590 training images. "
        "While residual shortcut connections enable deep feature representation, the model's expressiveness appears to have partially adapted "
        "to idiosyncrasies of the validation set that did not generalize equally well to the test partition."
    )
    story.append(Paragraph(disc_gap_text, body_style))

    # Performance-Efficiency Trade-offs & Deployment Discussion
    disc_trade_title = "Performance-Efficiency Trade-offs & Edge Deployment Prospects"
    story.append(Paragraph(disc_trade_title, h2_style))
    disc_trade_text = (
        "<b>MobileNetV2 achieved the strongest overall performance-efficiency trade-off among the three evaluated architectures "
        "under the current experimental setup</b>. Despite utilizing nearly 10× fewer parameters than ResNet50, MobileNetV2 achieved slightly "
        "higher test accuracy (70.08% vs. 69.29%) and a higher Macro F1-Score (71.12% vs. 70.86%). DenseNet121, while theoretically "
        "benefiting from multi-scale feature concatenation, achieved 65.35% test accuracy and required the longest inference latency (50.80 ms/image).<br/><br/>"
        "Regarding potential practical deployment, <b>the lower measured inference latency (16.55 ms/image on experimental hardware) "
        "and smaller model size (11.12 MB) indicate that MobileNetV2 is a promising candidate for future edge-device or mobile deployment studies</b>. "
        "However, because the latency reported in this study was benchmarked on a standard local workstation CPU, mobile hardware was not directly "
        "benchmarked; actual on-device throughput and battery consumption would require dedicated testing on embedded mobile platforms."
    )
    story.append(Paragraph(disc_trade_text, body_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 16. LIMITATIONS
    # =============================================================
    story.append(Paragraph("16. Limitations of the Study", h1_style))
    lim_bullets = [
        "&bull; <b>Absence of Patient Identifiers</b>: Patient-level duplication, multiple crops of the same lesion, and varying lighting conditions "
        "from the same patient could not be completely excluded because patient metadata was unavailable in the public dataset.",
        "&bull; <b>Compact Sample Size</b>: With 843 unique images across 9 classes (~65 images per class in training), minority classes like "
        "<i>Tinea Ringworm Candidiasis</i> (63 images) and <i>Atopic Dermatitis</i> (80 images) had fewer samples, contributing to performance variance.",
        "&bull; <b>Unimodal Vision-Only Decision Boundary</b>: Clinical dermatological diagnosis routinely combines visual inspection with patient age, "
        "family history of melanoma, lesion evolution rate, and anatomical location. A visual-only CNN operates with inherent informational limits on ambiguous lesions.",
        "&bull; <b>CPU Execution Setting</b>: Models were executed on the CPU backend under native Windows. While oneDNN enabled efficient training "
        "(~93s for MobileNetV2), end-to-end full backbone fine-tuning across all 50+ layers was constrained."
    ]
    for b in lim_bullets:
        story.append(Paragraph(b, bullet_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 17. CONCLUSION
    # =============================================================
    story.append(Paragraph("17. Conclusion & Recommendations", h1_style))
    concl_text = (
        "In this study, MobileNetV2, ResNet50, and DenseNet121 were systematically evaluated for multi-class skin disease classification. "
        "Key findings from the investigation include:<br/>"
        "1. <b>MobileNetV2 exhibited the most favorable balance of classification performance and operational efficiency</b> among the evaluated models, "
        "attaining <b>70.08% Test Accuracy</b> and <b>71.12% Macro F1-Score</b> with a lightweight 11.12 MB checkpoint size.<br/>"
        "2. Larger capacity backbones (ResNet50) achieved higher training and validation scores but did not outperform MobileNetV2 on independent test data, "
        "demonstrating an approximately 14 percentage-point validation-to-test performance gap.<br/>"
        "3. Classification errors strongly mirrored recognized clinical dermatological ambiguities (Actinic Keratosis vs. Squamous Cell Carcinoma, "
        "and Melanocytic Nevus vs. Melanoma).<br/><br/>"
        "<b>Future Work Recommendations</b>: Subsequent studies should investigate multimodal models that fuse clinical patient metadata with dermatoscopic "
        "images, explore test-time data augmentation (TTA), and implement prediction confidence thresholds to flag uncertain predictions for mandatory specialist review."
    )
    story.append(Paragraph(concl_text, body_style))
    story.append(Spacer(1, 8))

    # =============================================================
    # 18. REFERENCES
    # =============================================================
    story.append(Paragraph("18. Academic References", h1_style))
    refs = [
        "[1] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. <i>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 4510-4520.",
        "[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. <i>IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 770-778.",
        "[3] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. <i>IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 4700-4708.",
        "[4] Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. <i>Nature</i>, 542(7639), 115-118.",
        "[5] Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. <i>Scientific Data</i>, 5, 180161.",
        "[6] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. <i>arXiv preprint arXiv:1412.6980</i>.",
        "[7] Shaju, R. E. (2022). Skin Disease Classification Image Dataset. <i>Kaggle Dataset Repository</i>. Available at: https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefStyle', parent=body_style, fontSize=8, leading=11, spaceAfter=2.5)))
    story.append(Spacer(1, 8))

    # =============================================================
    # 19. CODE AND REPRODUCIBILITY
    # =============================================================
    story.append(Paragraph("19. Code and Reproducibility", h1_style))
    code_text = (
        "To enable verification and peer review, all experiment code, model weights, split definitions, and metric logs are maintained in the project workspace:<br/>"
        "&bull; <b>Master Submission Notebook</b>: <code>skin_disease_cnn_assignment/main.ipynb</code> (fully executed with outputs embedded)<br/>"
        "&bull; <b>Modular Python Source</b>: <code>skin_disease_cnn_assignment/src/</code> (data pipeline, models, training, evaluation, flowchart, report generation)<br/>"
        "&bull; <b>Saved Model Checkpoints</b>: <code>skin_disease_cnn_assignment/models/</code> (<code>mobilenetv2_best.keras</code>, <code>resnet50_best.keras</code>, <code>densenet121_best.keras</code>)<br/>"
        "&bull; <b>Raw Metrics & Tables</b>: <code>skin_disease_cnn_assignment/results/metrics/</code> (CSV & JSON logs)<br/>"
        "&bull; <b>Dependencies</b>: Documented in <code>requirements.txt</code> under Python 3.12.14.<br/><br/>"
        "<b>Source Code Repository:</b> <code>[INSERT SHAREABLE URL BEFORE SUBMISSION - e.g., GitHub or Google Drive link]</code>"
    )
    story.append(Paragraph(code_text, body_style))

    # Build PDF with dynamic NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF Report Generated Successfully] Path: {pdf_path}")

    # Copy to workspace root if possible
    workspace_pdf = PROJECT_ROOT.parent / "Skin_Disease_CNN_Assignment_Report.pdf"
    try:
        shutil.copy2(pdf_path, workspace_pdf)
        print(f"[PDF Report Copied to Workspace Root] Path: {workspace_pdf}")
    except PermissionError:
        print(f"[Notice] Workspace root PDF ({workspace_pdf}) is currently open/locked in another program.")
        print(f"[Notice] The generated file is available at: {pdf_path}")
    return pdf_path, workspace_pdf

if __name__ == "__main__":
    build_pdf_report()
