"""
visualization.py
================
Fungsi-fungsi visualisasi untuk Exploratory Data Analysis (EDA).

Memisahkan logika plotting dari notebook untuk menjaga kode notebook 
tetap bersih dan ringan.
"""

import logging
from pathlib import Path
from typing import Dict, List
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

logger = logging.getLogger(__name__)

# Palet Warna dan Theme
DARK_BG    = '#1e1e2e'
PANEL_BG   = '#313244'
TEXT_COLOR = '#cdd6f4'
GRID_COLOR = '#45475a'
PALETTE    = ['#89dceb', '#a6e3a1', '#fab387', '#f38ba8', '#cba6f7']


def set_plot_style() -> None:
    """Mengatur gaya visualisasi yang konsisten di seluruh notebook."""
    plt.rcParams.update({
        'figure.facecolor' : DARK_BG,
        'axes.facecolor'   : DARK_BG,
        'axes.edgecolor'   : GRID_COLOR,
        'axes.labelcolor'  : TEXT_COLOR,
        'xtick.color'      : TEXT_COLOR,
        'ytick.color'      : TEXT_COLOR,
        'text.color'       : TEXT_COLOR,
        'grid.color'       : PANEL_BG,
        'grid.linestyle'   : '--',
        'grid.alpha'       : 0.5,
        'axes.titlesize'   : 14,
        'axes.labelsize'   : 12,
        'figure.dpi'       : 120,
    })


def plot_class_distribution(dataset_info: Dict, total_images: int, save_path: Path) -> None:
    """Plot distribusi kelas (Bar chart dan Pie chart)."""
    set_plot_style()
    
    class_names_display = [v['class_name'] for v in dataset_info.values()]
    class_counts        = [v['image_count'] for v in dataset_info.values()]
    class_percentages   = [round(c / total_images * 100, 2) for c in class_counts]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Class Distribution — Training Dataset', fontsize=16, fontweight='bold', color=TEXT_COLOR, y=1.02)

    # --- Bar Chart ---
    ax1 = axes[0]
    bars = ax1.bar(class_names_display, class_counts, color=PALETTE[:len(class_counts)], edgecolor=GRID_COLOR, linewidth=0.8, width=0.5)
    ax1.set_title('Image Count per Class', fontsize=13, pad=10)
    ax1.set_xlabel('Class', fontsize=11)
    ax1.set_ylabel('Number of Images', fontsize=11)
    ax1.grid(axis='y', alpha=0.4)
    
    for bar, count, pct in zip(bars, class_counts, class_percentages):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(class_counts) * 0.01,
            f'{count:,}\\n({pct}%)',
            ha='center', va='bottom', fontsize=10, color=TEXT_COLOR, fontweight='bold'
        )
    ax1.set_ylim(0, max(class_counts) * 1.18)

    # --- Pie Chart ---
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        class_counts,
        labels=class_names_display,
        colors=PALETTE[:len(class_counts)],
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor=DARK_BG, linewidth=2)
    )
    for t in texts: t.set_color(TEXT_COLOR)
    for at in autotexts: at.set_color(DARK_BG); at.set_fontweight('bold')
    ax2.set_title('Percentage Distribution', fontsize=13, pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.show()
    logger.info(f"Saved: {save_path}")


def plot_image_properties(df_props: pd.DataFrame, save_path: Path) -> None:
    """Plot distribusi properti gambar (width, aspect ratio, file size, color mode)."""
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Image Properties Distribution', fontsize=16, fontweight='bold', color=TEXT_COLOR)

    # Width Distribution
    ax = axes[0, 0]
    ax.hist(df_props['width'], bins=60, color=PALETTE[0], edgecolor=DARK_BG, linewidth=0.3, alpha=0.85)
    ax.axvline(df_props['width'].median(), color=PALETTE[3], linestyle='--', linewidth=1.5, label=f'Median: {df_props["width"].median():.0f}px')
    ax.set_title('Image Width Distribution', fontsize=12)
    ax.set_xlabel('Width (px)')
    ax.set_ylabel('Count')
    ax.legend(facecolor=PANEL_BG)

    # Aspect Ratio Distribution
    ax = axes[0, 1]
    ax.hist(df_props['aspect_ratio'], bins=50, color=PALETTE[1], edgecolor=DARK_BG, linewidth=0.3, alpha=0.85)
    ax.axvline(1.0, color=PALETTE[3], linestyle='--', linewidth=1.5, label='Square (1:1)')
    ax.set_title('Aspect Ratio Distribution', fontsize=12)
    ax.set_xlabel('Aspect Ratio (Width / Height)')
    ax.legend(facecolor=PANEL_BG)

    # File Size Distribution
    ax = axes[1, 0]
    # filter out outliers for better visualization
    q99 = df_props['file_size_kb'].quantile(0.99)
    filtered_sizes = df_props[df_props['file_size_kb'] < q99]['file_size_kb']
    ax.hist(filtered_sizes, bins=50, color=PALETTE[2], edgecolor=DARK_BG, linewidth=0.3, alpha=0.85)
    ax.set_title('File Size Distribution (99th percentile)', fontsize=12)
    ax.set_xlabel('File Size (KB)')
    ax.set_ylabel('Count')

    # Color Mode Distribution
    ax = axes[1, 1]
    mode_counts = df_props['mode'].value_counts()
    bars = ax.bar(mode_counts.index, mode_counts.values, color=PALETTE[4], edgecolor=DARK_BG, width=0.5)
    ax.set_title('Color Mode Distribution', fontsize=12)
    ax.set_xlabel('Color Mode')
    ax.set_yscale('log')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1, f'{int(bar.get_height()):,}', 
                ha='center', va='bottom', color=TEXT_COLOR, fontsize=10)

    for ax in axes.flatten():
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.show()
    logger.info(f"Saved: {save_path}")


def plot_sample_images(all_image_paths: Dict[str, List[Path]], class_names: Dict[str, str], save_path: Path, n_samples: int = 5) -> None:
    """Menampilkan grid gambar contoh dari setiap kelas."""
    set_plot_style()
    n_classes = len(all_image_paths)
    fig = plt.figure(figsize=(n_samples * 2.5, n_classes * 2.8))
    fig.suptitle('Representative Sample Images', fontsize=16, fontweight='bold', color=TEXT_COLOR, y=1.02)

    gs = gridspec.GridSpec(n_classes, n_samples, wspace=0.1, hspace=0.3)

    for i, (folder_name, paths) in enumerate(sorted(all_image_paths.items())):
        display_name = class_names.get(folder_name, folder_name)
        samples = random.sample(paths, min(n_samples, len(paths)))
        
        for j, img_path in enumerate(samples):
            ax = fig.add_subplot(gs[i, j])
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                size_str = f"{img.size[0]}x{img.size[1]}"
            except Exception:
                ax.text(0.5, 0.5, 'Corrupt', ha='center', va='center', color=TEXT_COLOR)
                size_str = "Error"
            
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COLOR)
                spine.set_linewidth(1.5)
            
            if j == 0:
                ax.set_ylabel(display_name, fontsize=12, fontweight='bold', color=TEXT_COLOR, rotation=0, labelpad=40, ha='right', va='center')
            
            ax.set_title(size_str, fontsize=9, color=TEXT_COLOR, pad=5)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.show()
    logger.info(f"Saved: {save_path}")
