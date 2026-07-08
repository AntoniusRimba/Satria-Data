"""
visualizer.py
=============
Fungsi-fungsi visualisasi untuk evaluasi model.

Menghasilkan:
  - Confusion Matrix (heatmap profesional)
  - Learning Curve (loss, accuracy, macro f1)
  - Per-class metric comparison (bar chart)
  - Error Analysis sample grid

Semua visualisasi disimpan dengan prefix 'baseline_eval_'.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

logger = logging.getLogger(__name__)

CLASS_NAMES = ['Recyclable', 'Electronic', 'Organic']

# Palet warna konsisten dengan notebook sebelumnya
DARK_BG    = '#1e1e2e'
PANEL_BG   = '#313244'
TEXT_COLOR = '#cdd6f4'
GRID_COLOR = '#45475a'
COLORS     = ['#89dceb', '#a6e3a1', '#fab387']
ACCENT     = '#f38ba8'


def _base_style(ax):
    """Terapkan gaya dasar ke axes."""
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(alpha=0.3, color=GRID_COLOR)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[Path] = None,
    experiment_name: str = 'Baseline CNN',
) -> None:
    """
    Visualisasi confusion matrix sebagai heatmap.

    Confusion matrix menunjukkan:
      - Diagonal: prediksi benar per kelas
      - Off-diagonal: jenis kesalahan klasifikasi
      - Pola error: kelas mana yang sering tertukar

    Args:
        cm              : Confusion matrix [num_classes, num_classes]
        class_names     : Nama kelas
        save_path       : Path untuk menyimpan gambar
        experiment_name : Judul plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f'Confusion Matrix — {experiment_name}',
                 fontsize=15, fontweight='bold', color=TEXT_COLOR)

    # --- Panel kiri: Count ---
    ax = axes[0]
    ax.set_facecolor(DARK_BG)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor=DARK_BG,
        annot_kws={'size': 12, 'weight': 'bold'},
    )
    ax.set_title('Count', fontsize=12, color=TEXT_COLOR, pad=10)
    ax.set_xlabel('Predicted Label', color=TEXT_COLOR)
    ax.set_ylabel('True Label', color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)

    # --- Panel kanan: Normalized (row-wise = recall per kelas) ---
    ax = axes[1]
    ax.set_facecolor(DARK_BG)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt='.3f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor=DARK_BG,
        vmin=0, vmax=1,
        annot_kws={'size': 11, 'weight': 'bold'},
    )
    ax.set_title('Normalized (Row = Recall per Class)', fontsize=12, color=TEXT_COLOR, pad=10)
    ax.set_xlabel('Predicted Label', color=TEXT_COLOR)
    ax.set_ylabel('True Label', color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_learning_curves(
    history_df: pd.DataFrame,
    best_epoch: int,
    save_path: Optional[Path] = None,
    experiment_name: str = 'Baseline CNN',
) -> None:
    """
    Visualisasi learning curves: Loss, Accuracy, Macro F1.

    Kurva ini membantu mendiagnosis:
      - Healthy convergence: val_loss turun, train_loss turun bersama
      - Overfitting: train_loss turun, val_loss naik / divergen
      - Underfitting: keduanya masih tinggi setelah banyak epoch

    Args:
        history_df      : DataFrame training history
        best_epoch      : Epoch dengan val_macro_f1 terbaik
        save_path       : Path untuk menyimpan gambar
        experiment_name : Judul plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f'Learning Curves — {experiment_name}',
                 fontsize=15, fontweight='bold', color=TEXT_COLOR)

    plot_pairs = [
        ('train_loss',     'val_loss',     'Loss',      '#f38ba8', '#89dceb'),
        ('train_acc',      'val_acc',      'Accuracy',  '#a6e3a1', '#fab387'),
        ('train_macro_f1', 'val_macro_f1', 'Macro F1',  '#cba6f7', '#f9e2af'),
    ]

    epochs = history_df['epoch'].tolist()

    for ax, (tc_col, vc_col, title, tc, vc) in zip(axes, plot_pairs):
        _base_style(ax)
        ax.plot(epochs, history_df[tc_col], color=tc, label='Train', linewidth=2, alpha=0.9)
        ax.plot(epochs, history_df[vc_col], color=vc, label='Val',   linewidth=2, linestyle='--', alpha=0.9)
        ax.axvline(x=best_epoch, color='#f9e2af', linestyle=':', linewidth=1.5, alpha=0.8, label=f'Best Ep.{best_epoch}')
        ax.set_title(title, fontsize=13, color=TEXT_COLOR)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT_COLOR, framealpha=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_per_class_metrics(
    metrics: Dict,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[Path] = None,
    experiment_name: str = 'Baseline CNN',
) -> None:
    """
    Bar chart metrik per kelas: Precision, Recall, F1.

    Berguna untuk:
      - Mengidentifikasi kelas mana yang paling lemah
      - Melihat trade-off precision vs recall per kelas
      - Menentukan prioritas perbaikan

    Args:
        metrics         : Output dari compute_metrics()
        class_names     : Nama kelas
        save_path       : Path simpan gambar
        experiment_name : Judul plot
    """
    per_class = metrics.get('per_class', {})
    prec  = [per_class[c]['precision'] for c in class_names]
    rec   = [per_class[c]['recall']    for c in class_names]
    f1    = [per_class[c]['f1']        for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    _base_style(ax)

    bars_p = ax.bar(x - width,      prec, width, label='Precision', color='#89dceb', edgecolor=DARK_BG)
    bars_r = ax.bar(x,              rec,  width, label='Recall',    color='#a6e3a1', edgecolor=DARK_BG)
    bars_f = ax.bar(x + width,      f1,   width, label='F1 Score',  color='#fab387', edgecolor=DARK_BG)

    # Garis macro F1 keseluruhan
    macro_f1 = metrics['macro_f1']
    ax.axhline(y=macro_f1, color='#f9e2af', linestyle='--', linewidth=1.5,
               label=f'Macro F1 = {macro_f1:.4f}')

    # Label nilai di atas bar
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8, color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Per-Class Metrics — {experiment_name}', fontsize=13, color=TEXT_COLOR)
    ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT_COLOR, framealpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_error_analysis(
    misclassified: List[Dict],
    max_show: int = 12,
    save_path: Optional[Path] = None,
    experiment_name: str = 'Baseline CNN',
) -> None:
    """
    Visualisasi grid gambar yang salah diklasifikasikan.

    Error analysis membantu memahami:
      - Apakah model kesulitan pada kondisi pencahayaan tertentu?
      - Apakah ada ambiguitas visual antar kelas?
      - Apakah ada pola sistematis dalam kesalahan?

    Args:
        misclassified   : Output dari get_misclassified_samples()
        max_show        : Jumlah gambar yang ditampilkan
        save_path       : Path simpan gambar
        experiment_name : Judul plot
    """
    samples = misclassified[:max_show]
    if not samples:
        logger.info("Tidak ada sampel salah klasifikasi untuk ditampilkan.")
        return

    n_cols = 4
    n_rows = (len(samples) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f'Error Analysis — {experiment_name}\n(Gambar yang Salah Diklasifikasikan)',
                 fontsize=13, fontweight='bold', color=TEXT_COLOR)

    axes = np.array(axes).flatten()

    for ax_idx, (ax, sample) in enumerate(zip(axes, samples)):
        try:
            img = Image.open(sample['filepath']).convert('RGB')
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'Load Error', ha='center', va='center',
                    color=TEXT_COLOR, transform=ax.transAxes)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(DARK_BG)

        # Border merah = salah
        for spine in ax.spines.values():
            spine.set_edgecolor(ACCENT)
            spine.set_linewidth(2)
            spine.set_visible(True)

        ax.set_title(
            f"True: {sample['true_class']}\nPred: {sample['pred_class']}",
            fontsize=8, color=TEXT_COLOR, pad=4
        )

    # Sembunyikan axes kosong
    for ax in axes[len(samples):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        logger.info(f"Saved: {save_path}")
    plt.show()
