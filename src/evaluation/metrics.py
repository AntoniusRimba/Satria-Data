"""
metrics.py
==========
Fungsi-fungsi kalkulasi metrik evaluasi untuk Smart Waste Classification.

Metrik utama: Macro F1 Score (sesuai metrik resmi BDC SATRIA DATA 2026).
Metrik pendukung: Accuracy, Precision, Recall, Classification Report.

Desain:
  - Semua kalkulasi menggunakan sklearn untuk konsistensi
  - Hasil selalu berbentuk dict yang mudah di-serialize ke JSON
  - Setiap fungsi dapat digunakan ulang di eksperimen berbeda
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ['Recyclable', 'Electronic', 'Organic']


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Menjalankan inferensi pada seluruh DataLoader.

    Tidak ada gradient update — murni forward pass.
    Mengumpulkan prediksi, label asli, dan probabilitas per kelas.

    Args:
        model  : Model PyTorch (sudah load best checkpoint)
        loader : DataLoader (validation atau test)
        device : torch.device

    Returns:
        Tuple[all_preds, all_labels, all_probs]:
          - all_preds  : array label prediksi [N]
          - all_labels : array label asli     [N]
          - all_probs  : array probabilitas   [N, num_classes]
    """
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Dict:
    """
    Menghitung seluruh metrik evaluasi.

    Prioritas: Macro F1 Score (metrik kompetisi BDC).

    Mengapa Macro F1, bukan hanya Accuracy?
    - Accuracy bisa menyesatkan jika kelas tidak seimbang.
    - Macro F1 menghitung F1 tiap kelas secara terpisah lalu dirata-rata,
      sehingga setiap kelas diperlakukan setara tanpa memandang frekuensinya.
    - Ini memastikan model tidak hanya baik di kelas mayoritas.

    Args:
        y_true       : Array label asli [N]
        y_pred       : Array label prediksi [N]
        class_names  : Nama kelas untuk laporan per kelas

    Returns:
        dict : Semua metrik evaluasi
    """
    macro_f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # Metrik per kelas
    f1_per_class  = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_per_cls  = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_cls   = recall_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, cls_name in enumerate(class_names):
        per_class[cls_name] = {
            'precision': round(float(prec_per_cls[i]), 4),
            'recall'   : round(float(rec_per_cls[i]),  4),
            'f1'       : round(float(f1_per_class[i]), 4),
            'support'  : int(np.sum(y_true == i)),
        }

    return {
        'macro_f1'  : round(float(macro_f1),  4),
        'accuracy'  : round(float(accuracy),  4),
        'precision' : round(float(precision), 4),
        'recall'    : round(float(recall),    4),
        'per_class' : per_class,
    }


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> str:
    """
    Menghasilkan sklearn classification report sebagai string.

    Args:
        y_true       : Array label asli
        y_pred       : Array label prediksi
        class_names  : Nama kelas

    Returns:
        str : Classification report terformat
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Menghitung confusion matrix.

    Args:
        y_true : Array label asli
        y_pred : Array label prediksi

    Returns:
        np.ndarray : Confusion matrix [num_classes, num_classes]
    """
    return confusion_matrix(y_true, y_pred)


def save_metrics(
    metrics: Dict,
    output_dir: Path,
    filename: str = 'baseline_evaluation_metrics.json',
) -> None:
    """
    Menyimpan metrik evaluasi ke JSON.

    Args:
        metrics    : Dictionary metrik dari compute_metrics()
        output_dir : Direktori tujuan
        filename   : Nama file output
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {path}")


def get_misclassified_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filepaths: List[str],
    class_names: List[str] = CLASS_NAMES,
    max_samples: int = 20,
) -> List[Dict]:
    """
    Mengambil sampel yang salah diklasifikasikan untuk error analysis.

    Args:
        y_true      : Label asli
        y_pred      : Label prediksi
        filepaths   : Path gambar (sesuai urutan DataLoader)
        class_names : Nama kelas
        max_samples : Jumlah maksimum sampel yang dikembalikan

    Returns:
        List[Dict] : Daftar sampel salah beserta info
    """
    wrong_indices = np.where(y_true != y_pred)[0]
    results = []
    for idx in wrong_indices[:max_samples]:
        results.append({
            'filepath'      : filepaths[idx],
            'true_label'    : int(y_true[idx]),
            'pred_label'    : int(y_pred[idx]),
            'true_class'    : class_names[y_true[idx]],
            'pred_class'    : class_names[y_pred[idx]],
        })
    return results
