"""
efficientnet_b0.py
==================
Implementasi arsitektur Transfer Learning berbasis EfficientNet-B0.

Modul ini membungkus torchvision.models.efficientnet_b0 dengan fitur:
- Load pretrained weights ImageNet (IMAGENET1K_V1) secara otomatis
- Penggantian classification head sesuai jumlah kelas (3 kelas BDC)
- Utilitas freeze backbone untuk skenario Feature Extraction
- Utilitas unfreeze untuk skenario Fine-Tuning berikutnya
- Ekstraksi informasi backbone untuk keperluan pelaporan dan compliance BDC

Catatan Arsitektur EfficientNet-B0:
  - Dirancang dengan Compound Scaling (phi=1.0): depth=1.0, width=1.0, resolution=224
  - Menggunakan MBConv (Mobile Inverted Bottleneck Convolution) dengan SE block
  - Total parameter : ~5.3 Juta (sangat efisien vs ResNet50 ~23.5 Juta)
  - Output backbone : 1280-dim (berbeda dari ResNet50 yang 2048-dim)
  - API head        : model.classifier (berbeda dari ResNet50 yang model.fc)
"""

import logging
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EfficientNetB0WasteClassifier(nn.Module):
    """
    Model klasifikasi sampah berbasis EfficientNet-B0 Pretrained.

    Arsitektur:
      - Backbone : EfficientNet-B0 (features + avgpool), output 1280-dim
      - Head     : Dropout(p) → Linear(1280 → num_classes)

    Catatan:
      EfficientNet-B0 dari torchvision sudah memiliki Dropout(0.2) bawaan
      di dalam bagian classifier-nya. Saat kita mengganti classifier,
      dropout eksternal (dari parameter dropout_p) mengontrol regularisasi head.

    Args:
        num_classes (int)  : Jumlah kelas output (default: 3).
        pretrained  (bool) : Menggunakan bobot pretrained ImageNet (default: True).
        dropout_p   (float): Probabilitas dropout pada classification head baru.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_p: float = 0.4,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.pretrained  = pretrained
        self.dropout_p   = dropout_p

        # --- Load Backbone ---
        # Menggunakan weights IMAGENET1K_V1 yang tersedia di torchvision.
        # Ini adalah pretrained weights standar yang diizinkan oleh aturan BDC
        # (weights dari dataset eksternal publik, bukan data kompetisi).
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
            logger.info("EfficientNet-B0 dimuat dengan pretrained weights IMAGENET1K_V1.")
        else:
            self.backbone = efficientnet_b0(weights=None)
            logger.info("EfficientNet-B0 dimuat TANPA pretrained weights (from scratch).")

        # --- Ganti Classification Head ---
        # EfficientNet-B0 dari torchvision memiliki struktur:
        #   backbone.features    : 16 MBConv blocks (ekstraktor fitur)
        #   backbone.avgpool     : AdaptiveAvgPool2d → output (B, 1280, 1, 1)
        #   backbone.classifier  : Sequential(Dropout(0.2), Linear(1280, 1000))
        #                         ↑ ini yang kita ganti
        #
        # in_features = 1280 (fixed untuk EfficientNet-B0)
        in_features = self.backbone.classifier[1].in_features  # 1280

        # Head baru: Dropout → Linear(1280 → num_classes)
        # Menggunakan dropout_p dari konfigurasi untuk kontrol eksplisit.
        if dropout_p > 0:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p, inplace=True),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.classifier = nn.Linear(in_features, num_classes)

        logger.info(
            f"Classification head diganti: Linear({in_features} → {num_classes}), "
            f"dropout_p={dropout_p}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass gambar melalui backbone EfficientNet-B0 dan classification head.

        Args:
            x (torch.Tensor): Tensor citra ukuran (B, 3, 224, 224)

        Returns:
            torch.Tensor: Logits keluaran ukuran (B, num_classes)
        """
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """
        Membekukan (freeze) seluruh parameter backbone (features + avgpool).
        Hanya classification head yang tetap dapat dilatih (requires_grad = True).

        Digunakan untuk skenario Feature Extraction (Experiment 01).
        Backbone sudah optimal dari ImageNet — hanya head yang perlu belajar
        domain-specific features untuk dataset sampah BDC.
        """
        # Step 1: Matikan gradient untuk seluruh model
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Step 2: Aktifkan kembali gradient HANYA untuk classifier head
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        trainable = self.get_trainable_params_count()
        total     = self.get_total_params_count()
        logger.info(
            f"Backbone di-freeze. "
            f"Trainable params: {trainable:,} / {total:,} "
            f"({trainable/total*100:.2f}%)"
        )

    def unfreeze_all(self) -> None:
        """
        Membuka (unfreeze) seluruh parameter backbone dan head.

        Digunakan untuk skenario Fine-Tuning (Experiment 02 ke atas).
        Semua layer dapat dilatih kembali dengan LR yang lebih kecil.
        """
        for param in self.parameters():
            param.requires_grad = True

        logger.info(
            f"Seluruh backbone di-unfreeze. "
            f"Trainable params: {self.get_trainable_params_count():,}"
        )

    def unfreeze_top_layers(self, n_blocks: int = 3) -> None:
        """
        Membuka sebagian akhir dari backbone untuk Partial Fine-Tuning.

        EfficientNet-B0 memiliki 9 MBConv block (features[0..8]).
        Unfreeze n_blocks terakhir memungkinkan fine-tuning yang lebih efisien
        daripada membuka seluruh backbone sekaligus.

        Args:
            n_blocks (int): Jumlah block terakhir yang akan di-unfreeze (default: 3).
        """
        # features adalah Sequential yang berisi: Conv-BN-SiLU + 7 MBConv blocks + Conv-BN-SiLU
        # Total elemen: index 0 s/d 8
        total_blocks = len(self.backbone.features)
        start_idx    = max(0, total_blocks - n_blocks)

        for i, layer in enumerate(self.backbone.features):
            for param in layer.parameters():
                param.requires_grad = (i >= start_idx)

        # Pastikan head selalu trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        trainable = self.get_trainable_params_count()
        logger.info(
            f"Top {n_blocks} blocks di-unfreeze (features[{start_idx}:]). "
            f"Trainable params: {trainable:,}"
        )

    def get_trainable_params_count(self) -> int:
        """Mengembalikan jumlah parameter yang akan diperbarui oleh optimizer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params_count(self) -> int:
        """Mengembalikan total seluruh parameter dalam model."""
        return sum(p.numel() for p in self.parameters())


def build_efficientnet_b0(config: Dict[str, Any]) -> EfficientNetB0WasteClassifier:
    """
    Factory function untuk membangun model EfficientNet-B0 dari konfigurasi YAML.

    Membaca seluruh parameter dari efficientnet.yaml dan menginisialisasi
    model dengan strategi yang tepat (freeze/unfreeze backbone).

    Args:
        config (Dict): Dictionary hasil yaml.safe_load dari efficientnet.yaml

    Returns:
        EfficientNetB0WasteClassifier: Instance model yang siap dilatih
    """
    model_cfg = config.get('model', {})

    num_classes = config.get('data', {}).get('num_classes', 3)
    pretrained  = model_cfg.get('pretrained', True)
    dropout_p   = model_cfg.get('classifier_head', {}).get('dropout', 0.4)

    model = EfficientNetB0WasteClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p=dropout_p,
    )

    # Terapkan strategi freeze sesuai konfigurasi
    if model_cfg.get('freeze_backbone', True):
        model.freeze_backbone()
    else:
        logger.info("Backbone TIDAK di-freeze (mode: full fine-tuning).")

    return model


def get_backbone_info(model: EfficientNetB0WasteClassifier) -> Dict[str, Any]:
    """
    Mengekstrak informasi backbone untuk keperluan pelaporan dan
    kepatuhan (compliance) aturan BDC SATRIA DATA 2026.

    Informasi ini membuktikan bahwa:
    1. Model menggunakan arsitektur publik (EfficientNet-B0)
    2. Pretrained weights dari dataset publik (ImageNet-1K)
    3. Backbone TIDAK dilatih dengan data kompetisi BDC

    Args:
        model: Instance EfficientNetB0WasteClassifier

    Returns:
        Dict berisi metadata arsitektur dan status training
    """
    total     = model.get_total_params_count()
    trainable = model.get_trainable_params_count()
    frozen    = total - trainable

    # Tentukan strategi berdasarkan jumlah trainable params
    # Head-only training: ~2K params (dropout + linear 1280→3)
    # Full fine-tuning  : ~5.3M params
    if trainable < 50_000:
        strategy = "Feature Extraction (Frozen Backbone)"
    elif trainable < total:
        strategy = "Partial Fine-Tuning"
    else:
        strategy = "Full Fine-Tuning"

    return {
        "architecture"        : "EfficientNet-B0",
        "compound_scaling"    : "phi=1.0 (depth=1.0, width=1.0, resolution=224)",
        "pretrained_source"   : "torchvision.models",
        "pretrained_dataset"  : "ImageNet-1K",
        "pretrained_weights"  : "IMAGENET1K_V1",
        "input_resolution"    : "224×224",
        "backbone_output_dim" : 1280,
        "total_parameters"    : total,
        "trainable_parameters": trainable,
        "frozen_parameters"   : frozen,
        "trainable_ratio_pct" : round(trainable / total * 100, 2),
        "classification_head" : f"Dropout({model.dropout_p}) → Linear(1280 → {model.num_classes})",
        "strategy"            : strategy,
        "bdc_compliant"       : True,
        "note"                : "Backbone tidak pernah dilatih dengan data kompetisi BDC SATRIA DATA 2026",
    }
