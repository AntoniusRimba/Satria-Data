"""
resnet50.py
===========
Implementasi arsitektur Transfer Learning berbasis ResNet50.

Modul ini membungkus torchvision.models.resnet50 dengan fitur:
- Load pretrained weights ImageNet secara otomatis
- Penggantian classification head sesuai jumlah kelas (3 kelas BDC)
- Utilitas freeze backbone (Feature Extraction)
- Ekstraksi informasi backbone untuk dokumentasi
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from typing import List, Dict, Any

class ResNet50WasteClassifier(nn.Module):
    """
    Model klasifikasi sampah berbasis ResNet50 Pretrained.
    
    Arsitektur:
      - Backbone : ResNet50 (sampai layer AvgPool)
      - Head     : Linear(2048 -> num_classes)
    
    Args:
        num_classes (int): Jumlah kelas output (default: 3).
        pretrained (bool): Menggunakan bobot ImageNet (default: True).
        dropout_p (float): Probabilitas dropout pada classification head.
    """
    
    def __init__(
        self, 
        num_classes: int = 3, 
        pretrained: bool = True,
        dropout_p: float = 0.5
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_p = dropout_p
        
        # Load backbone
        if pretrained:
            # Menggunakan weights default (IMAGENET1K_V1 atau V2)
            weights = ResNet50_Weights.DEFAULT
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)
            
        # Dapatkan ukuran input (in_features) ke layer fully connected terakhir
        in_features = self.backbone.fc.in_features
        
        # Ganti classification head (fc)
        # Sesuai aturan feature extraction, kita buang head lama dan pasang yang baru
        if dropout_p > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass gambar melalui backbone dan head.
        
        Args:
            x (torch.Tensor): Tensor citra ukuran (B, 3, 224, 224)
            
        Returns:
            torch.Tensor: Logits keluaran ukuran (B, num_classes)
        """
        return self.backbone(x)
        
    def freeze_backbone(self) -> None:
        """
        Membekukan (freeze) seluruh parameter backbone.
        Hanya classification head yang bisa dilatih (requires_grad = True).
        Dipanggil untuk skenario Feature Extraction.
        """
        # Matikan grad untuk seluruh parameter
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Nyalakan grad HANYA untuk layer fc (classification head)
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
            
    def unfreeze_all(self) -> None:
        """
        Membuka (unfreeze) seluruh parameter backbone dan head.
        Dipanggil untuk skenario Fine-Tuning.
        """
        for param in self.parameters():
            param.requires_grad = True
            
    def get_trainable_params_count(self) -> int:
        """Mengembalikan jumlah parameter yang akan diperbarui optimizer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def get_total_params_count(self) -> int:
        """Mengembalikan total jumlah seluruh parameter dalam model."""
        return sum(p.numel() for p in self.parameters())


def build_resnet50(config: Dict[str, Any]) -> ResNet50WasteClassifier:
    """
    Factory function untuk membangun model ResNet50 berdasarkan konfigurasi.
    
    Args:
        config (Dict): Dictionary dari resnet50.yaml
        
    Returns:
        ResNet50WasteClassifier: Instansiasi model siap pakai
    """
    model_cfg = config.get('model', {})
    
    num_classes = config.get('data', {}).get('num_classes', 3)
    pretrained  = model_cfg.get('pretrained', True)
    dropout_p   = model_cfg.get('classifier_head', {}).get('dropout', 0.5)
    
    model = ResNet50WasteClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p=dropout_p
    )
    
    # Terapkan strategi freeze backbone (Feature Extraction)
    if model_cfg.get('freeze_backbone', True):
        model.freeze_backbone()
        
    return model


def get_backbone_info(model: ResNet50WasteClassifier) -> Dict[str, Any]:
    """
    Mengekstrak informasi backbone untuk keperluan pelaporan
    dan kepatuhan (compliance) BDC SATRIA DATA 2026.
    
    Returns:
        Dict berisi metadata pretrained architecture
    """
    total = model.get_total_params_count()
    trainable = model.get_trainable_params_count()
    frozen = total - trainable
    
    return {
        "architecture": "ResNet50",
        "pretrained_source": "torchvision.models",
        "pretrained_dataset": "ImageNet-1K",
        "input_resolution": "224x224",
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "classification_head": "Dropout + Linear" if model.dropout_p > 0 else "Linear",
        "strategy": "Feature Extraction" if trainable < 100_000 else "Fine-Tuning",
        "bdc_compliant": True,
        "note": "Backbone tidak pernah dilatih dengan data kompetisi BDC"
    }
