"""
baseline.py
===========
Arsitektur CNN Baseline untuk Smart Waste Classification.

Arsitektur:
    Input [B, 3, 224, 224]
      → ConvBlock 1: 3  → 32  ch  [B, 32,  112, 112]
      → ConvBlock 2: 32 → 64  ch  [B, 64,  56,  56 ]
      → ConvBlock 3: 64 → 128 ch  [B, 128, 28,  28 ]
      → ConvBlock 4: 128→ 256 ch  [B, 256, 14,  14 ]
      → AdaptiveAvgPool2d(1, 1)   [B, 256, 1,   1  ]
      → Flatten                   [B, 256]
      → Linear(256 → 128)
      → ReLU
      → Dropout(0.5)
      → Linear(128 → 3)           [B, 3]  ← logits

Desain Rationale:
  - 4 ConvBlocks cukup untuk menangkap fitur tekstur dan bentuk objek sampah
  - Channel scaling 32→64→128→256 mengikuti pola standar CNN modern
  - Global Average Pooling mengurangi parameter secara dramatis vs Flatten
  - Dropout(0.5) pada head mencegah overfitting pada feature vector
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from src.models.cnn.layers import ConvBlock


class BaselineCNN(nn.Module):
    """
    Custom CNN Baseline untuk klasifikasi sampah 3 kelas.

    Dirancang sebagai model baseline:
      - Sederhana dan mudah dijelaskan
      - Modular (dapat dimodifikasi dengan mengganti ConvBlock)
      - Bukan model terbaik, tetapi menjadi acuan terukur

    Args:
        num_classes      : Jumlah kelas output (default 3)
        dropout_conv     : Dropout rate pada setiap ConvBlock (default 0.25)
        dropout_fc       : Dropout rate pada Fully Connected head (default 0.5)
        conv_channels    : List jumlah filter per ConvBlock

    Example:
        >>> model = BaselineCNN(num_classes=3)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> logits = model(x)
        >>> logits.shape  # torch.Size([4, 3])
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout_conv: float = 0.25,
        dropout_fc: float = 0.5,
        conv_channels: List[int] = [32, 64, 128, 256],
    ) -> None:
        super().__init__()

        self.num_classes   = num_classes
        self.conv_channels = conv_channels

        # ── Feature Extractor (4 ConvBlocks) ──────────────────────────────
        # Setiap block: Conv2d → BN → ReLU → MaxPool → Dropout2d
        # Input: [B, 3, 224, 224]
        # Output setelah 4 MaxPool 2×: [B, 256, 14, 14]
        in_channels = 3
        blocks = []
        for out_ch in conv_channels:
            blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_ch,
                    dropout_p=dropout_conv,
                )
            )
            in_channels = out_ch
        self.features = nn.Sequential(*blocks)

        # ── Global Average Pooling ─────────────────────────────────────────
        # Menggantikan Flatten + FC besar.
        # Mengkompresi [B, 256, 14, 14] → [B, 256, 1, 1]
        # Keuntungan:
        #   - Parameter jauh lebih sedikit
        #   - Lebih robust terhadap variasi ukuran input
        #   - Mengurangi overfitting
        #   - Lebih dekat dengan arsitektur ResNet/EfficientNet (transfer learning)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Classifier Head ───────────────────────────────────────────────
        # Linear(256 → 128) → ReLU → Dropout(0.5) → Linear(128 → 3)
        # Dua lapis FC memberikan kapasitas yang cukup untuk klasifikasi
        # tanpa menambah terlalu banyak parameter.
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(128, num_classes),
        )

        # Inisialisasi bobot dengan He initialization (optimal untuk ReLU)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass CNN Baseline.

        Args:
            x : Input tensor [B, 3, 224, 224]

        Returns:
            logits : Output tensor [B, num_classes]
                     (raw logits, bukan probability → gunakan CrossEntropyLoss)
        """
        # Feature extraction melalui 4 ConvBlocks
        x = self.features(x)               # [B, 256, 14, 14]

        # Global Average Pooling
        x = self.global_avg_pool(x)        # [B, 256, 1, 1]
        x = x.flatten(start_dim=1)         # [B, 256]

        # Classification head
        x = self.classifier(x)             # [B, 3]

        return x

    def _initialize_weights(self) -> None:
        """
        Inisialisasi bobot dengan He (Kaiming) initialization.

        He initialization optimal untuk aktivasi ReLU karena
        mempertahankan variance signal melalui layer-layer.
        Mencegah vanishing/exploding gradient di awal training.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def get_model_info(self) -> Dict:
        """
        Mengembalikan informasi model: jumlah parameter, arsitektur.

        Returns:
            dict : Informasi model
        """
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'model_name'      : 'BaselineCNN',
            'num_classes'     : self.num_classes,
            'conv_channels'   : self.conv_channels,
            'total_params'    : total_params,
            'trainable_params': trainable_params,
            'non_trainable'   : total_params - trainable_params,
        }

    def __repr__(self) -> str:
        info = self.get_model_info()
        return (
            f"BaselineCNN(\n"
            f"  conv_channels  = {info['conv_channels']}\n"
            f"  num_classes    = {info['num_classes']}\n"
            f"  total_params   = {info['total_params']:,}\n"
            f"  trainable      = {info['trainable_params']:,}\n"
            f")"
        )


def build_baseline_cnn(config: dict) -> BaselineCNN:
    """
    Factory function: membangun BaselineCNN dari konfigurasi YAML.

    Args:
        config : Dictionary konfigurasi (dari baseline.yaml)

    Returns:
        BaselineCNN : Model yang siap untuk training
    """
    num_classes = config.get('data', {}).get('num_classes', 3)
    model = BaselineCNN(num_classes=num_classes)
    return model
