"""
layers.py
=========
Reusable building blocks untuk arsitektur CNN.

ConvBlock adalah unit dasar yang digunakan oleh seluruh CNN dalam proyek ini.
Pola yang digunakan:

    Conv2d → BatchNorm2d → ReLU → MaxPool2d → Dropout2d

Alasan pemilihan setiap komponen:
  - Conv2d        : Ekstraksi fitur spasial (tekstur, tepi, bentuk)
  - BatchNorm2d   : Menstabilkan distribusi aktivasi → training lebih stabil & cepat
  - ReLU          : Non-linearitas standar, cepat, tidak menyebabkan vanishing gradient
  - MaxPool2d     : Reduksi dimensi spasial, mengambil fitur paling dominan per region
  - Dropout2d     : Regularisasi pada level channel → mencegah overfitting
"""

from typing import Optional
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional Block standar untuk CNN Baseline.

    Pipeline per block:
        Conv2d(in_ch → out_ch, kernel=3, padding=1)
        → BatchNorm2d(out_ch)
        → ReLU(inplace=True)
        → MaxPool2d(kernel=2, stride=2)
        → Dropout2d(p)

    Args:
        in_channels  : Jumlah channel input
        out_channels : Jumlah channel output (jumlah filter)
        kernel_size  : Ukuran kernel konvolusi (default 3)
        padding      : Padding (default 1, mempertahankan dimensi spasial)
        pool_size    : Ukuran kernel MaxPooling (default 2)
        dropout_p    : Probabilitas dropout channel (default 0.25)

    Example:
        >>> block = ConvBlock(in_channels=3, out_channels=32)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> out = block(x)
        >>> out.shape  # torch.Size([4, 32, 112, 112])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
        dropout_p: float = 0.25,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            # Konvolusi: belajar fitur visual (tekstur, tepi, pola)
            # padding='same' (padding=1, kernel=3) mempertahankan dimensi spasial
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,   # bias=False karena BatchNorm sudah menghandle bias
            ),

            # Batch Normalization: normalisasi distribusi aktivasi per batch
            # Mengurangi Internal Covariate Shift → konvergensi lebih cepat
            nn.BatchNorm2d(out_channels),

            # ReLU: aktivasi non-linear, memperkenalkan kemampuan belajar pola non-linear
            # inplace=True menghemat memori
            nn.ReLU(inplace=True),

            # MaxPooling: reduksi dimensi spasial 2× (224→112→56→28→14)
            # Mengambil nilai maksimum per region → mempertahankan fitur paling dominan
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),

            # Dropout2d: dropout pada level channel (bukan pixel)
            # Lebih efektif untuk data spasial karena menghilangkan channel sepenuhnya
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass melalui ConvBlock."""
        return self.block(x)
