"""
waste_dataset.py
================
Custom PyTorch Dataset untuk Smart Waste Classification.

Dataset class ini menerima DataFrame berisi filepath dan label,
sehingga mendukung in-memory split tanpa perlu menyalin file.

Fitur:
  - Menerima list path gambar dan label (dari splitter.py)
  - Mendukung transforms yang berbeda per split
  - Menangani gambar corrupt secara graceful
  - Menyediakan informasi kelas yang lengkap
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


CLASS_NAMES = ['Recyclable', 'Electronic', 'Organic']


class WasteDataset(Dataset):
    """
    Custom PyTorch Dataset untuk dataset klasifikasi sampah BDC 2026.

    Menerima DataFrame dari splitter.py dengan kolom:
      - 'filepath'   : Path lengkap ke file gambar
      - 'label'      : Integer label (0, 1, 2)
      - 'class_name' : Nama kelas (Recyclable, Electronic, Organic)

    Args:
        dataframe  : pd.DataFrame berisi data split
        transform  : torchvision.transforms pipeline (opsional)
        split_name : Nama split ('train', 'val', 'test') untuk logging

    Example:
        >>> train_dataset = WasteDataset(
        ...     dataframe=df_train,
        ...     transform=get_train_transforms(),
        ...     split_name='train'
        ... )
        >>> img, label = train_dataset[0]
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
        split_name: str = 'unknown',
    ) -> None:
        self.df         = dataframe.reset_index(drop=True)
        self.transform  = transform
        self.split_name = split_name
        self.class_names = CLASS_NAMES

        # Validasi kolom yang dibutuhkan
        required_cols = {'filepath', 'label'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame kurang kolom: {missing}")

        logger.info(
            f"WasteDataset [{split_name}] dibuat: "
            f"{len(self.df):,} gambar, "
            f"transform={'Yes' if transform else 'No'}"
        )

    def __len__(self) -> int:
        """Jumlah total gambar dalam dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Mengambil satu sampel dari dataset.

        Args:
            idx : Index sampel

        Returns:
            Tuple[image_tensor, label] :
              - image_tensor : FloatTensor hasil transform
              - label        : Integer class label

        Note:
            Gambar corrupt akan di-skip dengan mengambil sampel berikutnya.
            Ini mencegah runtime error saat training.
        """
        row       = self.df.iloc[idx]
        filepath  = Path(row['filepath'])
        label     = int(row['label'])

        # Load gambar dengan graceful fallback untuk corrupt files
        try:
            image = Image.open(filepath).convert('RGB')
        except (UnidentifiedImageError, OSError, Exception) as e:
            logger.warning(f"Gambar tidak bisa dibaca: {filepath.name} — {e}")
            # Fallback: ambil sampel berikutnya (circular)
            fallback_idx = (idx + 1) % len(self.df)
            return self.__getitem__(fallback_idx)

        # Terapkan transformasi jika ada
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_counts(self) -> dict:
        """
        Menghitung jumlah gambar per kelas.

        Returns:
            dict : {class_name: count}
        """
        return self.df['class_name'].value_counts().to_dict()

    def get_class_weights(self) -> torch.Tensor:
        """
        Menghitung bobot kelas untuk WeightedRandomSampler atau loss weighting.

        Bobot inversely proportional terhadap frekuensi kelas:
          weight[c] = total_samples / (num_classes * count[c])

        Returns:
            torch.Tensor : Tensor bobot kelas shape [num_classes]
        """
        counts = self.df['label'].value_counts().sort_index()
        total  = len(self.df)
        n_cls  = len(CLASS_NAMES)
        weights = total / (n_cls * counts.values)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Menghasilkan bobot per sampel untuk WeightedRandomSampler.

        Setiap sampel mendapat bobot kelas-nya masing-masing,
        sehingga kelas minoritas lebih sering di-sample.

        Returns:
            torch.Tensor : Tensor bobot shape [len(dataset)]
        """
        class_weights = self.get_class_weights()
        sample_weights = torch.tensor(
            [float(class_weights[label]) for label in self.df['label']],
            dtype=torch.float32
        )
        return sample_weights

    def __repr__(self) -> str:
        counts = self.get_class_counts()
        dist_str = ', '.join(f"{k}:{v}" for k, v in counts.items())
        return (
            f"WasteDataset(split='{self.split_name}', "
            f"n={len(self)}, "
            f"distribution=[{dist_str}])"
        )
