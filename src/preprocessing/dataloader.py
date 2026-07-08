"""
dataloader.py
=============
Factory functions untuk membangun DataLoader Training dan Validation.

Modul ini mengintegrasikan WasteDataset + transforms + WeightedRandomSampler
menjadi DataLoader yang siap digunakan oleh training loop.

Desain:
  - Training DataLoader : menggunakan WeightedRandomSampler untuk
    menangani class imbalance secara transparan
  - Validation DataLoader : deterministik, tanpa sampling khusus
  - Semua parameter dikonfigurasi dari baseline.yaml
"""

import logging
from typing import Tuple, Optional
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.datasets.waste_dataset import WasteDataset
from src.preprocessing.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


def build_train_loader(
    df_train: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 224,
    resize_to: int = 256,
    mean: list = [0.485, 0.456, 0.406],
    std: list  = [0.229, 0.224, 0.225],
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> DataLoader:
    """
    Membangun DataLoader untuk Training Set.

    Menggunakan WeightedRandomSampler jika use_weighted_sampler=True,
    sehingga setiap batch memiliki distribusi kelas yang seimbang.
    Ini lebih baik dari shuffle biasa untuk dataset yang imbalanced.

    Args:
        df_train             : Training DataFrame dari splitter
        batch_size           : Ukuran batch
        num_workers          : Jumlah worker untuk parallel loading
        pin_memory           : Aktifkan untuk mempercepat transfer ke GPU
        image_size           : Ukuran gambar input CNN
        resize_to            : Ukuran resize sebelum crop
        mean                 : Mean normalisasi
        std                  : Std normalisasi
        use_weighted_sampler : Gunakan WeightedRandomSampler
        seed                 : Random seed

    Returns:
        DataLoader : Train DataLoader yang siap digunakan
    """
    transform = get_train_transforms(
        image_size=image_size,
        resize_to=resize_to,
        mean=mean,
        std=std,
    )

    dataset = WasteDataset(
        dataframe=df_train,
        transform=transform,
        split_name='train',
    )

    # WeightedRandomSampler: menangani class imbalance
    # Setiap epoch, sampler mengambil len(dataset) sampel dengan probabilitas
    # yang berbeda per kelas, sehingga kelas minoritas lebih sering muncul.
    if use_weighted_sampler:
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        shuffle = False  # Tidak bisa digunakan bersamaan dengan sampler
        logger.info("WeightedRandomSampler diaktifkan untuk training.")
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,     # Drop batch terakhir jika tidak lengkap → gradient lebih stabil
    )

    logger.info(
        f"Train DataLoader: {len(dataset):,} gambar, "
        f"batch_size={batch_size}, "
        f"num_batches={len(loader)}"
    )
    return loader


def build_val_loader(
    df_val: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 224,
    resize_to: int = 256,
    mean: list = [0.485, 0.456, 0.406],
    std: list  = [0.229, 0.224, 0.225],
) -> DataLoader:
    """
    Membangun DataLoader untuk Validation Set.

    Tidak ada augmentasi, tidak ada sampling khusus.
    Deterministik untuk hasil evaluasi yang konsisten.

    Args:
        df_val     : Validation DataFrame dari splitter
        batch_size : Ukuran batch (bisa lebih besar dari train untuk kecepatan)
        num_workers: Jumlah worker
        pin_memory : Aktifkan untuk GPU transfer
        image_size : Ukuran gambar input
        resize_to  : Ukuran resize sebelum crop
        mean       : Mean normalisasi
        std        : Std normalisasi

    Returns:
        DataLoader : Validation DataLoader deterministik
    """
    transform = get_val_transforms(
        image_size=image_size,
        resize_to=resize_to,
        mean=mean,
        std=std,
    )

    dataset = WasteDataset(
        dataframe=df_val,
        transform=transform,
        split_name='val',
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,     # WAJIB False — evaluasi harus deterministik
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,   # Evaluasi seluruh gambar tanpa ada yang di-drop
    )

    logger.info(
        f"Val DataLoader: {len(dataset):,} gambar, "
        f"batch_size={batch_size}, "
        f"num_batches={len(loader)}"
    )
    return loader


def build_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    config: dict,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory utama: membangun train dan val DataLoader dari konfigurasi.

    Args:
        df_train             : Training DataFrame
        df_val               : Validation DataFrame
        config               : Dictionary konfigurasi dari baseline.yaml
        use_weighted_sampler : Gunakan WeightedRandomSampler pada training

    Returns:
        Tuple[train_loader, val_loader]
    """
    pre_cfg    = config.get('preprocessing', {})
    dl_cfg     = config.get('dataloader', {})
    exp_cfg    = config.get('experiment', {})

    image_size  = pre_cfg.get('image_size', 224)
    resize_to   = pre_cfg.get('resize_to', 256)
    mean        = pre_cfg.get('normalize', {}).get('mean', [0.485, 0.456, 0.406])
    std         = pre_cfg.get('normalize', {}).get('std',  [0.229, 0.224, 0.225])
    batch_size  = dl_cfg.get('batch_size', 32)
    num_workers = dl_cfg.get('num_workers', 4)
    pin_memory  = dl_cfg.get('pin_memory', True)
    seed        = exp_cfg.get('seed', 42)

    train_loader = build_train_loader(
        df_train=df_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        image_size=image_size,
        resize_to=resize_to,
        mean=mean,
        std=std,
        use_weighted_sampler=use_weighted_sampler,
        seed=seed,
    )

    val_loader = build_val_loader(
        df_val=df_val,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        image_size=image_size,
        resize_to=resize_to,
        mean=mean,
        std=std,
    )

    return train_loader, val_loader


def verify_dataloader(
    loader: DataLoader,
    split_name: str = 'train',
    n_batches: int = 1,
) -> dict:
    """
    Verifikasi DataLoader: ambil beberapa batch dan periksa shape & range.

    Args:
        loader     : DataLoader yang akan diverifikasi
        split_name : Nama split untuk logging
        n_batches  : Jumlah batch yang diperiksa

    Returns:
        dict : Informasi verifikasi {'shape', 'dtype', 'min', 'max', 'labels'}
    """
    info = {}
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        info = {
            'batch_index'  : i,
            'split'        : split_name,
            'images_shape' : tuple(images.shape),
            'labels_shape' : tuple(labels.shape),
            'dtype'        : str(images.dtype),
            'pixel_min'    : round(float(images.min()), 4),
            'pixel_max'    : round(float(images.max()), 4),
            'pixel_mean'   : round(float(images.mean()), 4),
            'unique_labels': sorted(labels.unique().tolist()),
        }
        logger.info(
            f"[{split_name}] Batch {i}: "
            f"shape={info['images_shape']}, "
            f"range=[{info['pixel_min']:.3f}, {info['pixel_max']:.3f}]"
        )
    return info
