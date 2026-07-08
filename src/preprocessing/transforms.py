"""
transforms.py
=============
Preprocessing dan augmentasi pipeline untuk Smart Waste Classification.

Modul ini mendefinisikan dua pipeline transformasi terpisah:
  - get_train_transforms() : Resize + Augmentasi + Normalize (Training Set)
  - get_val_transforms()   : Resize + CenterCrop + Normalize (Validation / Test Set)

Desain:
  Augmentasi HANYA diterapkan pada Training Set.
  Validation dan Test Set menggunakan transformasi deterministik (tanpa randomness).
  Ini mencegah data leakage evaluasi dan memastikan konsistensi metrik.
"""

from typing import Tuple, List
from torchvision import transforms


def get_train_transforms(
    image_size: int = 224,
    resize_to: int = 256,
    mean: List[float] = [0.485, 0.456, 0.406],
    std:  List[float] = [0.229, 0.224, 0.225],
    rotation_degrees: int = 15,
    jitter_brightness: float = 0.3,
    jitter_contrast: float = 0.3,
    jitter_saturation: float = 0.2,
    jitter_hue: float = 0.05,
    flip_p: float = 0.5,
    crop_scale: Tuple[float, float] = (0.8, 1.0),
) -> transforms.Compose:
    """
    Pipeline transformasi untuk Training Set.

    Urutan transformasi:
      1. RandomResizedCrop  — variasi skala & posisi objek dalam frame
      2. RandomHorizontalFlip — variasi orientasi kiri/kanan
      3. RandomRotation     — variasi sudut pandang kamera
      4. ColorJitter        — variasi kondisi pencahayaan dan warna
      5. ToTensor           — konversi PIL Image ke FloatTensor [0,1]
      6. Normalize          — standarisasi distribusi pixel (ImageNet stats)

    Args:
        image_size         : Ukuran output akhir (px), default 224
        resize_to          : Ukuran intermediate sebelum crop (px), default 256
        mean               : Mean normalisasi per channel (R, G, B)
        std                : Std normalisasi per channel (R, G, B)
        rotation_degrees   : Batas rotasi acak (±derajat)
        jitter_brightness  : Faktor variasi kecerahan
        jitter_contrast    : Faktor variasi kontras
        jitter_saturation  : Faktor variasi saturasi
        jitter_hue         : Faktor variasi hue (warna)
        flip_p             : Probabilitas horizontal flip
        crop_scale         : Range skala untuk RandomResizedCrop

    Returns:
        transforms.Compose : Pipeline transformasi yang siap digunakan
    """
    return transforms.Compose([
        # Augmentasi 1: RandomResizedCrop
        # Mensimulasikan variasi skala objek dan posisi dalam frame.
        # Objek sampah dapat terlihat dari jarak dekat maupun jauh.
        transforms.RandomResizedCrop(
            size=image_size,
            scale=crop_scale,
        ),

        # Augmentasi 2: RandomHorizontalFlip
        # Sampah tidak memiliki orientasi kiri-kanan yang inherent.
        # Flip horizontal menggandakan keragaman data secara efektif.
        transforms.RandomHorizontalFlip(p=flip_p),

        # Augmentasi 3: RandomRotation
        # Kamera konveyor atau foto manual dapat menghasilkan gambar
        # yang sedikit miring. Rotasi ±15° mensimulasikan kondisi nyata.
        transforms.RandomRotation(degrees=rotation_degrees),

        # Augmentasi 4: ColorJitter
        # Kondisi pencahayaan berbeda-beda (dalam ruangan, luar ruangan,
        # lampu kuning, siang hari). ColorJitter membuat model lebih robust
        # terhadap variasi ini tanpa mengubah identitas objek.
        transforms.ColorJitter(
            brightness=jitter_brightness,
            contrast=jitter_contrast,
            saturation=jitter_saturation,
            hue=jitter_hue,
        ),

        # Konversi ke tensor: PIL Image → FloatTensor [0.0, 1.0]
        transforms.ToTensor(),

        # Normalisasi dengan statistik ImageNet.
        # Meskipun CNN dilatih dari scratch, normalisasi ImageNet digunakan
        # untuk menjaga KONSISTENSI pipeline dengan eksperimen Transfer Learning
        # berikutnya (ResNet50, EfficientNet-B0) yang memang menggunakan statistik ini.
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    image_size: int = 224,
    resize_to: int = 256,
    mean: List[float] = [0.485, 0.456, 0.406],
    std:  List[float] = [0.229, 0.224, 0.225],
) -> transforms.Compose:
    """
    Pipeline transformasi untuk Validation Set dan Test Set.

    Tidak ada augmentasi — hanya transformasi deterministik:
      1. Resize ke 256px (sisi terpendek)
      2. CenterCrop ke 224px
      3. ToTensor
      4. Normalize

    Urutan ini menjamin hasil evaluasi yang konsisten dan tidak bergantung
    pada randomness. Menggunakan pipeline yang sama antara val dan test
    memastikan tidak ada distribusi shift saat inferensi akhir.

    Args:
        image_size : Ukuran output akhir (px), default 224
        resize_to  : Ukuran intermediate sebelum crop (px), default 256
        mean       : Mean normalisasi per channel (R, G, B)
        std        : Std normalisasi per channel (R, G, B)

    Returns:
        transforms.Compose : Pipeline transformasi deterministik
    """
    return transforms.Compose([
        # Resize: sisi terpendek menjadi resize_to px
        # Mempertahankan aspect ratio untuk menghindari distorsi objek
        transforms.Resize(resize_to),

        # CenterCrop: ambil area tengah 224×224
        # Objek utama biasanya terletak di tengah frame
        transforms.CenterCrop(image_size),

        # Konversi ke tensor
        transforms.ToTensor(),

        # Normalisasi — IDENTIK dengan training pipeline
        transforms.Normalize(mean=mean, std=std),
    ])


def get_transforms_from_config(config: dict, split: str = 'train') -> transforms.Compose:
    """
    Factory function: membuat pipeline transformasi dari konfigurasi YAML.

    Args:
        config : Dictionary konfigurasi (dari baseline.yaml)
        split  : 'train', 'val', atau 'test'

    Returns:
        transforms.Compose

    Raises:
        ValueError : Jika split tidak dikenal
    """
    pre_cfg = config.get('preprocessing', {})
    aug_cfg = config.get('augmentation', {})

    image_size = pre_cfg.get('image_size', 224)
    resize_to  = pre_cfg.get('resize_to', 256)
    mean       = pre_cfg.get('normalize', {}).get('mean', [0.485, 0.456, 0.406])
    std        = pre_cfg.get('normalize', {}).get('std',  [0.229, 0.224, 0.225])

    if split == 'train':
        jitter = aug_cfg.get('color_jitter', {})
        return get_train_transforms(
            image_size=image_size,
            resize_to=resize_to,
            mean=mean,
            std=std,
            rotation_degrees=aug_cfg.get('rotation', {}).get('degrees', 15),
            jitter_brightness=jitter.get('brightness', 0.3),
            jitter_contrast=jitter.get('contrast', 0.3),
            jitter_saturation=jitter.get('saturation', 0.2),
            jitter_hue=jitter.get('hue', 0.05),
            flip_p=aug_cfg.get('horizontal_flip', {}).get('p', 0.5),
            crop_scale=aug_cfg.get('random_resized_crop', {}).get('scale', [0.8, 1.0]),
        )
    elif split in ('val', 'test'):
        return get_val_transforms(
            image_size=image_size,
            resize_to=resize_to,
            mean=mean,
            std=std,
        )
    else:
        raise ValueError(f"split harus 'train', 'val', atau 'test'. Diterima: '{split}'")
