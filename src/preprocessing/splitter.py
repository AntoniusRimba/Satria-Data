"""
splitter.py
===========
Stratified dataset splitting untuk Smart Waste Classification.

Modul ini mengimplementasikan in-memory stratified split:
  - Tidak ada file yang disalin ke folder baru
  - Split disimpan sebagai CSV di outputs/reports/
  - Class distribution dipertahankan di semua partisi

Desain:
  Menggunakan sklearn.model_selection.train_test_split dengan stratify=y
  untuk memastikan proporsi kelas yang representatif di train dan val set.
"""

import logging
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# Label mapping sesuai nama folder dataset BDC
LABEL_MAP: Dict[str, int] = {
    '0_Recyclable': 0,
    '1_Electronic' : 1,
    '2_Organic'    : 2,
}

CLASS_NAMES: Dict[int, str] = {
    0: 'Recyclable',
    1: 'Electronic',
    2: 'Organic',
}

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def collect_dataset(train_dir: Path) -> pd.DataFrame:
    """
    Mengumpulkan seluruh path gambar dan label dari direktori train.

    Struktur folder yang diharapkan:
        train_dir/
        ├── 0_Recyclable/
        ├── 1_Electronic/
        └── 2_Organic/

    Args:
        train_dir : Path ke direktori training dataset

    Returns:
        pd.DataFrame dengan kolom ['filepath', 'label', 'class_name', 'folder']

    Raises:
        FileNotFoundError : Jika train_dir tidak ditemukan
        ValueError        : Jika tidak ada kelas yang terdeteksi
    """
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory tidak ditemukan: {train_dir}")

    records = []
    detected_classes = []

    for class_folder in sorted(train_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        folder_name = class_folder.name
        if folder_name not in LABEL_MAP:
            logger.warning(f"Folder tidak dikenal, dilewati: {folder_name}")
            continue

        label = LABEL_MAP[folder_name]
        class_name = CLASS_NAMES[label]
        detected_classes.append(folder_name)

        for img_path in class_folder.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                records.append({
                    'filepath'  : str(img_path),
                    'label'     : label,
                    'class_name': class_name,
                    'folder'    : folder_name,
                })

    if not records:
        raise ValueError(f"Tidak ditemukan gambar di: {train_dir}")

    logger.info(f"Kelas terdeteksi: {detected_classes}")
    logger.info(f"Total gambar dikumpulkan: {len(records):,}")

    return pd.DataFrame(records)


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.20,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Membagi DataFrame menjadi train dan validation set secara stratified.

    Stratified split memastikan proporsi kelas di train dan val
    mencerminkan distribusi kelas pada dataset keseluruhan.
    Ini KRITIS untuk Macro F1 Score karena setiap kelas harus
    terwakili secara proporsional di validation set.

    Args:
        df           : DataFrame berisi seluruh dataset (output collect_dataset)
        train_ratio  : Proporsi training set (default 0.80)
        val_ratio    : Proporsi validation set (default 0.20)
        random_seed  : Seed untuk reproduktibilitas

    Returns:
        Tuple[df_train, df_val] : Train dan val DataFrame

    Raises:
        ValueError : Jika train_ratio + val_ratio != 1.0
    """
    total = train_ratio + val_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio harus = 1.0. Diterima: {total:.4f}"
        )

    df_train, df_val = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df['label'],
        random_state=random_seed,
        shuffle=True,
    )

    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)

    # Logging distribusi
    logger.info(f"Split selesai (seed={random_seed}):")
    logger.info(f"  Train : {len(df_train):,} gambar ({train_ratio*100:.0f}%)")
    logger.info(f"  Val   : {len(df_val):,} gambar ({val_ratio*100:.0f}%)")

    for split_name, split_df in [('Train', df_train), ('Val', df_val)]:
        dist = split_df['class_name'].value_counts().to_dict()
        logger.info(f"  {split_name} distribution: {dist}")

    return df_train, df_val


def save_split_report(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    output_dir: Path,
    config: Optional[dict] = None,
) -> None:
    """
    Menyimpan dokumentasi split ke disk.

    Files yang dihasilkan:
        preprocessing_split.csv         : Daftar seluruh gambar + split assignment
        preprocessing_split_stats.json  : Statistik distribusi per split

    Args:
        df_train   : Training DataFrame
        df_val     : Validation DataFrame
        output_dir : Direktori tujuan (outputs/reports/)
        config     : Konfigurasi eksperimen (opsional, untuk dokumentasi)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gabungkan dengan kolom 'split'
    df_train_tagged = df_train.copy()
    df_train_tagged['split'] = 'train'
    df_val_tagged   = df_val.copy()
    df_val_tagged['split'] = 'val'
    df_full = pd.concat([df_train_tagged, df_val_tagged], ignore_index=True)

    # Simpan CSV
    csv_path = output_dir / 'preprocessing_split.csv'
    df_full.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    # Statistik distribusi
    stats = {
        'total': int(len(df_full)),
        'train': {
            'count': int(len(df_train)),
            'distribution': df_train['class_name'].value_counts().to_dict()
        },
        'val': {
            'count': int(len(df_val)),
            'distribution': df_val['class_name'].value_counts().to_dict()
        },
    }
    if config:
        stats['config'] = config.get('split', {})

    json_path = output_dir / 'preprocessing_split_stats.json'
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved: {json_path}")


def get_split_summary(df_train: pd.DataFrame, df_val: pd.DataFrame) -> str:
    """
    Menghasilkan teks ringkasan split untuk ditampilkan di notebook.

    Args:
        df_train : Training DataFrame
        df_val   : Validation DataFrame

    Returns:
        str : Ringkasan teks split
    """
    total = len(df_train) + len(df_val)
    lines = [
        "=" * 52,
        "  DATASET SPLIT SUMMARY",
        "=" * 52,
        f"  Total gambar      : {total:,}",
        f"  Training set      : {len(df_train):,} ({len(df_train)/total*100:.1f}%)",
        f"  Validation set    : {len(df_val):,} ({len(df_val)/total*100:.1f}%)",
        "",
        "  Training Distribution:",
    ]
    for cls, cnt in df_train['class_name'].value_counts().items():
        pct = cnt / len(df_train) * 100
        lines.append(f"    {cls:<14}: {cnt:,} ({pct:.1f}%)")

    lines.append("  Validation Distribution:")
    for cls, cnt in df_val['class_name'].value_counts().items():
        pct = cnt / len(df_val) * 100
        lines.append(f"    {cls:<14}: {cnt:,} ({pct:.1f}%)")

    lines.append("=" * 52)
    return "\n".join(lines)
