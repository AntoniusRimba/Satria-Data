"""
dataset_summary.py
==================
Fungsi-fungsi analisis data eksploratif (EDA) untuk Smart Waste Classification.

Menyediakan utilitas untuk:
  - Scanning folder dataset
  - Membaca properti gambar (resolusi, warna, format)
  - Mendeteksi gambar corrupt
  - Mendeteksi gambar duplikat
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
from PIL import Image, UnidentifiedImageError

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def get_image_paths(class_dir: Path) -> List[Path]:
    """Mengambil seluruh path gambar dalam sebuah folder kelas."""
    return [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]


def compute_md5(filepath: Path) -> str:
    """Menghitung MD5 hash dari sebuah file gambar untuk deteksi duplikat."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)  # Baca 64KB per chunk
        while buf:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def read_image_properties(filepath: Path) -> Optional[Dict]:
    """
    Membaca properti gambar: ukuran, mode warna, format.
    Mengembalikan None jika gambar corrupt/tidak bisa dibaca.
    """
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            return {
                'path'        : str(filepath),
                'filename'    : filepath.name,
                'width'       : width,
                'height'      : height,
                'aspect_ratio': round(width / height, 3) if height > 0 else 0,
                'mode'        : img.mode,
                'format'      : img.format,
                'extension'   : filepath.suffix.lower(),
                'file_size_kb': round(filepath.stat().st_size / 1024, 2),
            }
    except (UnidentifiedImageError, Exception):
        return None


def scan_dataset(train_dir: Path, class_names: Dict[str, str]) -> Tuple[Dict, Dict]:
    """
    Memindai dataset untuk mendapatkan informasi umum dan path gambar per kelas.

    Args:
        train_dir: Path direktori training
        class_names: Mapping dari nama folder ke nama display

    Returns:
        dataset_info: Ringkasan per kelas
        all_image_paths: Path gambar per kelas
    """
    dataset_info = {}
    all_image_paths = {}

    for class_folder in sorted(train_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        folder_name = class_folder.name
        display_name = class_names.get(folder_name, folder_name)
        image_paths = get_image_paths(class_folder)
        all_image_paths[folder_name] = image_paths
        dataset_info[folder_name] = {
            'class_label'  : folder_name,
            'class_name'   : display_name,
            'image_count'  : len(image_paths),
            'folder_path'  : str(class_folder)
        }
    return dataset_info, all_image_paths


def get_image_properties_df(all_image_paths: Dict[str, List[Path]], class_names: Dict[str, str]) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Membaca properti dari semua gambar dalam dataset.
    
    Returns:
        df_props: DataFrame berisi properti setiap gambar
        corrupted_files: Daftar file yang corrupt
    """
    records = []
    corrupted_files = []

    for folder_name, paths in all_image_paths.items():
        display_name = class_names.get(folder_name, folder_name)
        for img_path in paths:
            props = read_image_properties(img_path)
            if props is not None:
                props['class'] = display_name
                props['class_label'] = folder_name
                records.append(props)
            else:
                corrupted_files.append({
                    'path': str(img_path),
                    'class': display_name,
                    'filename': img_path.name
                })
    return pd.DataFrame(records), corrupted_files


def find_duplicates(all_image_paths: Dict[str, List[Path]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Mencari duplikat gambar dalam dataset menggunakan hash MD5.
    
    Returns:
        exact_duplicates: File identik di dalam kelas yang sama
        cross_class_duplicates: File identik lintas kelas
    """
    hash_dict = defaultdict(list)
    for folder_name, paths in all_image_paths.items():
        for img_path in paths:
            file_hash = compute_md5(img_path)
            hash_dict[file_hash].append({
                'path': str(img_path),
                'class_label': folder_name,
                'filename': img_path.name
            })

    exact_duplicates = []
    cross_class_duplicates = []

    for h, files in hash_dict.items():
        if len(files) > 1:
            classes = {f['class_label'] for f in files}
            if len(classes) == 1:
                exact_duplicates.append({'hash': h, 'files': files})
            else:
                cross_class_duplicates.append({'hash': h, 'files': files})

    return exact_duplicates, cross_class_duplicates
