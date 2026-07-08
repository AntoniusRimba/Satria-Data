"""
trainer.py
==========
Training engine untuk Smart Waste Classification.

Modul ini mengimplementasikan training loop yang modular, reproducible,
dan lengkap dengan monitoring Macro F1 Score sesuai metrik BDC 2026.

Komponen:
  - train_one_epoch() : Satu epoch training dengan gradient update
  - validate_one_epoch() : Satu epoch evaluasi tanpa gradient
  - Trainer class  : Orkestrasi lengkap: train, validate, checkpoint, log

Desain:
  - Semua hyperparameter berasal dari config (tidak hardcode)
  - Early stopping berbasis val_macro_f1
  - Checkpoint otomatis menyimpan best & last model
  - Training history disimpan sebagai CSV
"""

import time
import json
import logging
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.training.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


# ============================================================
# STANDALONE EPOCH FUNCTIONS
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Menjalankan satu epoch training.

    Args:
        model     : Model PyTorch
        loader    : Training DataLoader
        criterion : Loss function
        optimizer : Optimizer
        device    : torch.device (cuda/cpu)
        epoch     : Nomor epoch (untuk logging)

    Returns:
        dict : {'train_loss', 'train_acc', 'train_macro_f1'}
    """
    model.train()
    running_loss = 0.0
    all_preds    = []
    all_labels   = []
    n_batches    = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Akumulasi metrik
        running_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Log setiap 50 batch
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            logger.info(
                f"  [Epoch {epoch}] Batch {batch_idx+1}/{n_batches} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss  = running_loss / n_batches
    avg_acc   = accuracy_score(all_labels, all_preds)
    macro_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'train_loss'     : round(avg_loss, 6),
        'train_acc'      : round(avg_acc, 6),
        'train_macro_f1' : round(macro_f1, 6),
    }


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Menjalankan satu epoch validasi (tanpa gradient update).

    Args:
        model     : Model PyTorch
        loader    : Validation DataLoader
        criterion : Loss function
        device    : torch.device
        epoch     : Nomor epoch

    Returns:
        dict : {'val_loss', 'val_acc', 'val_macro_f1',
                'val_precision', 'val_recall'}
    """
    model.eval()
    running_loss = 0.0
    all_preds    = []
    all_labels   = []
    n_batches    = len(loader)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            running_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss  = running_loss / n_batches
    avg_acc   = accuracy_score(all_labels, all_preds)
    macro_f1  = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall    = recall_score(all_labels, all_preds,    average='macro', zero_division=0)

    return {
        'val_loss'      : round(avg_loss, 6),
        'val_acc'       : round(avg_acc, 6),
        'val_macro_f1'  : round(macro_f1, 6),
        'val_precision' : round(precision, 6),
        'val_recall'    : round(recall, 6),
    }


# ============================================================
# TRAINER CLASS
# ============================================================

class Trainer:
    """
    Training orchestrator untuk Smart Waste Classification.

    Mengintegrasikan:
      - train_one_epoch() + validate_one_epoch()
      - EarlyStopping (berbasis val_macro_f1)
      - ReduceLROnPlateau scheduler
      - Checkpoint management (best & last model)
      - Training history (CSV)
      - Logging lengkap

    Args:
        model        : Model PyTorch
        train_loader : Training DataLoader
        val_loader   : Validation DataLoader
        config       : Dictionary konfigurasi dari baseline.yaml
        device       : torch.device
        output_dirs  : Dict path direktori output

    Example:
        >>> trainer = Trainer(model, train_loader, val_loader, config, device, dirs)
        >>> history = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        project_root: Path,
    ) -> None:
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model        = model.to(self.device)
        self.config       = config
        self.project_root = project_root
        
        # Setup output dirs
        self.output_dirs = {
            'checkpoints': project_root / config['output']['checkpoints_dir'],
            'reports': project_root / config['output']['reports_dir'],
        }
        for d in self.output_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        train_cfg = config['training']
        exp_cfg   = config['experiment']

        # --- Optimizer: AdamW ---
        # AdamW memisahkan weight decay dari gradient update (berbeda dengan Adam).
        # Ini menghasilkan regularisasi yang lebih baik dan menjadi default modern.
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'],
        )

        # --- Loss Function: CrossEntropyLoss ---
        # Standar untuk multi-class classification dengan kelas mutually exclusive.
        # Menggabungkan LogSoftmax + NLLLoss secara numerik stabil.
        self.criterion = nn.CrossEntropyLoss()

        # --- Scheduler: ReduceLROnPlateau ---
        # LR dikurangi saat val_macro_f1 tidak meningkat selama patience epoch.
        # Lebih adaptif dari StepLR karena bereaksi terhadap perilaku aktual training.
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=train_cfg['scheduler_mode'],
            factor=train_cfg['scheduler_factor'],
            patience=train_cfg['scheduler_patience'],
            min_lr=train_cfg['scheduler_min_lr'],
            verbose=True,
        )

        # --- Early Stopping ---
        self.early_stopping = EarlyStopping(
            patience=train_cfg['early_stopping_patience'],
            verbose=True,
        )

        self.epochs     = train_cfg['epochs']
        self.exp_name   = exp_cfg['name']
        self.history    : List[Dict] = []

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> List[Dict]:
        """
        Menjalankan full training loop.

        Returns:
            List[Dict] : Training history per epoch
        """
        logger.info("=" * 60)
        logger.info(f"  TRAINING DIMULAI: {self.exp_name}")
        logger.info(f"  Device   : {self.device}")
        logger.info(f"  Epochs   : {self.epochs}")
        logger.info(f"  Optimizer: AdamW (lr={self.config['training']['learning_rate']})")
        logger.info(f"  Scheduler: ReduceLROnPlateau")
        logger.info(f"  Early Stop: patience={self.config['training']['early_stopping_patience']}")
        logger.info("=" * 60)

        total_start = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # --- Training ---
            train_metrics = train_one_epoch(
                self.model, train_loader, self.criterion, self.optimizer, self.device, epoch
            )

            # --- Validation ---
            val_metrics = validate_one_epoch(
                self.model, val_loader, self.criterion, self.device, epoch
            )

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            # --- Gabungkan metrik ---
            epoch_log = {
                'epoch'         : epoch,
                'lr'            : current_lr,
                'epoch_time_s'  : round(epoch_time, 2),
                **train_metrics,
                **val_metrics,
            }
            self.history.append(epoch_log)

            # --- Log epoch ---
            logger.info(
                f"Epoch [{epoch:02d}/{self.epochs}] "
                f"| train_loss={train_metrics['train_loss']:.4f} "
                f"| train_f1={train_metrics['train_macro_f1']:.4f} "
                f"| val_loss={val_metrics['val_loss']:.4f} "
                f"| val_f1={val_metrics['val_macro_f1']:.4f} "
                f"| lr={current_lr:.6f} "
                f"| {epoch_time:.1f}s"
            )

            # --- Scheduler step ---
            self.scheduler.step(val_metrics['val_macro_f1'])

            # --- Early Stopping check ---
            self.early_stopping(
                score=val_metrics['val_macro_f1'],
                model=self.model,
                epoch=epoch,
            )

            # --- Simpan last checkpoint setiap epoch ---
            self._save_checkpoint(epoch, tag='last')

            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered di epoch {epoch}.")
                break

        total_time = time.time() - total_start

        # --- Restore best weights ---
        if self.early_stopping.best_weights is not None:
            self.model.load_state_dict(self.early_stopping.best_weights)
            logger.info(
                f"Best weights restored dari epoch {self.early_stopping.best_epoch} "
                f"(val_f1={self.early_stopping.best_score:.4f})"
            )

        # --- Simpan best model ---
        self._save_checkpoint(self.early_stopping.best_epoch, tag='best')

        # --- Simpan history ---
        self._save_history()

        # --- Summary ---
        es_summary = self.early_stopping.get_summary()
        summary = {
            'experiment'          : self.exp_name,
            'total_epochs_trained': len(self.history),
            'total_time_min'      : round(total_time / 60, 2),
            'best_epoch'          : es_summary['best_epoch'],
            'best_val_macro_f1'   : es_summary['best_val_f1'],
            'stopped_early'       : es_summary['stopped_early'],
            'final_train_loss'    : self.history[-1]['train_loss'],
            'final_val_loss'      : self.history[-1]['val_loss'],
            'final_val_f1'        : self.history[-1]['val_macro_f1'],
        }
        self._save_summary(summary)

        logger.info("=" * 60)
        logger.info("  TRAINING SELESAI")
        logger.info(f"  Best Epoch    : {summary['best_epoch']}")
        logger.info(f"  Best Val F1   : {summary['best_val_macro_f1']:.4f}")
        logger.info(f"  Total Time    : {summary['total_time_min']:.1f} menit")
        logger.info("=" * 60)

        return self.history

    def _save_checkpoint(self, epoch: int, tag: str = 'last') -> None:
        """Menyimpan model checkpoint ke outputs/checkpoints/."""
        ckpt_dir  = self.output_dirs.get('checkpoints', Path('outputs/checkpoints'))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        filename  = f"{self.exp_name}_{tag}_model.pth"
        save_path = ckpt_dir / filename

        torch.save({
            'epoch'       : epoch,
            'model_state' : self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config'      : self.config,
            'tag'         : tag,
        }, save_path)
        logger.debug(f"Checkpoint saved: {save_path}")

    def _save_history(self) -> None:
        """Menyimpan training history ke CSV di outputs/reports/."""
        reports_dir = self.output_dirs.get('reports', Path('outputs/reports'))
        reports_dir.mkdir(parents=True, exist_ok=True)

        csv_path = reports_dir / f"{self.exp_name}_training_history.csv"
        if self.history:
            fieldnames = list(self.history[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.history)
        logger.info(f"History saved: {csv_path}")

    def _save_summary(self, summary: dict) -> None:
        """Menyimpan experiment summary ke JSON di outputs/reports/."""
        reports_dir = self.output_dirs.get('reports', Path('outputs/reports'))
        reports_dir.mkdir(parents=True, exist_ok=True)

        json_path = reports_dir / f"{self.exp_name}_experiment_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved: {json_path}")
