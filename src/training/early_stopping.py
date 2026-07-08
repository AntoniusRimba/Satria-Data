"""
early_stopping.py
=================
EarlyStopping callback untuk menghentikan training saat model berhenti berkembang.

Monitor: val_macro_f1 (higher is better)
Patience: jumlah epoch yang ditunggu sebelum training dihentikan.

Desain:
  - Monitor Macro F1 Score (bukan val_loss) karena ini adalah metrik kompetisi BDC
  - Menyimpan best model weights saat score meningkat
  - Memberikan delta minimum untuk dianggap sebagai peningkatan (min_delta)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early Stopping untuk monitoring Macro F1 Score.

    Menghentikan training jika val_macro_f1 tidak meningkat
    dalam `patience` epoch berturut-turut.

    Args:
        patience  : Jumlah epoch tanpa peningkatan sebelum stop (default 5)
        min_delta : Peningkatan minimum yang dianggap signifikan (default 1e-4)
        verbose   : Tampilkan log setiap epoch (default True)

    Example:
        >>> es = EarlyStopping(patience=5)
        >>> for epoch in range(30):
        ...     val_f1 = validate(model, val_loader)
        ...     es(val_f1, model)
        ...     if es.early_stop:
        ...         break
        >>> model.load_state_dict(es.best_weights)
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        verbose: bool = True,
    ) -> None:
        self.patience       = patience
        self.min_delta      = min_delta
        self.verbose        = verbose

        self.best_score     : Optional[float] = None
        self.best_weights   : Optional[dict]  = None
        self.best_epoch     : int = 0
        self.counter        : int = 0
        self.early_stop     : bool = False

    def __call__(self, score: float, model: nn.Module, epoch: int) -> None:
        """
        Memperbarui state early stopping.

        Args:
            score : Nilai val_macro_f1 epoch saat ini
            model : Model PyTorch untuk menyimpan best weights
            epoch : Nomor epoch saat ini (untuk logging)
        """
        if self.best_score is None:
            # Epoch pertama: selalu simpan sebagai best
            self._update_best(score, model, epoch)

        elif score >= self.best_score + self.min_delta:
            # Peningkatan signifikan ditemukan
            if self.verbose:
                logger.info(
                    f"[EarlyStopping] Val F1 meningkat: "
                    f"{self.best_score:.4f} → {score:.4f} ✓"
                )
            self._update_best(score, model, epoch)

        else:
            # Tidak ada peningkatan
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"[EarlyStopping] Tidak ada peningkatan. "
                    f"Counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"[EarlyStopping] Training dihentikan di epoch {epoch}. "
                    f"Best epoch: {self.best_epoch} (F1={self.best_score:.4f})"
                )

    def _update_best(self, score: float, model: nn.Module, epoch: int) -> None:
        """Simpan best score dan best model weights."""
        self.best_score   = score
        self.best_epoch   = epoch
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.counter      = 0

    def get_best_weights(self) -> Optional[dict]:
        """Mengembalikan best model weights (state_dict)."""
        return self.best_weights

    def get_summary(self) -> dict:
        """Mengembalikan ringkasan early stopping."""
        return {
            'best_epoch'    : self.best_epoch,
            'best_val_f1'   : round(self.best_score, 4) if self.best_score else None,
            'stopped_early' : self.early_stop,
            'patience'      : self.patience,
            'final_counter' : self.counter,
        }
