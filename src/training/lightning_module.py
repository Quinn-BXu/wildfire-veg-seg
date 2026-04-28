"""PyTorch Lightning module — model-agnostic segmentation training."""
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex

from src.training.losses import DiceCELoss


class SegModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        ce_w = torch.tensor(class_weights) if class_weights is not None else None
        self.loss_fn = DiceCELoss(ce_weight=ce_w)

        self.train_iou = MulticlassJaccardIndex(num_classes=num_classes, average="none")
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, average="none")
        self.test_iou = MulticlassJaccardIndex(num_classes=num_classes, average="none")

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)

        if stage == "train":
            metric = self.train_iou
        elif stage == "val":
            metric = self.val_iou
        else:
            metric = self.test_iou
        metric.update(preds, y)

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def test_step(self, batch, _):
        return self._step(batch, "test")

    def on_train_epoch_end(self):
        self._log_iou(self.train_iou, "train")

    def on_validation_epoch_end(self):
        self._log_iou(self.val_iou, "val")

    def on_test_epoch_end(self):
        self._log_iou(self.test_iou, "test")

    def _log_iou(self, metric: MulticlassJaccardIndex, stage: str):
        iou = metric.compute()  # (num_classes,)
        for i, v in enumerate(iou):
            self.log(f"{stage}/iou_class_{i}", v, on_epoch=True)
        self.log(f"{stage}/miou", iou.mean(), prog_bar=True, on_epoch=True)
        metric.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs or 30)
        return {"optimizer": opt, "lr_scheduler": sched}
