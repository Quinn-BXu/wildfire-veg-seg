"""CLI: train a segmentation model from a YAML config."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset

from src.data.dataset import ChipDataset
from src.data.transforms import build_train_transform, build_val_transform
from src.models.unet import build_unet
from src.training.lightning_module import SegModule


def build_model(cfg):
    if cfg.model.name == "unet":
        return build_unet(
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
        )
    raise ValueError(f"unknown model: {cfg.model.name}")


_CHIP_RE = re.compile(r"(?:.+_)?chip_(\d+)_(\d+)$")


def _chip_xy(chip_id: str) -> tuple[int, int]:
    m = _CHIP_RE.match(chip_id)
    if not m:
        raise ValueError(f"unexpected chip id format: {chip_id}")
    y, x = m.groups()
    return int(x), int(y)


def build_loaders(cfg):
    chip_size = cfg.data.chip_size
    base_kwargs = dict(
        chip_dir=cfg.data.chip_dir,
        band_indices=tuple(cfg.data.band_indices),
        task=cfg.data.task,
        prescaled=bool(cfg.data.get("prescaled", False)),
    )
    full = ChipDataset(**base_kwargs, transform=None)

    ids_with_idx = [(i, full.ids[i], _chip_xy(full.ids[i])) for i in range(len(full))]
    ids_with_idx.sort(key=lambda row: (row[2][0], row[2][1]))  # spatial split by x, then y

    n_total = len(ids_with_idx)
    n_val = int(cfg.data.val_fraction * n_total)
    n_test = int(cfg.data.test_fraction * n_total)
    n_train = n_total - n_val - n_test

    train_idx = [i for i, _, _ in ids_with_idx[:n_train]]
    val_idx = [i for i, _, _ in ids_with_idx[n_train : n_train + n_val]]
    test_idx = [i for i, _, _ in ids_with_idx[n_train + n_val :]]

    train_ds = Subset(full, train_idx)
    val_ds = Subset(full, val_idx)
    test_ds = Subset(full, test_idx)

    # attach per-split transforms by wrapping __getitem__ via the underlying dataset
    train_tf = build_train_transform(chip_size)
    val_tf = build_val_transform(chip_size)

    def _wrap(ds, tf):
        inner = ChipDataset(**base_kwargs, transform=tf)
        ds.dataset = inner
        return ds

    train_ds = _wrap(train_ds, train_tf)
    val_ds = _wrap(val_ds, val_tf)
    test_ds = _wrap(test_ds, val_tf)

    loader_kwargs = dict(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    pl.seed_everything(cfg.data.seed, workers=True)

    model = build_model(cfg)
    module = SegModule(
        model=model,
        num_classes=cfg.model.num_classes,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        class_weights=tuple(cfg.training.class_weights) if cfg.training.class_weights else None,
    )

    train_loader, val_loader, test_loader = build_loaders(cfg)

    save_dir = Path(cfg.logging.save_dir)
    ckpt_dir = Path("checkpoints") / cfg.experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{val/miou:.3f}",
            monitor="val/miou",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor="val/miou", mode="max", patience=8),
    ]
    logger = TensorBoardLogger(save_dir=str(save_dir), name=cfg.experiment_name)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        accelerator=cfg.training.accelerator,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.logging.log_every_n_steps,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(module, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
