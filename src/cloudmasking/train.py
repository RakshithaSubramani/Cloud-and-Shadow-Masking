from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloudmasking.data.cloudsen12_taco import CloudSEN12TacoDataset, CloudSEN12TacoOptions
from cloudmasking.metrics import compute_iou_f1, update_confusion_matrix
from cloudmasking.models.unet import UNet
from cloudmasking.utils import save_json, set_seed


def _make_run_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _device(device: str) -> torch.device:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def train_main(
    data_root: Path,
    out_dir: Path,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 3e-4,
    num_workers: int = 4,
    device: str = "cuda",
    img_size: int = 256,
    base: int = 32,
    prepared_dir: Path | None = None,
    limit_train: int = 0,
    limit_val: int = 0,
) -> None:
    set_seed(42)
    data_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TACO_CACHE_DIR", str(data_root))
    run_dir = _make_run_dir(out_dir)
    dev = _device(device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if prepared_dir is not None:
        from cloudmasking.data.prepared_npz import PreparedNPZDataset, PreparedNPZOptions

        train_ds = PreparedNPZDataset(PreparedNPZOptions(root=prepared_dir / "train", img_size=img_size, limit=limit_train))
        val_ds = PreparedNPZDataset(PreparedNPZOptions(root=prepared_dir / "val", img_size=img_size, limit=limit_val))
        train_opts = {"prepared_dir": str(prepared_dir), "img_size": img_size}
        val_opts = {"prepared_dir": str(prepared_dir), "img_size": img_size}
    else:
        train_opts = CloudSEN12TacoOptions(img_size=img_size, split="train", limit=limit_train)
        val_opts = CloudSEN12TacoOptions(img_size=img_size, split="val", limit=limit_val)
        train_ds = CloudSEN12TacoDataset(train_opts)
        val_ds = CloudSEN12TacoDataset(val_opts)

    if num_workers > 0:
        _ = train_ds[0]
        _ = val_ds[0]
        mp_ctx = torch.multiprocessing.get_context("spawn")
    else:
        mp_ctx = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=dev.type == "cuda",
        persistent_workers=num_workers > 0,
        multiprocessing_context=mp_ctx,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=dev.type == "cuda",
        persistent_workers=num_workers > 0,
        multiprocessing_context=mp_ctx,
        drop_last=False,
    )

    in_ch = len(train_opts.bands) if hasattr(train_opts, "bands") else 4
    model = UNet(in_channels=in_ch, num_classes=3, base=int(base)).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    save_json(
        run_dir / "config.json",
        {
            "train": asdict(train_opts) if hasattr(train_opts, "__dataclass_fields__") else train_opts,
            "val": asdict(val_opts) if hasattr(val_opts, "__dataclass_fields__") else val_opts,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "num_workers": num_workers,
            "device": str(dev),
            "base": base,
        },
    )

    best_val_iou = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}"):
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        cm = torch.zeros((3, 3), dtype=torch.int64, device=dev)
        for x, y in tqdm(val_loader, desc=f"val epoch {epoch}/{epochs}"):
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            val_loss += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            cm = update_confusion_matrix(cm, preds, y, num_classes=3, ignore_index=255)

        val_loss /= max(1, len(val_loader))
        scores = compute_iou_f1(cm)
        mean_iou = float(torch.tensor(scores["iou"]).mean().item())

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "iou": scores["iou"],
            "f1": scores["f1"],
            "mean_iou": mean_iou,
        }
        save_json(run_dir / "metrics_last.json", metrics)
        print(metrics)

        last_ckpt = run_dir / "checkpoints" / "last.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": {"bands": getattr(train_opts, "bands", (4, 3, 2, 8)), "num_classes": 3, "img_size": img_size, "base": base},
            },
            last_ckpt,
        )

        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            best_ckpt = run_dir / "checkpoints" / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {"bands": getattr(train_opts, "bands", (4, 3, 2, 8)), "num_classes": 3, "img_size": img_size, "base": base},
                    "metrics": metrics,
                },
                best_ckpt,
            )
