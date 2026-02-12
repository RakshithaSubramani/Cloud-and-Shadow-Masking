from __future__ import annotations

import torch


@torch.no_grad()
def update_confusion_matrix(
    cm: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255
) -> torch.Tensor:
    mask = targets != ignore_index
    preds = preds[mask].view(-1).to(torch.int64)
    targets = targets[mask].view(-1).to(torch.int64)
    k = (targets >= 0) & (targets < num_classes)
    targets = targets[k]
    preds = preds[k]
    inds = targets * num_classes + preds
    cm += torch.bincount(inds, minlength=num_classes * num_classes).view(num_classes, num_classes)
    return cm


def compute_iou_f1(cm: torch.Tensor) -> dict[str, list[float]]:
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    iou = tp / (tp + fp + fn + 1e-7)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
    return {
        "iou": iou.cpu().tolist(),
        "f1": f1.cpu().tolist(),
    }
