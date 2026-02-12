from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio as rio
import torch
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm

from cloudmasking.models.unet import UNet


def _device(device: str) -> torch.device:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _iter_tiles(height: int, width: int, tile: int, overlap: int) -> list[Window]:
    stride = max(1, tile - overlap)
    windows: list[Window] = []
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            h = min(tile, height - top)
            w = min(tile, width - left)
            windows.append(Window(left, top, w, h))
    return windows


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[UNet, dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    bands = tuple(int(b) for b in cfg.get("bands", (4, 3, 2, 8)))
    num_classes = int(cfg.get("num_classes", 3))
    base = int(cfg.get("base", 32))
    model = UNet(in_channels=len(bands), num_classes=num_classes, base=base)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, {"bands": bands, "num_classes": num_classes, "base": base}


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    rgb = np.clip(rgb / 10000.0, 0.0, 1.0)
    out = np.empty_like(rgb, dtype=np.uint8)
    for c in range(3):
        band = rgb[c]
        lo = float(np.percentile(band, 2))
        hi = float(np.percentile(band, 98))
        if hi <= lo + 1e-6:
            lo, hi = 0.0, 1.0
        band = (band - lo) / (hi - lo + 1e-6)
        out[c] = np.clip(band * 255.0, 0, 255).astype(np.uint8)
    return out


def _write_previews(src: rio.DatasetReader, pred: np.ndarray, out_dir: Path) -> None:
    max_side = 1024
    scale = min(1.0, max_side / max(src.height, src.width))
    out_h = max(1, int(round(src.height * scale)))
    out_w = max(1, int(round(src.width * scale)))

    rgb16 = src.read([4, 3, 2], out_shape=(3, out_h, out_w), resampling=rio.enums.Resampling.bilinear)
    rgb8 = _to_uint8_rgb(rgb16).transpose(1, 2, 0)
    Image.fromarray(rgb8).save(out_dir / "preview_rgb.png")

    if (out_h, out_w) != pred.shape:
        import cv2

        pred_small = cv2.resize(pred.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    else:
        pred_small = pred

    overlay = rgb8.copy()
    cloud = pred_small == 1
    shadow = pred_small == 2
    overlay[cloud] = (255, 255, 255)
    overlay[shadow] = (255, 215, 0)
    alpha = 0.45
    blended = (rgb8.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha).astype(np.uint8)
    Image.fromarray(blended).save(out_dir / "preview_overlay.png")

    mask_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mask_rgb[pred_small == 1] = (255, 255, 255)
    mask_rgb[pred_small == 2] = (255, 215, 0)
    Image.fromarray(mask_rgb).save(out_dir / "preview_mask.png")


@torch.no_grad()
def predict_geotiff(
    ckpt_path: Path,
    input_tif: Path | str,
    out_dir: Path,
    device: str = "cuda",
    tile_size: int = 512,
    overlap: int = 64,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if isinstance(input_tif, Path):
        input_tif = input_tif.expanduser().resolve()
        if not input_tif.exists():
            raise FileNotFoundError(
                f"Input GeoTIFF not found: {input_tif}. Pass the real path to your .tif file."
            )

    dev = _device(device)
    model, cfg = _load_model(ckpt_path, dev)
    bands: tuple[int, ...] = cfg["bands"]
    num_classes: int = cfg["num_classes"]

    with rio.open(str(input_tif)) as src:
        height, width = src.height, src.width
        windows = _iter_tiles(height, width, tile=tile_size, overlap=overlap)

        probs = np.zeros((num_classes, height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)

        for w in tqdm(windows, desc="predict tiles"):
            img = src.read(list(bands), window=w).astype(np.float32)
            img = np.clip(img / 10000.0, 0.0, 1.0)
            x = torch.from_numpy(img).unsqueeze(0).to(dev)

            logits = model(x)
            p = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            top = int(w.row_off)
            left = int(w.col_off)
            h = int(w.height)
            ww = int(w.width)
            probs[:, top : top + h, left : left + ww] += p[:, :h, :ww]
            counts[top : top + h, left : left + ww] += 1.0

        probs /= np.maximum(counts[None, :, :], 1e-6)
        pred = probs.argmax(axis=0).astype(np.uint8)

        mask_path = out_dir / "mask_classes.tif"
        profile = src.profile.copy()
        profile.update(count=1, dtype=rio.uint8, nodata=255, compress="deflate")
        with rio.open(mask_path, "w", **profile) as dst:
            dst.write(pred, 1)

        cloud = (pred == 1).astype(np.uint8) * 255
        shadow = (pred == 2).astype(np.uint8) * 255
        bin_profile = src.profile.copy()
        bin_profile.update(count=1, dtype=rio.uint8, nodata=0, compress="deflate")
        with rio.open(out_dir / "mask_cloud.tif", "w", **bin_profile) as dst:
            dst.write(cloud, 1)
        with rio.open(out_dir / "mask_shadow.tif", "w", **bin_profile) as dst:
            dst.write(shadow, 1)

        _write_previews(src, pred, out_dir)

    with rio.open(str(input_tif)) as src:
        masked_profile = src.profile.copy()
        nodata = masked_profile.get("nodata")
        if nodata is None:
            nodata = 0
            masked_profile.update(nodata=nodata)
        masked_profile.update(compress="deflate")

        with rio.open(out_dir / "masked_image.tif", "w", **masked_profile) as dst:
            windows = _iter_tiles(src.height, src.width, tile=tile_size, overlap=0)
            for w in tqdm(windows, desc="write masked"):
                data = src.read(window=w)
                top = int(w.row_off)
                left = int(w.col_off)
                h = int(w.height)
                ww = int(w.width)
                m = pred[top : top + h, left : left + ww]
                bad = (m == 1) | (m == 2)
                for b in range(data.shape[0]):
                    band = data[b]
                    band[bad] = nodata
                    data[b] = band
                dst.write(data, window=w)
