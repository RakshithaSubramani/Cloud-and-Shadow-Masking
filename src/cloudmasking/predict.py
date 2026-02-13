from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio as rio
import torch
import torch.nn.functional as F
from PIL import Image
from rasterio.enums import ColorInterp
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


def _read_model_input(
    src: rio.DatasetReader, bands: tuple[int, ...], window: Window
) -> np.ndarray:
    desired = [int(b) for b in bands]

    if src.count == 3 and desired == [4, 3, 2, 8]:
        desired = [1, 2, 3, 0]

    h = int(window.height)
    w = int(window.width)
    out = np.zeros((len(desired), h, w), dtype=np.float32)
    for i, b in enumerate(desired):
        if 1 <= b <= src.count:
            out[i] = src.read(b, window=window).astype(np.float32)
        else:
            out[i] = 0.0
    return out


def _pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    h = int(x.shape[-2])
    w = int(x.shape[-1])
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    mode = "reflect"
    if h < 2 or w < 2:
        mode = "replicate"
    x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (pad_h, pad_w)


def _unpad(x: torch.Tensor, pad_hw: tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return x
    h = x.shape[-2] - pad_h
    w = x.shape[-1] - pad_w
    return x[..., :h, :w]


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
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


def _normalize_input_to_unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    vmax = float(np.percentile(x[finite], 99.9))
    if vmax <= 0:
        return np.zeros_like(x, dtype=np.float32)

    if vmax <= 1.5:
        div = 1.0
    elif vmax <= 255.0 * 1.5:
        div = 255.0
    elif vmax <= 10000.0 * 1.5:
        div = 10000.0
    elif vmax <= 65535.0 * 1.1:
        div = 65535.0
    else:
        div = vmax

    return np.clip(x / div, 0.0, 1.0)


def _probs_to_pred(
    probs: np.ndarray,
    cloud_threshold: float | None,
    shadow_threshold: float | None,
    mask_shadow: bool,
) -> np.ndarray:
    if cloud_threshold is None and shadow_threshold is None:
        pred = probs.argmax(axis=0).astype(np.uint8)
        if not mask_shadow:
            pred[pred == 2] = 0
        return pred

    pred = np.zeros(probs.shape[1:], dtype=np.uint8)

    cloud_t = 0.5 if cloud_threshold is None else float(cloud_threshold)
    cloud_mask = probs[1] >= cloud_t
    pred[cloud_mask] = 1

    if mask_shadow and probs.shape[0] >= 3:
        shadow_t = 0.5 if shadow_threshold is None else float(shadow_threshold)
        shadow_mask = (probs[2] >= shadow_t) & (~cloud_mask)
        pred[shadow_mask] = 2

    return pred


def _choose_rgb_bands(src: rio.DatasetReader) -> list[int]:
    try:
        cis = list(src.colorinterp)
        if cis:
            r = cis.index(ColorInterp.red) + 1 if ColorInterp.red in cis else None
            g = cis.index(ColorInterp.green) + 1 if ColorInterp.green in cis else None
            b = cis.index(ColorInterp.blue) + 1 if ColorInterp.blue in cis else None
            if r and g and b:
                return [r, g, b]
    except Exception:
        pass

    if src.count == 4:
        return [1, 2, 3]
    if src.count > 4:
        return [4, 3, 2]
    if src.count >= 3:
        return [1, 2, 3]
    if src.count == 2:
        return [1, 2, 2]
    if src.count == 1:
        return [1, 1, 1]
    raise ValueError("Input must have at least 1 band to create a preview.")


def _rgb_white_cloud_mask(
    src: rio.DatasetReader,
    window: Window,
    brightness_threshold: float,
    whiteness_threshold: float,
) -> np.ndarray:
    bands = _choose_rgb_bands(src)
    rgb = src.read(bands, window=window).astype(np.float32)
    rgb = _normalize_input_to_unit(rgb)
    r, g, b = rgb[0], rgb[1], rgb[2]
    brightness = np.minimum(np.minimum(r, g), b)
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    return (brightness >= float(brightness_threshold)) & (chroma <= float(whiteness_threshold))


def _write_previews(src: rio.DatasetReader, pred: np.ndarray, out_dir: Path) -> None:
    max_side = 1024
    scale = min(1.0, max_side / max(src.height, src.width))
    out_h = max(1, int(round(src.height * scale)))
    out_w = max(1, int(round(src.width * scale)))

    rgb_bands = _choose_rgb_bands(src)
    try:
        rgb16 = src.read(
            rgb_bands, out_shape=(3, out_h, out_w), resampling=rio.enums.Resampling.bilinear
        )
    except Exception:
        safe = [b for b in (1, 2, 3) if b <= src.count]
        if not safe:
            safe = [1]
        while len(safe) < 3:
            safe.append(safe[-1])
        rgb16 = src.read(
            safe, out_shape=(3, out_h, out_w), resampling=rio.enums.Resampling.bilinear
        )
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
    overlay[cloud] = (0, 0, 255)
    overlay[shadow] = (0, 255, 255)
    alpha = 0.45
    blended = (rgb8.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha).astype(np.uint8)
    Image.fromarray(blended).save(out_dir / "preview_overlay.png")

    mask_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mask_rgb[pred_small == 1] = (0, 0, 255)
    mask_rgb[pred_small == 2] = (0, 255, 255)
    Image.fromarray(mask_rgb).save(out_dir / "preview_mask.png")


@torch.no_grad()
def predict_geotiff(
    ckpt_path: Path | None,
    input_tif: Path | str,
    out_dir: Path,
    device: str = "cuda",
    tile_size: int = 512,
    overlap: int = 64,
    mask_shadow: bool = True,
    cloud_threshold: float | None = None,
    shadow_threshold: float | None = None,
    mask_method: str = "model",
    rgb_brightness_threshold: float = 0.8,
    rgb_whiteness_threshold: float = 0.2,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if mask_method not in {"model", "rgb_white"}:
        raise ValueError("mask_method must be one of: 'model', 'rgb_white'")

    ckpt = Path(ckpt_path) if ckpt_path is not None else None
    if mask_method == "model":
        if ckpt is None or not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if isinstance(input_tif, Path):
        input_tif = input_tif.expanduser().resolve()
        if not input_tif.exists():
            raise FileNotFoundError(
                f"Input GeoTIFF not found: {input_tif}. Pass the real path to your .tif file."
            )

    model = None
    bands: tuple[int, ...] = (4, 3, 2, 8)
    num_classes = 3
    dev = _device(device)
    if mask_method == "model":
        if ckpt is None:
            raise RuntimeError("Checkpoint path is missing")
        model, cfg = _load_model(ckpt, dev)
        bands = cfg["bands"]
        num_classes = cfg["num_classes"]

    with rio.open(str(input_tif)) as src:
        height, width = src.height, src.width
        windows = _iter_tiles(height, width, tile=tile_size, overlap=overlap)

        if mask_method == "rgb_white":
            pred = np.zeros((height, width), dtype=np.uint8)
            for w in tqdm(windows, desc="mask rgb tiles"):
                top = int(w.row_off)
                left = int(w.col_off)
                h = int(w.height)
                ww = int(w.width)
                m = _rgb_white_cloud_mask(
                    src=src,
                    window=w,
                    brightness_threshold=rgb_brightness_threshold,
                    whiteness_threshold=rgb_whiteness_threshold,
                )
                pred[top : top + h, left : left + ww][m] = 1
        else:
            probs = np.zeros((num_classes, height, width), dtype=np.float32)
            counts = np.zeros((height, width), dtype=np.float32)

            for w in tqdm(windows, desc="predict tiles"):
                img = _read_model_input(src, bands, w)
                img = _normalize_input_to_unit(img)
                x = torch.from_numpy(img).unsqueeze(0).to(dev)
                x, pad_hw = _pad_to_multiple(x, multiple=16)

                if model is None:
                    raise RuntimeError("Model is not loaded")
                logits = model(x)
                logits = _unpad(logits, pad_hw)
                p = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                top = int(w.row_off)
                left = int(w.col_off)
                h = int(w.height)
                ww = int(w.width)
                probs[:, top : top + h, left : left + ww] += p[:, :h, :ww]
                counts[top : top + h, left : left + ww] += 1.0

            probs /= np.maximum(counts[None, :, :], 1e-6)
            pred = _probs_to_pred(
                probs=probs,
                cloud_threshold=cloud_threshold,
                shadow_threshold=shadow_threshold,
                mask_shadow=mask_shadow,
            )

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

        try:
            _write_previews(src, pred, out_dir)
        except Exception:
            pass

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
                bad = (m == 1) | ((m == 2) if mask_shadow else False)
                for b in range(data.shape[0]):
                    band = data[b]
                    band[bad] = nodata
                    data[b] = band
                dst.write(data, window=w)
