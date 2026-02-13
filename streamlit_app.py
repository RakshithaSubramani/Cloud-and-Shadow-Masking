from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio as rio
import streamlit as st
from PIL import Image
from rasterio.enums import ColorInterp

from cloudmasking.predict import predict_geotiff


def _find_checkpoints(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    ckpts: list[Path] = []
    for run_dir in sorted(runs_dir.glob("run_*"), reverse=True):
        for name in ("best.pt", "last.pt"):
            p = run_dir / "checkpoints" / name
            if p.exists():
                ckpts.append(p)
    return ckpts


def _zip_dir_bytes(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in folder.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(folder)))
    return buf.getvalue()


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
    raise ValueError("Input must have at least 1 band to create an RGB preview.")


@st.cache_data(show_spinner=False)
def _rgb_preview_png(path: str, max_side: int = 512) -> bytes:
    with rio.open(path) as src:
        scale = min(1.0, max_side / max(src.height, src.width))
        out_h = max(1, int(round(src.height * scale)))
        out_w = max(1, int(round(src.width * scale)))
        bands = _choose_rgb_bands(src)
        try:
            rgb16 = src.read(
                bands, out_shape=(3, out_h, out_w), resampling=rio.enums.Resampling.bilinear
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
        buf = io.BytesIO()
        Image.fromarray(rgb8).save(buf, format="PNG")
        return buf.getvalue()


@st.cache_resource(show_spinner=False)
def _taco_dataset(dataset: str, cache_dir: str):
    import os
    import tacoreader.v1 as tacoreader  # type: ignore

    os.environ.setdefault("TACO_CACHE_DIR", cache_dir)
    return tacoreader.load(f"tacofoundation:{dataset}")


def _cloudsen12_sample_uri(dataset: str, index: int, cache_dir: str) -> str:
    ds = _taco_dataset(dataset, cache_dir)
    sample = ds.read(int(index))
    return str(sample.read(0))


def _st_image(container, image, caption: str):
    try:
        return container.image(image, caption=caption, width="stretch")
    except TypeError:
        try:
            return container.image(image, caption=caption, use_container_width=True)
        except TypeError:
            pass
        try:
            return container.image(image, caption=caption, use_column_width=True)
        except TypeError:
            return container.image(image, caption=caption)


def main() -> None:
    st.set_page_config(page_title="Cloud + Shadow Masking", layout="wide")
    st.title("Cloud + Shadow Masking")

    root = Path(__file__).resolve().parent
    runs_dir = root / "runs"
    outputs_dir = root / "outputs" / "streamlit"

    with st.sidebar:
        st.header("Model")
        mask_method_ui = st.selectbox(
            "Masking method",
            options=["AI model", "Simple RGB (white clouds)"],
            index=1,
        )
        mask_method = "model" if mask_method_ui == "AI model" else "rgb_white"

        ckpt_mode = st.radio("Checkpoint", ["Pick latest run", "Enter path", "Upload .pt"], index=0)

        ckpt_path: Path | None = None
        if ckpt_mode == "Pick latest run":
            ckpts = _find_checkpoints(runs_dir)
            if ckpts:
                choice = st.selectbox("Checkpoint", options=[str(p) for p in ckpts], index=0)
                ckpt_path = Path(choice)
            else:
                st.warning("No checkpoints found in ./runs. Train first or upload a checkpoint.")
        elif ckpt_mode == "Enter path":
            p = st.text_input("Checkpoint path", value="")
            if p.strip():
                ckpt_path = Path(p.strip()).expanduser().resolve()
        else:
            up = st.file_uploader("Upload checkpoint (.pt)", type=["pt"])
            if up is not None:
                tmp_dir = outputs_dir / "uploaded"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = tmp_dir / f"ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                ckpt_path.write_bytes(up.getbuffer())

        st.header("Input")
        input_mode = st.radio("Input source", ["Upload GeoTIFF", "CloudSEN12 sample"], index=0)

        device = st.selectbox("Device", options=["cuda", "cpu"], index=0)
        tile_size = st.slider("Tile size", min_value=128, max_value=1024, value=512, step=64)
        overlap = st.slider("Overlap", min_value=0, max_value=256, value=64, step=16)

        cache_dir = root / "data" / "cloudsen12_cache"
        cache_dir_text = st.text_input("CloudSEN12 cache dir", value=str(cache_dir))

        st.header("Masking")
        if mask_method == "model":
            mask_shadow = st.checkbox("Mask shadows too", value=False)
            use_thresholds = st.checkbox("Use confidence threshold", value=True)
            if use_thresholds:
                cloud_threshold = st.slider("Cloud threshold", 0.0, 1.0, 0.6, 0.05)
                shadow_threshold = (
                    st.slider("Shadow threshold", 0.0, 1.0, 0.6, 0.05) if mask_shadow else None
                )
            else:
                cloud_threshold = None
                shadow_threshold = None
            rgb_brightness_threshold = 0.8
            rgb_whiteness_threshold = 0.2
        else:
            mask_shadow = False
            cloud_threshold = None
            shadow_threshold = None
            rgb_brightness_threshold = st.slider("Brightness (cloud)", 0.0, 1.0, 0.8, 0.02)
            rgb_whiteness_threshold = st.slider("Whiteness (cloud)", 0.0, 1.0, 0.2, 0.02)

    input_path: Path | str | None = None
    sample_dataset: str | None = None
    sample_index: int | None = None
    input_preview_uri: str | None = None
    if input_mode == "Upload GeoTIFF":
        up_img = st.file_uploader("Upload a GeoTIFF (.tif/.tiff)", type=["tif", "tiff"])
        if up_img is not None:
            in_dir = outputs_dir / "inputs"
            in_dir.mkdir(parents=True, exist_ok=True)
            input_path = in_dir / f"scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
            input_path.write_bytes(up_img.getbuffer())
            input_preview_uri = str(input_path)
    else:
        st.subheader("CloudSEN12 sample")
        sample_dataset = st.selectbox("Dataset", options=["cloudsen12-l1c", "cloudsen12-l2a"], index=0)
        sample_index = int(st.number_input("Index", min_value=0, value=0, step=1))
        input_path = "cloudsen12_sample"
        input_preview_uri = None

    st.subheader("Before / After")
    before_col, after_col = st.columns(2)
    if input_mode == "CloudSEN12 sample" and sample_dataset is not None and sample_index is not None:
        try:
            cache_dir_resolved = str(Path(cache_dir_text).expanduser().resolve())
            input_preview_uri = _cloudsen12_sample_uri(sample_dataset, sample_index, cache_dir_resolved)
            _st_image(before_col, _rgb_preview_png(input_preview_uri), caption="Before (Input RGB)")
        except Exception as e:  # noqa: BLE001
            before_col.info(f"Input preview will appear after the first download/caching. Details: {e}")
    elif input_preview_uri is not None:
        try:
            _st_image(before_col, _rgb_preview_png(input_preview_uri), caption="Before (Input RGB)")
        except Exception as e:  # noqa: BLE001
            before_col.info(f"Could not render input preview: {e}")
    else:
        before_col.info("Provide an input to see the preview.")

    run_btn = st.button("Run Masking", type="primary", use_container_width=True)

    if run_btn:
        if mask_method == "model":
            if ckpt_path is None or not ckpt_path.exists():
                st.error("Checkpoint not found. Pick a run, enter a valid path, or upload a .pt file.")
                return
        if input_path is None:
            st.error("Input not provided.")
            return

        out_dir = outputs_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)

        with st.status("Running inference...", expanded=True) as status:
            if input_path == "cloudsen12_sample":
                if sample_dataset is None or sample_index is None:
                    raise RuntimeError("Sample dataset/index missing")
                cache_dir_resolved = str(Path(cache_dir_text).expanduser().resolve())
                img_uri = _cloudsen12_sample_uri(sample_dataset, sample_index, cache_dir_resolved)
                predict_geotiff(
                    ckpt_path=ckpt_path,
                    input_tif=str(img_uri),
                    out_dir=out_dir,
                    device=device,
                    tile_size=tile_size,
                    overlap=overlap,
                    mask_shadow=mask_shadow,
                    cloud_threshold=cloud_threshold,
                    shadow_threshold=shadow_threshold,
                    mask_method=mask_method,
                    rgb_brightness_threshold=rgb_brightness_threshold,
                    rgb_whiteness_threshold=rgb_whiteness_threshold,
                )
            else:
                predict_geotiff(
                    ckpt_path=ckpt_path,
                    input_tif=input_path,
                    out_dir=out_dir,
                    device=device,
                    tile_size=tile_size,
                    overlap=overlap,
                    mask_shadow=mask_shadow,
                    cloud_threshold=cloud_threshold,
                    shadow_threshold=shadow_threshold,
                    mask_method=mask_method,
                    rgb_brightness_threshold=rgb_brightness_threshold,
                    rgb_whiteness_threshold=rgb_whiteness_threshold,
                )
            status.update(label=f"Done. Outputs in {out_dir}", state="complete", expanded=False)

        col1, col2, col3, col4 = st.columns(4)
        rgb = out_dir / "preview_rgb.png"
        overlay = out_dir / "preview_overlay.png"
        mask = out_dir / "preview_mask.png"
        masked_tif = out_dir / "masked_image.tif"

        if rgb.exists():
            _st_image(col1, str(rgb), caption="Before (Input RGB)")
        if overlay.exists():
            _st_image(col2, str(overlay), caption="After (Overlay)")
        if mask.exists():
            _st_image(col3, str(mask), caption="Mask preview")
        if masked_tif.exists():
            try:
                _st_image(col4, _rgb_preview_png(str(masked_tif)), caption="After (Masked RGB)")
            except Exception:
                pass

        st.subheader("Download")
        zip_bytes = _zip_dir_bytes(out_dir)
        st.download_button(
            "Download outputs (.zip)",
            data=zip_bytes,
            file_name=f"{out_dir.name}.zip",
            mime="application/zip",
            use_container_width=True,
        )

        st.caption(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
