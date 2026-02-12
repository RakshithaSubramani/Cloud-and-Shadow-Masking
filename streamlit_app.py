from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio as rio
import streamlit as st
from PIL import Image

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


def _choose_rgb_bands(src: rio.DatasetReader) -> list[int]:
    if src.count >= 4:
        return [4, 3, 2]
    if src.count >= 3:
        return [1, 2, 3]
    raise ValueError("Input must have at least 3 bands to create an RGB preview.")


@st.cache_data(show_spinner=False)
def _rgb_preview_png(path: str, max_side: int = 512) -> bytes:
    with rio.open(path) as src:
        scale = min(1.0, max_side / max(src.height, src.width))
        out_h = max(1, int(round(src.height * scale)))
        out_w = max(1, int(round(src.width * scale)))
        bands = _choose_rgb_bands(src)
        rgb16 = src.read(bands, out_shape=(3, out_h, out_w), resampling=rio.enums.Resampling.bilinear)
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


def main() -> None:
    st.set_page_config(page_title="Cloud + Shadow Masking", layout="wide")
    st.title("Cloud + Shadow Masking")

    root = Path(__file__).resolve().parent
    runs_dir = root / "runs"
    outputs_dir = root / "outputs" / "streamlit"

    with st.sidebar:
        st.header("Model")
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
            before_col.image(_rgb_preview_png(input_preview_uri), caption="Before (Input RGB)", use_container_width=True)
        except Exception as e:  # noqa: BLE001
            before_col.info(f"Input preview will appear after the first download/caching. Details: {e}")
    elif input_preview_uri is not None:
        try:
            before_col.image(_rgb_preview_png(input_preview_uri), caption="Before (Input RGB)", use_container_width=True)
        except Exception as e:  # noqa: BLE001
            before_col.info(f"Could not render input preview: {e}")
    else:
        before_col.info("Provide an input to see the preview.")

    run_btn = st.button("Run Masking", type="primary", use_container_width=True)

    if run_btn:
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
                )
            else:
                predict_geotiff(
                    ckpt_path=ckpt_path,
                    input_tif=input_path,
                    out_dir=out_dir,
                    device=device,
                    tile_size=tile_size,
                    overlap=overlap,
                )
            status.update(label=f"Done. Outputs in {out_dir}", state="complete", expanded=False)

        col1, col2, col3, col4 = st.columns(4)
        rgb = out_dir / "preview_rgb.png"
        overlay = out_dir / "preview_overlay.png"
        mask = out_dir / "preview_mask.png"
        masked_tif = out_dir / "masked_image.tif"

        if rgb.exists():
            col1.image(str(rgb), caption="Before (Input RGB)", use_container_width=True)
        if overlay.exists():
            col2.image(str(overlay), caption="After (Overlay)", use_container_width=True)
        if mask.exists():
            col3.image(str(mask), caption="Mask preview", use_container_width=True)
        if masked_tif.exists():
            try:
                col4.image(_rgb_preview_png(str(masked_tif)), caption="After (Masked RGB)", use_container_width=True)
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
