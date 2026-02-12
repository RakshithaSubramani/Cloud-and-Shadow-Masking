# Cloud + Shadow Masking (CloudSEN12 baseline)

This repo trains a simple semantic-segmentation model to detect **clouds** and **cloud shadows** in Sentinel-2 imagery, then exports masks and a masked GeoTIFF for agriculture analysis.

## What it produces

- `mask_classes.tif`: 0=clear, 1=cloud (thick+thin), 2=shadow
- `mask_cloud.tif`: 255 where cloud, else 0
- `mask_shadow.tif`: 255 where shadow, else 0
- `masked_image.tif`: input image with cloud/shadow pixels set to nodata
- `preview_rgb.png`, `preview_overlay.png`, `preview_mask.png`: easy-to-view previews (use these if Windows Photos can’t open GeoTIFFs)

## Setup (Windows, NVIDIA GPU)

1) Create a project venv (your preferred workflow):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

2) Install PyTorch with CUDA inside THIS venv (this is what makes GPU work per-project):
- https://pytorch.org/get-started/locally/

3) Install the rest (keep your venv activated):

```powershell
pip install -r requirements.txt
pip install -e .
```

4) Quick GPU check (must print CUDA available: True):

```powershell
cloudmask check --device cuda
```

## Training (CloudSEN12 via TACO)

CloudSEN12 is provided as a Cloud-Optimized Dataset (TACO). This project uses `tacoreader` to read the dataset and cache it locally.

- Use `--data` as your **cache folder** (you said you will store it in this project folder, so use `.\data\cloudsen12_cache`).
- CloudSEN12 is currently published as a legacy TACO dataset (`tacofoundation:cloudsen12-*`). This repo uses `tacoreader.v1` internally, so you can keep modern `tacoreader` installed and training will still work.

### Sanity run (small)

```powershell
cloudmask train `
  --data .\data\cloudsen12_cache `
  --out  .\runs `
  --epochs 2 `
  --batch-size 4 `
  --img-size 256 `
  --limit-train 64 `
  --limit-val 16
```

This creates:
- `runs/run_*/checkpoints/best.pt`

### Real training

Increase epochs and remove limits:

```powershell
cloudmask train --data .\data\cloudsen12_cache --out .\runs --epochs 20 --batch-size 8 --img-size 256
```

## Speed tips (important)

If training is very slow, it usually means the dataset is being streamed/cached during training. You have two options:

1) Quick training (streaming, but fewer samples):

```powershell
cloudmask train --data .\data\cloudsen12_cache --out .\runs --epochs 10 --batch-size 8 --img-size 256 --limit-train 2000 --limit-val 300 --num-workers 4
```

2) Fast training (prepare locally once, then train fast):

```powershell
cloudmask prepare --data .\data\cloudsen12_cache --dataset cloudsen12-l1c --train-count 2400 --val-count 300 --out .\data\prepared --bands 4,3,2,8
```

This creates:
- `.\data\prepared\train\*.npz`
- `.\data\prepared\val\*.npz`

Then train from local files:

```powershell
cloudmask train --data .\data\cloudsen12_cache --out .\runs --epochs 10 --batch-size 8 --img-size 256 --prepared .\data\prepared --num-workers 4
```

## Inference on a GeoTIFF

You need a GeoTIFF with Sentinel-2-like band ordering (so that band numbers match training bands). The default checkpoint expects bands `(4,3,2,8)` = (Red, Green, Blue, NIR).

```powershell
cloudmask predict `
  --ckpt .\runs\run_YYYYMMDD_HHMMSS\checkpoints\best.pt `
  --input .\your_real_scene.tif `
  --out .\outputs\scene1 `
  --tile 512 `
  --overlap 64
```

## Inference on a CloudSEN12 sample (no GeoTIFF needed)

This is the easiest way to test your model immediately after training:

```powershell
cloudmask predict-sample `
  --ckpt .\runs\run_YYYYMMDD_HHMMSS\checkpoints\best.pt `
  --data .\data\cloudsen12_cache `
  --dataset cloudsen12-l1c `
  --index 0 `
  --out .\outputs\sample0
```

## Notes

- Labels are remapped from CloudSEN12 classes to 3 classes: clear / cloud / shadow.
- Pixels labeled as 99 (no-data) are ignored during training.

## Next steps (to finish the full project)

1) Train a stronger model:
- Run 15–30 epochs, increase `--batch-size` as your GPU allows.
- Optionally switch to more input bands (e.g., add SWIR) and retrain.

2) Validate visually:
- Pick 10 random cached samples and view RGB + predicted masks.

3) Production inference:
- Collect a few Sentinel-2 GeoTIFF scenes (same band ordering as training).
- Run `cloudmask predict` and verify outputs align in GIS (QGIS).

4) Packaging:
- Add a Streamlit demo (upload GeoTIFF → preview mask overlay → download outputs).

## Streamlit demo

Run the UI:

```powershell
streamlit run .\streamlit_app.py
```
