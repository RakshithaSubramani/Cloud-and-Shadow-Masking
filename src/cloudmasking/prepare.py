from __future__ import annotations

from pathlib import Path

import numpy as np
import time


def prepare_main(
    cache_dir: Path,
    dataset: str,
    out_dir: Path,
    train_count: int,
    val_count: int,
    bands: tuple[int, ...] = (4, 3, 2, 8),
    start_index: int = 0,
    resume: bool = True,
) -> None:
    import rasterio as rio
    import tacoreader.v1 as tacoreader  # type: ignore

    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    ds = tacoreader.load(f"tacofoundation:{dataset}")

    total_needed = int(train_count) + int(val_count)
    if total_needed <= 0:
        raise ValueError("train_count + val_count must be > 0")

    end_index = min(int(start_index) + total_needed, len(ds))
    actual = max(0, end_index - int(start_index))
    if actual <= 0:
        raise ValueError("start_index is beyond dataset length")

    print(f"Preparing CloudSEN12 ({dataset}) into: {out_dir}")
    print(f"Writing train={train_count} val={val_count} (bands={bands})")
    print("First items may take longer while downloading/caching.")

    t0 = time.time()
    last_log = t0
    total = end_index - int(start_index)
    log_every = 25
    for j, i in enumerate(range(int(start_index), end_index)):
        target_dir = train_dir if j < int(train_count) else val_dir
        out_path = target_dir / f"{j:06d}.npz"
        if resume and out_path.exists():
            continue

        sample = ds.read(int(i))
        img_uri = sample.read(0)
        label_uri = sample.read(1)

        with rio.open(img_uri) as src:
            img = src.read(list(bands)).astype(np.uint16)

        with rio.open(label_uri) as src:
            label = src.read(1).astype(np.uint8)

        np.savez_compressed(out_path, image=img, label=label)

        now = time.time()
        if (j + 1) % log_every == 0 or (now - last_log) > 30:
            last_log = now
            done = j + 1
            rate = done / max(1e-6, now - t0)
            remaining = total - done
            eta_s = remaining / max(1e-6, rate)
            pct = 100.0 * done / max(1, total)
            print(f"prepare {done}/{total} ({pct:.1f}%)  ~{rate:.2f} items/s  ETA {eta_s/60:.1f} min", flush=True)
