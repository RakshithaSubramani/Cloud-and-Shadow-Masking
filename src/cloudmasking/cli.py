import argparse
import os
from pathlib import Path


def _path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cloudmask")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a cloud + shadow segmentation model")
    train.add_argument("--data", type=_path, required=True, help="CloudSEN12 root folder")
    train.add_argument("--out", type=_path, required=True, help="Output folder for runs/checkpoints")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--device", type=str, default="cuda")
    train.add_argument("--img-size", type=int, default=256)
    train.add_argument("--base", type=int, default=32, help="UNet base channels (lower is faster)")
    train.add_argument("--prepared", type=_path, help="Use locally prepared dataset directory")
    train.add_argument("--limit-train", type=int, default=0, help="Limit train samples (0 = no limit)")
    train.add_argument("--limit-val", type=int, default=0, help="Limit val samples (0 = no limit)")

    pred = sub.add_parser("predict", help="Run inference on a GeoTIFF and export masks")
    pred.add_argument("--ckpt", type=_path, required=True, help="Path to model checkpoint (.pt)")
    pred.add_argument("--input", type=_path, required=True, help="Input GeoTIFF")
    pred.add_argument("--out", type=_path, required=True, help="Output folder")
    pred.add_argument("--device", type=str, default="cuda")
    pred.add_argument("--tile", type=int, default=512, help="Tile size for inference")
    pred.add_argument("--overlap", type=int, default=64, help="Tile overlap")

    pred_sample = sub.add_parser("predict-sample", help="Run inference on a CloudSEN12 sample (no GeoTIFF needed)")
    pred_sample.add_argument("--ckpt", type=_path, required=True, help="Path to model checkpoint (.pt)")
    pred_sample.add_argument("--data", type=_path, required=True, help="Cache folder for TACO downloads")
    pred_sample.add_argument("--dataset", type=str, default="cloudsen12-l1c", help="cloudsen12-l1c or cloudsen12-l2a")
    pred_sample.add_argument("--index", type=int, default=0, help="Sample index")
    pred_sample.add_argument("--out", type=_path, required=True, help="Output folder")
    pred_sample.add_argument("--device", type=str, default="cuda")
    pred_sample.add_argument("--tile", type=int, default=512, help="Tile size for inference")
    pred_sample.add_argument("--overlap", type=int, default=64, help="Tile overlap")

    prep = sub.add_parser("prepare", help="Prefetch CloudSEN12 samples to local NPZ for faster training")
    prep.add_argument("--data", type=_path, required=True, help="Cache folder for TACO downloads")
    prep.add_argument("--dataset", type=str, default="cloudsen12-l1c", help="cloudsen12-l1c or cloudsen12-l2a")
    prep.add_argument("--out", type=_path, required=True, help="Output directory for prepared NPZ files")
    prep.add_argument("--bands", type=str, default="4,3,2,8", help="Band list, comma separated")
    prep.add_argument("--train-count", type=int, default=2000, help="Number of training samples to prepare")
    prep.add_argument("--val-count", type=int, default=300, help="Number of validation samples to prepare")
    prep.add_argument("--start-index", type=int, default=0, help="Start index in CloudSEN12")

    check = sub.add_parser("check", help="Quick environment + GPU check")
    check.add_argument("--device", type=str, default="cuda")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check":
        from cloudmasking.env import env_check

        env_check(args.device)
        return

    if args.command == "train":
        from cloudmasking.train import train_main

        train_main(
            data_root=args.data,
            out_dir=args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            device=args.device,
            img_size=args.img_size,
            base=args.base,
            prepared_dir=args.prepared,
            limit_train=args.limit_train,
            limit_val=args.limit_val,
        )
        return

    if args.command == "predict":
        from cloudmasking.predict import predict_geotiff

        predict_geotiff(
            ckpt_path=args.ckpt,
            input_tif=args.input,
            out_dir=args.out,
            device=args.device,
            tile_size=args.tile,
            overlap=args.overlap,
        )
        return

    if args.command == "predict-sample":
        from cloudmasking.predict import predict_geotiff

        args.data.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TACO_CACHE_DIR", str(args.data))

        import tacoreader.v1 as tacoreader  # type: ignore

        ds = tacoreader.load(f"tacofoundation:{args.dataset}")
        sample = ds.read(int(args.index))
        img_uri = sample.read(0)

        predict_geotiff(
            ckpt_path=args.ckpt,
            input_tif=str(img_uri),
            out_dir=args.out,
            device=args.device,
            tile_size=args.tile,
            overlap=args.overlap,
        )
        return

    if args.command == "prepare":
        from cloudmasking.prepare import prepare_main

        args.data.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TACO_CACHE_DIR", str(args.data))
        bands = tuple(int(b.strip()) for b in str(args.bands).split(",") if b.strip())
        prepare_main(
            cache_dir=args.data,
            dataset=args.dataset,
            out_dir=args.out,
            train_count=args.train_count,
            val_count=args.val_count,
            bands=bands,
            start_index=args.start_index,
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")
