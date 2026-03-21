"""Project benchmark entrypoint.

This script benchmarks image codecs on a flat input directory, writes the
compressed bitstreams and reconstructions to disk, and stores JSON reports in a
dedicated report directory.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (REPO_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import compressai
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from compressai.ops import compute_padding
from compressai.supernova import Supernova
from compressai.utils.eval_model import __main__ as eval_model_main

from helpers import BENCHMARKS_DIR, ensure_dir, normalize_passthrough_args

IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


def log_progress(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_argv:
        separator_index = raw_argv.index("--")
        wrapper_argv = raw_argv[:separator_index]
        passthrough_argv = raw_argv[separator_index + 1 :]
    else:
        wrapper_argv = raw_argv
        passthrough_argv = []

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark CompressAI image models on a flat image directory and save "
            "compressed outputs, reconstructions, and reports."
        )
    )
    parser.add_argument(
        "source",
        choices=["pretrained", "checkpoint"],
        help="Whether to evaluate a pretrained model or a checkpoint.",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=None,
        help="Top-level image directory passed to eval_model.",
    )
    parser.add_argument(
        "--input-dir",
        "--dataset-dir",
        dest="input_dir",
        type=Path,
        default=None,
        help=(
            "Top-level image directory to benchmark. Only files directly under this "
            "directory are processed; subdirectories are ignored."
        ),
    )
    parser.add_argument(
        "-a",
        "--architecture",
        required=True,
        help="Architecture name understood by compressai.utils.eval_model.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional subdirectory name under artifacts/benchmarks/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BENCHMARKS_DIR,
        help="Base directory used when --report-root is not provided.",
    )
    parser.add_argument(
        "--report-root",
        "--report-dir",
        type=Path,
        dest="report_root",
        default=None,
        help=(
            "Exact output directory for this benchmark run. If omitted, "
            "<output-root>/<run-name> is used."
        ),
    )
    parser.add_argument(
        "--grayscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load images as single-channel grayscale and report psnr_gray (default: True). "
             "Use --no-grayscale for RGB mode.",
    )
    return parser.parse_args(wrapper_argv), passthrough_argv


def collect_top_level_images(dataset: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset.iterdir()
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    )


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    dataset = args.input_dir or args.dataset
    if dataset is None:
        raise ValueError(
            "Missing dataset directory. Provide it as the positional dataset argument "
            "or with --input-dir."
        )

    if args.input_dir is not None and args.dataset is not None:
        if args.input_dir.resolve() != args.dataset.resolve():
            raise ValueError(
                "Received two different dataset directories. Use either the positional "
                "dataset argument or --input-dir, or make sure they point to the same path."
            )

    return dataset


def resolve_report_root(args: argparse.Namespace) -> Path:
    if args.report_root is not None:
        return ensure_dir(args.report_root)

    run_name = args.run_name or f"{args.architecture}-{args.source}"
    return ensure_dir(args.output_root / run_name)


def parse_eval_args(
    source: str, dataset: Path, architecture: str, passthrough: list[str]
) -> argparse.Namespace:
    parser = eval_model_main.setup_args()
    return parser.parse_args([source, str(dataset), "-a", architecture, *passthrough])


def make_run_label(args: argparse.Namespace, run: int | str) -> str:
    if args.source == "pretrained":
        return f"{args.architecture}-{args.metric}-q{run}"
    return f"{args.architecture}-{Path(run).stem}"


def load_model_for_run(args: argparse.Namespace, run: int | str):
    if args.source == "pretrained":
        return eval_model_main.load_pretrained(args.architecture, args.metric, int(run))
    return eval_model_main.load_checkpoint(args.architecture, args.no_update, str(run))


@torch.no_grad()
def compress_and_reconstruct(
    model, image_path: Path, device: str, use_half: bool, grayscale: bool = True
) -> dict:
    pil_mode = "L" if grayscale else "RGB"
    x = to_tensor(Image.open(image_path).convert(pil_mode))
    x_batched = x.unsqueeze(0)

    h, w = x_batched.size(2), x_batched.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)

    x_padded = F.pad(x_batched, pad, mode="constant", value=0).to(device)
    if use_half:
        x_padded = x_padded.half()

    enc_start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - enc_start

    dec_start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - dec_start

    x_hat = F.pad(out_dec["x_hat"], unpad).float().cpu()
    num_pixels = x_batched.size(0) * x_batched.size(2) * x_batched.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    mse = F.mse_loss(x_hat, x_batched).item()
    psnr_gray = 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")

    result_metrics: dict = {
        "psnr_gray": psnr_gray,
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

    if not grayscale:
        rgb_metrics = eval_model_main.compute_metrics(x_batched, x_hat, 255)
        result_metrics["psnr-rgb"] = rgb_metrics["psnr-rgb"]
        result_metrics["ms-ssim-rgb"] = rgb_metrics["ms-ssim-rgb"]

    return {
        "strings": out_enc["strings"],
        "shape": out_enc["shape"],
        "x_hat": x_hat,
        "num_pixels": num_pixels,
        "metrics": result_metrics,
    }


def write_compressed_output(output_path: Path, compressed: dict) -> None:
    payload = {
        "strings": compressed["strings"],
        "shape": compressed["shape"],
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f)


def write_reconstruction(output_path: Path, x_hat: torch.Tensor) -> None:
    transforms.ToPILImage()(x_hat.squeeze(0).clamp(0, 1)).save(output_path)


def benchmark_run(
    args: argparse.Namespace,
    image_paths: list[Path],
    compressed_root: Path,
    decompressed_root: Path,
    residual_root: Path,
    report_root: Path,
    grayscale: bool = True,
) -> dict:
    if args.entropy_estimation:
        raise ValueError(
            "This benchmark workflow writes compressed outputs, so --entropy-estimation "
            "is not supported."
        )

    if args.architecture.endswith("-vbr"):
        raise ValueError("This benchmark workflow does not support VBR architectures yet.")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    compressai.set_entropy_coder(args.entropy_coder)

    if args.source == "pretrained":
        runs: list[int | str] = sorted(int(q) for q in args.qualities.split(",") if q)
    else:
        runs = args.checkpoint_paths

    log_progress(
        f"Starting benchmark with {len(runs)} run(s) over {len(image_paths)} image(s) on {device}."
    )

    all_reports = []
    for run_index, run in enumerate(runs, start=1):
        run_label = make_run_label(args, run)
        log_progress(f"[run {run_index}/{len(runs)}] Loading model {run_label}.")
        model = load_model_for_run(args, run).eval().to(device)
        if args.half:
            model = model.half()
        log_progress(f"[run {run_index}/{len(runs)}] Model {run_label} is ready.")

        per_image = []
        totals: dict[str, float] = {
            "psnr_gray": 0.0,
            "bpp": 0.0,
            "encoding_time": 0.0,
            "decoding_time": 0.0,
        }
        if not grayscale:
            totals["psnr-rgb"] = 0.0
            totals["ms-ssim-rgb"] = 0.0
        if grayscale:
            totals["residual_time"] = 0.0
            totals["bpp_residual"] = 0.0
            totals["bpp_total"] = 0.0

        for image_index, image_path in enumerate(image_paths, start=1):
            log_progress(
                f"[run {run_index}/{len(runs)}][image {image_index}/{len(image_paths)}] "
                f"Compressing {image_path.name}."
            )
            compressed = compress_and_reconstruct(model, image_path, device, args.half, grayscale)

            compressed_path = compressed_root / f"{image_path.stem}.bin"
            reconstruction_path = decompressed_root / f"{image_path.stem}.png"

            write_compressed_output(compressed_path, compressed)
            write_reconstruction(reconstruction_path, compressed["x_hat"])

            image_metrics = {
                "source": image_path.name,
                "compressed_file": str(compressed_path),
                "reconstruction_file": str(reconstruction_path),
                **compressed["metrics"],
            }

            if grayscale:
                residual_path = residual_root / f"{image_path.stem}.res"
                res_start = time.time()
                Supernova.makeResidual(image_path, compressed["x_hat"], residual_path)
                res_time = time.time() - res_start

                res_bytes = residual_path.stat().st_size
                n_px = compressed["num_pixels"]
                bpp_res = res_bytes * 8.0 / n_px
                bpp_total = image_metrics["bpp"] + bpp_res

                image_metrics.update({
                    "residual_file": str(residual_path),
                    "residual_size_bytes": res_bytes,
                    "bpp_residual": bpp_res,
                    "bpp_total": bpp_total,
                    "residual_time": res_time,
                })

            per_image.append(image_metrics)
            for key in totals:
                totals[key] += image_metrics[key]

            psnr_label = (
                f"PSNR(gray) {image_metrics['psnr_gray']:.4f}"
                if grayscale
                else f"PSNR(rgb) {image_metrics['psnr-rgb']:.4f}"
            )
            metrics_message = (
                f"[run {run_index}/{len(runs)}][image {image_index}/{len(image_paths)}] "
                f"Finished {image_path.name}: {image_metrics['bpp']:.4f} bpp, "
                f"{psnr_label}, "
                f"enc {image_metrics['encoding_time']:.3f}s, "
                f"dec {image_metrics['decoding_time']:.3f}s"
            )
            if grayscale:
                metrics_message += (
                    f", res {image_metrics['residual_size_bytes'] / 1024:.1f}KB"
                    f" ({image_metrics['bpp_residual']:.4f} bpp_res,"
                    f" total {image_metrics['bpp_total']:.4f} bpp),"
                    f" res_time {image_metrics['residual_time']:.3f}s"
                )
            if args.verbose and not grayscale:
                metrics_message += f" MS-SSIM {image_metrics['ms-ssim-rgb']:.6f}."
            log_progress(metrics_message)

        averages = {key: totals[key] / len(per_image) for key in totals}
        report = {
            "name": run_label,
            "source": args.source,
            "architecture": args.architecture,
            "metric": args.metric,
            "dataset": str(args.dataset),
            "num_images": len(per_image),
            "compressed_dir": str(compressed_root),
            "reconstruction_dir": str(decompressed_root),
            "residual_dir": str(residual_root) if grayscale else None,
            "results": averages,
            "per_image": per_image,
        }

        report_path = report_root / f"{run_label}.json"
        log_progress(f"[run {run_index}/{len(runs)}] Writing report to {report_path}.")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        all_reports.append(report)

        avg_psnr_msg = (
            f"avg psnr_gray {averages['psnr_gray']:.4f}"
            if grayscale
            else f"avg psnr-rgb {averages['psnr-rgb']:.4f}"
        )
        avg_summary = (
            f"avg {averages['bpp']:.4f} bpp, {avg_psnr_msg}"
        )
        if grayscale:
            avg_summary += (
                f", avg bpp_res {averages['bpp_residual']:.4f}"
                f", avg bpp_total {averages['bpp_total']:.4f}"
            )
        log_progress(f"[run {run_index}/{len(runs)}] Completed {run_label}: {avg_summary}.")

    return {
        "name": f"{args.architecture}-{args.metric}",
        "source": args.source,
        "dataset": str(args.dataset),
        "report_dir": str(report_root),
        "runs": all_reports,
    }


def main() -> None:
    wrapper_args, passthrough = parse_args()
    dataset_root = resolve_dataset_root(wrapper_args)
    passthrough = normalize_passthrough_args(passthrough)
    log_progress(
        f"Parsed wrapper arguments. Source={wrapper_args.source}, "
        f"architecture={wrapper_args.architecture}, dataset={dataset_root}."
    )
    args = parse_eval_args(
        wrapper_args.source,
        dataset_root,
        wrapper_args.architecture,
        passthrough,
    )

    args.dataset = dataset_root
    report_root = resolve_report_root(wrapper_args)
    compressed_root = ensure_dir(report_root / "compressed_image")
    decompressed_root = ensure_dir(report_root / "decompressed_image")
    residual_root = ensure_dir(report_root / "residual")
    json_report_root = ensure_dir(report_root / "report")
    log_progress(f"Benchmark output root: {report_root}")
    log_progress(f"Compressed files will be written to: {compressed_root}")
    log_progress(f"Decoded images will be written to: {decompressed_root}")
    log_progress(f"Residual files will be written to: {residual_root}")
    log_progress(f"JSON reports will be written to: {json_report_root}")

    if not args.dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset}")

    image_paths = collect_top_level_images(args.dataset)
    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found directly under dataset directory: {args.dataset}"
        )
    log_progress(
        f"Collected {len(image_paths)} top-level image(s) from {args.dataset}. "
        "Subdirectories are ignored."
    )

    summary = benchmark_run(
        args,
        image_paths,
        compressed_root=compressed_root,
        decompressed_root=decompressed_root,
        residual_root=residual_root,
        report_root=json_report_root,
        grayscale=wrapper_args.grayscale,
    )
    log_progress("Benchmark finished. Printing JSON summary.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI entrypoint
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
