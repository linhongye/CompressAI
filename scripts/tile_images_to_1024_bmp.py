"""Slice images into 1024x1024 grayscale BMP tiles.

This script scans one or more directories, converts each source image to
8-bit grayscale, writes 1024x1024 BMP tiles next to the source image, and
deletes the original image after tiling succeeds.
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError  # pyright: ignore[reportMissingImports]

Image.MAX_IMAGE_PIXELS = None


SUPPORTED_SUFFIXES = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class ProcessSummary:
    scanned: int = 0
    processed: int = 0
    skipped_small: int = 0
    deleted: int = 0
    failed: int = 0
    tiles_written: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slice images into fixed-size grayscale BMP tiles and delete the "
            "original images after success."
        )
    )
    parser.add_argument(
        "directories",
        nargs="*",
        type=Path,
        default=[Path("image/train"), Path("image/test")],
        help="Directories to scan recursively for source images.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Square tile size in pixels. Defaults to 1024.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1024,
        help=(
            "Sliding window stride in pixels. Use a value smaller than the tile "
            "size to allow overlap."
        ),
    )
    return parser.parse_args()


def iter_source_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and "__x" not in path.stem
        and "__y" not in path.stem
    )


def iter_positions(length: int, tile_size: int, stride: int) -> range:
    return range(0, length - tile_size + 1, stride)


def process_image(
    image_path: Path, tile_size: int, stride: int, summary: ProcessSummary
) -> None:
    summary.scanned += 1

    try:
        image_bytes = image_path.read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            grayscale = image.convert("L").copy()

    except (OSError, UnidentifiedImageError) as exc:
        summary.failed += 1
        print(f"FAIL open image: {image_path} ({exc})")
        return

    try:
        width, height = grayscale.size

        if width < tile_size or height < tile_size:
            summary.skipped_small += 1
            print(
                f"SKIP small image: {image_path} ({width}x{height}) is smaller "
                f"than {tile_size}x{tile_size}."
            )
            return

        tile_paths: list[Path] = []
        for top in iter_positions(height, tile_size, stride):
            for left in iter_positions(width, tile_size, stride):
                tile_path = image_path.with_name(
                    f"{image_path.stem}__x{left:05d}__y{top:05d}.bmp"
                )
                tile = grayscale.crop((left, top, left + tile_size, top + tile_size))
                tile.save(tile_path, format="BMP")
                tile.close()
                tile_paths.append(tile_path)

        if not tile_paths:
            summary.failed += 1
            print(f"FAIL no tiles written: {image_path}")
            return

        image_path.unlink()
    except OSError as exc:
        summary.failed += 1
        print(f"FAIL process image: {image_path} ({exc})")
        return
    finally:
        grayscale.close()

    summary.processed += 1
    summary.deleted += 1
    summary.tiles_written += len(tile_paths)
    print(f"OK {image_path} -> {len(tile_paths)} tiles")


def process_directory(
    directory: Path, tile_size: int, stride: int, summary: ProcessSummary
) -> None:
    if not directory.exists():
        print(f"SKIP missing directory: {directory}")
        return

    if not directory.is_dir():
        print(f"SKIP not a directory: {directory}")
        return

    image_paths = iter_source_images(directory)
    if not image_paths:
        print(f"INFO no source images found in: {directory}")
        return

    for image_path in image_paths:
        process_image(image_path, tile_size, stride, summary)


def main() -> None:
    args = parse_args()

    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")

    summary = ProcessSummary()
    for directory in args.directories:
        process_directory(directory, args.tile_size, args.stride, summary)

    print(
        "SUMMARY "
        f"scanned={summary.scanned} "
        f"processed={summary.processed} "
        f"skipped_small={summary.skipped_small} "
        f"deleted={summary.deleted} "
        f"failed={summary.failed} "
        f"tiles_written={summary.tiles_written}"
    )


if __name__ == "__main__":
    main()
