"""Project decompression entrypoint.

This is a thin wrapper around ``examples/codec.py decode`` with project
defaults for reconstruction placement.
"""

from __future__ import annotations

import argparse

from pathlib import Path

from helpers import (
    RECON_DIR,
    ensure_dir,
    normalize_passthrough_args,
    run_python_file,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Decode a bitstream using the existing codec example."
    )
    parser.add_argument("input", type=Path, help="Bitstream to decode.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RECON_DIR,
        help="Directory where decoded reconstructions should be written.",
    )
    return parser.parse_known_args()


def has_output_arg(argv: list[str]) -> bool:
    return "-o" in argv or "--output" in argv


def main() -> None:
    args, passthrough = parse_args()
    passthrough = normalize_passthrough_args(passthrough)
    output_root = ensure_dir(args.output_root)
    output_path = output_root / f"{args.input.stem}.png"

    forward_argv = ["decode", str(args.input), *passthrough]
    if not has_output_arg(passthrough):
        forward_argv.extend(["-o", str(output_path)])

    run_python_file("examples/codec.py", forward_argv)


if __name__ == "__main__":
    main()
