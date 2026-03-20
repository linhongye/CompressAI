"""Project compression entrypoint.

This is a thin wrapper around ``examples/codec.py encode`` with project
defaults for bitstream placement.
"""

from __future__ import annotations

import argparse

from pathlib import Path

from helpers import (
    BITSTREAMS_DIR,
    ensure_dir,
    normalize_passthrough_args,
    run_python_file,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Encode an image or sequence using the existing codec example."
    )
    parser.add_argument("input", type=Path, help="Input file to encode.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BITSTREAMS_DIR,
        help="Directory where encoded bitstreams should be written.",
    )
    return parser.parse_known_args()


def has_output_arg(argv: list[str]) -> bool:
    return "-o" in argv or "--output" in argv


def main() -> None:
    args, passthrough = parse_args()
    passthrough = normalize_passthrough_args(passthrough)
    output_root = ensure_dir(args.output_root)
    output_path = output_root / f"{args.input.stem}.bin"

    forward_argv = ["encode", str(args.input), *passthrough]
    if not has_output_arg(passthrough):
        forward_argv.extend(["-o", str(output_path)])

    run_python_file("examples/codec.py", forward_argv)


if __name__ == "__main__":
    main()
