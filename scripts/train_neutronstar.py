"""Project training entrypoint.

This is a thin wrapper around ``examples/train.py`` so project-specific runs can
store checkpoints under ``artifacts/checkpoints/`` from day one.
"""

from __future__ import annotations

import argparse

from pathlib import Path

from helpers import (
    CHECKPOINTS_DIR,
    ensure_dir,
    normalize_passthrough_args,
    run_python_file,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run the image training example with project output conventions."
    )
    parser.add_argument(
        "--run-name",
        default="manual-train",
        help="Subdirectory name created under artifacts/checkpoints/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CHECKPOINTS_DIR,
        help="Directory where checkpoint artifacts should be stored.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    passthrough = normalize_passthrough_args(passthrough)
    run_dir = ensure_dir(args.output_root / args.run_name)
    run_python_file("examples/train.py", passthrough, cwd=run_dir)


if __name__ == "__main__":
    main()
