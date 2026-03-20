"""Project benchmark entrypoint.

This wrapper standardizes where evaluation results are written while reusing the
existing ``compressai.utils.eval_model`` implementation.
"""

from __future__ import annotations

import argparse

from pathlib import Path

from helpers import BENCHMARKS_DIR, ensure_dir, normalize_passthrough_args, run_module


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run CompressAI image evaluation with project output conventions."
    )
    parser.add_argument(
        "source",
        choices=["pretrained", "checkpoint"],
        help="Whether to evaluate a pretrained model or a checkpoint.",
    )
    parser.add_argument("dataset", type=Path, help="Dataset directory passed to eval_model.")
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
        help="Directory where benchmark JSON outputs should be stored.",
    )
    return parser.parse_known_args()


def has_flag(argv: list[str], short_flag: str, long_flag: str) -> bool:
    return short_flag in argv or long_flag in argv


def main() -> None:
    args, passthrough = parse_args()
    passthrough = normalize_passthrough_args(passthrough)

    run_name = args.run_name or f"{args.architecture}-{args.source}"
    output_dir = ensure_dir(args.output_root / run_name)

    forward_argv = [
        args.source,
        str(args.dataset),
        "-a",
        args.architecture,
        *passthrough,
    ]

    if not has_flag(passthrough, "-d", "--output_directory"):
        forward_argv.extend(["-d", str(output_dir)])

    if not has_flag(passthrough, "-o", "--output-file"):
        forward_argv.extend(["-o", run_name])

    run_module("compressai.utils.eval_model.__main__", forward_argv)


if __name__ == "__main__":
    main()
