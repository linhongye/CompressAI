"""Shared helpers for project entrypoint scripts."""

from __future__ import annotations

import os
import runpy
import sys

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
BITSTREAMS_DIR = ARTIFACTS_DIR / "bitstreams"
RECON_DIR = ARTIFACTS_DIR / "recon"
BENCHMARKS_DIR = ARTIFACTS_DIR / "benchmarks"
LOGS_DIR = ARTIFACTS_DIR / "logs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def run_python_file(relative_path: str, argv: Sequence[str], cwd: Path | None = None) -> None:
    script_path = REPO_ROOT / relative_path
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")

    previous_argv = sys.argv[:]
    sys.argv = [str(script_path), *argv]
    try:
        if cwd is None:
            runpy.run_path(str(script_path), run_name="__main__")
        else:
            with pushd(cwd):
                runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = previous_argv


def run_module(module_name: str, argv: Sequence[str], cwd: Path | None = None) -> None:
    previous_argv = sys.argv[:]
    sys.argv = [module_name, *argv]
    try:
        if cwd is None:
            runpy.run_module(module_name, run_name="__main__")
        else:
            with pushd(cwd):
                runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = previous_argv


def normalize_passthrough_args(argv: Sequence[str]) -> list[str]:
    args = list(argv)
    if args and args[0] == "--":
        return args[1:]
    return args
