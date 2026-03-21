"""Supernova — Phase 5 residual lossless recovery utility.

Provides two static methods that together form a lossless recovery layer on
top of any lossy main bitstream:

- :meth:`Supernova.makeResidual`: original BMP + model reconstruction → residual file
- :meth:`Supernova.restore`:      residual file + model reconstruction → original BMP

Residual file format
--------------------
A binary ``.res`` file with a compact header followed by a zstd-compressed
int16 payload:

    [4 bytes] uint32 H  (big-endian)
    [4 bytes] uint32 W  (big-endian)
    [N bytes] zstd-compressed int16 array of shape (H, W)

zstd (Zstandard) is used because its internal FSE/ANS entropy coder
approaches arithmetic-coding compression ratios at Huffman-level speed,
making it well-suited for Laplacian-distributed residual data (small values
clustered around 0).  Compression level 3 (default) gives ~300-500 MB/s
encode and ~1 GB/s decode on typical hardware.

Numerical convention
--------------------
- Original image: 8-bit grayscale BMP, pixels ∈ [0, 255], read as uint8.
- Model output ``x_hat``: Tensor of shape ``(1, 1, H, W)``, values in [0, 1].
- Quantisation: ``recon_uint8 = round(x_hat * 255).clamp(0, 255).to(uint8)``.
- Residual:     ``residual = original_uint8.astype(int16) − recon_uint8.astype(int16)``.
- Recovery:     ``restored = (recon_uint8.astype(int16) + residual).astype(uint8)``.

Correctness guarantee
---------------------
For any ``x_hat`` derived from the same model run, ``restore`` after
``makeResidual`` must produce a byte-identical copy of the original BMP.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd
from PIL import Image

__all__ = ["Supernova"]

# zstd compression level: 3 is the library default (speed/ratio sweet-spot).
_ZSTD_LEVEL = 3
_HEADER_FMT = ">II"   # two big-endian uint32: H, W
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _quantise(x_hat: torch.Tensor) -> np.ndarray:
    """Convert a model output tensor to a uint8 numpy array of shape (H, W).

    Args:
        x_hat: Tensor of shape ``(1, 1, H, W)`` with values in ``[0, 1]``.

    Returns:
        uint8 numpy array of shape ``(H, W)``.
    """
    if x_hat.ndim != 4 or x_hat.shape[0] != 1 or x_hat.shape[1] != 1:
        raise ValueError(
            f"x_hat must have shape (1, 1, H, W), got {tuple(x_hat.shape)}"
        )
    return (
        x_hat[0, 0]
        .detach()
        .float()
        .mul(255)
        .round()
        .clamp(0, 255)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


def _save_residual(path: Path, residual: np.ndarray) -> None:
    """Serialise a (H, W) int16 residual array to a zstd-compressed .res file."""
    H, W = residual.shape
    header = struct.pack(_HEADER_FMT, H, W)
    payload = zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress(
        residual.astype(np.int16).tobytes()
    )
    path.write_bytes(header + payload)


def _load_residual(path: Path) -> np.ndarray:
    """Deserialise a zstd-compressed .res file back to a (H, W) int16 array."""
    raw = path.read_bytes()
    H, W = struct.unpack(_HEADER_FMT, raw[:_HEADER_SIZE])
    payload = zstd.ZstdDecompressor().decompress(raw[_HEADER_SIZE:])
    return np.frombuffer(payload, dtype=np.int16).reshape(H, W)


class Supernova:
    """Lossless residual codec for grayscale BMP images.

    All methods are static; no instantiation is required.
    """

    @staticmethod
    def makeResidual(
        original_bmp: str | Path,
        x_hat: torch.Tensor,
        residual_path: str | Path,
    ) -> Path:
        """Compute and save the residual between the original image and the model reconstruction.

        The residual is defined as ``original_uint8 − recon_uint8`` and stored
        as a zstd-compressed int16 binary file (``.res``).

        Args:
            original_bmp:  Path to the original 8-bit grayscale BMP file.
            x_hat:         Model reconstruction tensor, shape ``(1, 1, H, W)``,
                           values in ``[0, 1]``.
            residual_path: Destination path for the ``.res`` residual file.
                           Parent directory is created if it does not exist.

        Returns:
            ``Path`` to the saved residual file.

        Raises:
            ValueError: If image and reconstruction dimensions do not match.
        """
        original_bmp = Path(original_bmp)
        residual_path = Path(residual_path)

        original = np.array(Image.open(original_bmp).convert("L"), dtype=np.uint8)
        recon = _quantise(x_hat)

        if original.shape != recon.shape:
            raise ValueError(
                f"Shape mismatch: original {original.shape} vs recon {recon.shape}"
            )

        residual = original.astype(np.int16) - recon.astype(np.int16)

        residual_path.parent.mkdir(parents=True, exist_ok=True)
        _save_residual(residual_path, residual)

        return residual_path

    @staticmethod
    def restore(
        residual_path: str | Path,
        x_hat: torch.Tensor,
        output_bmp: str | Path,
    ) -> Path:
        """Restore the original image by adding the residual to the model reconstruction.

        Args:
            residual_path: Path to the ``.res`` residual file produced by
                           :meth:`makeResidual`.
            x_hat:         Model reconstruction tensor, shape ``(1, 1, H, W)``,
                           values in ``[0, 1]``.  Must be produced by the same
                           model run that generated the residual.
            output_bmp:    Destination path for the restored 8-bit grayscale BMP.
                           Parent directory is created if it does not exist.

        Returns:
            ``Path`` to the saved BMP file.

        Raises:
            ValueError: If residual and reconstruction dimensions do not match.
        """
        residual_path = Path(residual_path)
        output_bmp = Path(output_bmp)

        residual = _load_residual(residual_path)
        recon = _quantise(x_hat)

        if residual.shape != recon.shape:
            raise ValueError(
                f"Shape mismatch: residual {residual.shape} vs recon {recon.shape}"
            )

        restored = (recon.astype(np.int16) + residual).astype(np.uint8)

        output_bmp.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(restored, mode="L").save(str(output_bmp), format="BMP")

        return output_bmp
