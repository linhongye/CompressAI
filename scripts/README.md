# Project Entrypoints

This directory contains project-level wrappers around the existing CompressAI
training, codec, and evaluation utilities.

## Files

- `train_neutronstar.py`: wrapper around `examples/train.py`
- `compress_neutronstar.py`: wrapper around `examples/codec.py encode`
- `decompress_neutronstar.py`: wrapper around `examples/codec.py decode`
- `benchmark.py`: benchmark entrypoint for a flat image directory with saved compressed outputs, reconstructions, and reports
- `tile_images_to_1024_bmp.py`: recursively slice source images into `1024x1024` grayscale BMP tiles
- `helpers/`: shared helper code used only by scripts in this directory

## Design Rules

- Keep these files thin CLI wrappers.
- Put reusable model, loss, dataset, and codec logic under `compressai/`.
- Put script-only orchestration helpers under `scripts/helpers/`.
- Write generated outputs to `artifacts/`, not next to source files.

## Example Commands

Train with project output conventions:

```bash
python scripts/train_neutronstar.py --run-name baseline-rgb -- -d /path/to/dataset --model cheng2020-anchor --epochs 1 --cuda
```

Benchmark a pretrained model from a top-level image directory and save outputs
under a custom report directory:

```bash
python scripts/benchmark.py pretrained --input-dir /path/to/images -a cheng2020-anchor --report-root /path/to/report_anchor -- -q 1
```

Slice `image/train` and `image/test` into grayscale BMP tiles:

```bash
python scripts/tile_images_to_1024_bmp.py
```

Allow overlap with a smaller stride:

```bash
python scripts/tile_images_to_1024_bmp.py --stride 512
```

Benchmark a checkpoint:

```bash
python scripts/benchmark.py checkpoint --input-dir /path/to/images -a cheng2020-anchor --report-root /path/to/report_anchor -- --path /path/to/checkpoint.pth.tar
```

Encode and decode:

```bash
python scripts/compress_neutronstar.py /path/to/image.png -- --model cheng2020-anchor --quality 3 --cuda
python scripts/decompress_neutronstar.py artifacts/bitstreams/image.bin
```

The optional `--` separator is supported and recommended when passing arguments
through to the upstream CompressAI script.

## Output Locations

- Checkpoints: `artifacts/checkpoints/`
- Bitstreams: `artifacts/bitstreams/`
- Reconstructions: `artifacts/recon/`
- Benchmark JSON: `artifacts/benchmarks/`
- Logs: `artifacts/logs/`

When `scripts/benchmark.py` runs, the chosen benchmark output root contains:

- `compressed_image/`: serialized compressed outputs grouped by run
- `decompressed_image/`: decoded PNG reconstructions grouped by run
- `report/`: benchmark JSON files grouped by run

`scripts/benchmark.py` only processes image files directly under the input
directory; nested subdirectories are ignored. It also prints progress messages
throughout the run so you can see model loading, per-image processing, and
report writing in real time.
