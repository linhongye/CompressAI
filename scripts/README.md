# Project Entrypoints

This directory contains project-level wrappers around the existing CompressAI
training, codec, and evaluation utilities.

## Files

- `train_neutronstar.py`: wrapper around `examples/train.py`
- `compress_neutronstar.py`: wrapper around `examples/codec.py encode`
- `decompress_neutronstar.py`: wrapper around `examples/codec.py decode`
- `benchmark.py`: wrapper around `compressai.utils.eval_model`
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

Benchmark a pretrained model:

```bash
python scripts/benchmark.py pretrained /path/to/images -a cheng2020-anchor -- -q 1
```

Benchmark a checkpoint:

```bash
python scripts/benchmark.py checkpoint /path/to/images -a cheng2020-anchor -- --path /path/to/checkpoint.pth.tar
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
