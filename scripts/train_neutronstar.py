"""NeutronStar2026 training entrypoint.

Self-contained training script for the NeutronStar2026 model.
Accepts flat image directories (no train/test subfolder requirement).
Checkpoints are saved to --output-dir (default: artifacts/checkpoints/).

Example
-------
python scripts/train_neutronstar.py \\
    --train-dir image/train \\
    --test-dir  image/test_process \\
    --output-dir image/model \\
    --epochs 10 --batch-size 2 --cuda
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (REPO_ROOT, SCRIPT_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from compressai.losses import RateDistortionLoss
from compressai.models import NeutronStar2026
from compressai.optimizers import net_aux_optimizer

IMG_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".pgm",
    ".tif", ".tiff", ".webp", ".ppm",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FlatImageDataset(Dataset):
    """Load all images from a flat directory (no subfolder split required)."""

    def __init__(self, root: str | Path, transform=None, grayscale: bool = True):
        root = Path(root)
        if not root.is_dir():
            raise RuntimeError(f'Image directory not found: "{root}"')
        self.samples = sorted(
            f for f in root.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.samples:
            raise RuntimeError(f'No images found in "{root}"')
        self.transform = transform
        self.grayscale = grayscale

    def __getitem__(self, index: int):
        mode = "L" if self.grayscale else "RGB"
        img = Image.open(self.samples[index]).convert(mode)
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.samples)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, criterion, loader, optimizer, aux_optimizer, epoch, clip_max_norm):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(loader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            pct = 100.0 * i / len(loader)
            print(
                f"Train epoch {epoch}: [{i * len(d)}/{len(loader.dataset)} ({pct:.0f}%)]"
                f'  loss={out_criterion["loss"].item():.4f}'
                f'  mse={out_criterion["mse_loss"].item():.4f}'
                f'  bpp={out_criterion["bpp_loss"].item():.4f}'
                f"  aux={aux_loss.item():.4f}",
                flush=True,
            )


@torch.no_grad()
def test_epoch(epoch, loader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    meters = {k: AverageMeter() for k in ("loss", "mse_loss", "bpp_loss", "aux_loss")}

    for d in loader:
        d = d.to(device)
        out_net = model(d)
        out_criterion = criterion(out_net, d)

        meters["loss"].update(out_criterion["loss"].item())
        meters["mse_loss"].update(out_criterion["mse_loss"].item())
        meters["bpp_loss"].update(out_criterion["bpp_loss"].item())
        meters["aux_loss"].update(model.aux_loss().item())

    print(
        f"Test  epoch {epoch}:"
        f'  loss={meters["loss"].avg:.4f}'
        f'  mse={meters["mse_loss"].avg:.4f}'
        f'  bpp={meters["bpp_loss"].avg:.4f}'
        f'  aux={meters["aux_loss"].avg:.4f}',
        flush=True,
    )
    return meters["loss"].avg


def save_checkpoint(state: dict, is_best: bool, output_dir: Path, run_name: str = "checkpoint"):
    last_path = output_dir / f"{run_name}_last.pth.tar"
    torch.save(state, last_path)
    if is_best:
        best_path = output_dir / f"{run_name}_best.pth.tar"
        shutil.copyfile(last_path, best_path)
        print(f"  [*] New best checkpoint saved → {best_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train NeutronStar2026.")

    parser.add_argument("--train-dir", type=Path, required=True,
                        help="Flat directory of training images.")
    parser.add_argument("--test-dir", type=Path, required=True,
                        help="Flat directory of test/validation images.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to write checkpoints. "
                             "Defaults to artifacts/checkpoints/neutronstar2026/.")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--lmbda", type=float, default=1e-2,
                        help="Rate-distortion trade-off lambda.")
    parser.add_argument("-n", "--num-workers", type=int, default=2)
    parser.add_argument("--clip-max-norm", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=192,
                        help="Main network channels (default: 192).")
    parser.add_argument("--M", type=int, default=320,
                        help="Latent channels (default: 320).")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Resume from this checkpoint.")
    parser.add_argument("--run-name", type=str, default="checkpoint",
                        help="Base name for saved checkpoint files. "
                             "Saves <run-name>_last.pth.tar and <run-name>_best.pth.tar.")
    parser.add_argument(
        "--grayscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train on single-channel grayscale images (default: True). "
             "Use --no-grayscale for RGB.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Output directory
    if args.output_dir is None:
        from helpers import CHECKPOINTS_DIR
        args.output_dir = CHECKPOINTS_DIR / "neutronstar2026"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints → {args.output_dir}", flush=True)

    # Datasets
    patch = tuple(args.patch_size)
    train_tf = transforms.Compose([
        transforms.RandomCrop(patch),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.CenterCrop(patch),
        transforms.ToTensor(),
    ])

    train_dataset = FlatImageDataset(args.train_dir, transform=train_tf, grayscale=args.grayscale)
    test_dataset = FlatImageDataset(args.test_dir, transform=test_tf, grayscale=args.grayscale)
    mode_label = "grayscale" if args.grayscale else "RGB"
    print(
        f"Train images: {len(train_dataset)}  |  Test images: {len(test_dataset)}  "
        f"[mode: {mode_label}]",
        flush=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # Model
    in_ch = 1 if args.grayscale else 3
    net = NeutronStar2026(N=args.N, M=args.M, in_ch=in_ch).to(device)
    print(f"Model: NeutronStar2026(N={args.N}, M={args.M}, in_ch={in_ch})", flush=True)

    # Optimizers
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optims = net_aux_optimizer(net, conf)
    optimizer, aux_optimizer = optims["net"], optims["aux"]
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint is not None:
        print(f"Resuming from {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=device)
        last_epoch = ckpt["epoch"] + 1
        net.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, last_epoch + args.epochs):
        print(f"\n--- Epoch {epoch}  lr={optimizer.param_groups[0]['lr']:.2e} ---", flush=True)
        train_one_epoch(net, criterion, train_loader, optimizer, aux_optimizer,
                        epoch, args.clip_max_norm)
        loss = test_epoch(epoch, test_loader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items()},
            },
            is_best,
            args.output_dir,
            run_name=args.run_name,
        )

    print(f"\nTraining complete. Best loss: {best_loss:.4f}", flush=True)
    print(f"Checkpoints saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
