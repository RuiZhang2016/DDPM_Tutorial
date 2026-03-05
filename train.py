"""
Training script for the DDPM Swiss Roll tutorial.

Usage:
    python train.py                     # train with defaults
    python train.py --epochs 2000       # more epochs
    python train.py --device cuda       # use GPU
    python train.py --schedule linear   # linear beta schedule
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from ddpm import (
    NoiseSchedule,
    ConditionalModel,
    GaussianDiffusion,
    SwissRollDataset,
    plot_training_loss,
    plot_swiss_roll,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DDPM on 2D Swiss Roll")
    p.add_argument("--n_steps", type=int, default=100, help="Number of diffusion steps")
    p.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    p.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--n_samples", type=int, default=10000, help="Dataset size")
    p.add_argument("--beta_start", type=float, default=1e-5, help="Min beta")
    p.add_argument("--beta_end", type=float, default=1e-2, help="Max beta")
    p.add_argument("--schedule", type=str, default="sigmoid", choices=["sigmoid", "linear"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save_dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- data ----
    dataset = SwissRollDataset(n_samples=args.n_samples, seed=args.seed)
    data = dataset.get_tensor().to(device)

    print(f"Dataset: {data.shape[0]} points, dim={data.shape[1]}")
    plot_swiss_roll(dataset.get_numpy(), save_path=save_dir / "swiss_roll.png")

    # ---- schedule & diffusion ----
    schedule = NoiseSchedule(
        n_steps=args.n_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule_type=args.schedule,
    ).to(device)
    diffusion = GaussianDiffusion(schedule)

    # ---- model & optimiser ----
    model = ConditionalModel(n_steps=args.n_steps).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- training loop ----
    epoch_losses = []
    pbar = tqdm(range(args.epochs), desc="Training")

    for epoch in pbar:
        permutation = torch.randperm(data.shape[0], device=device)
        batch_losses = []

        for i in range(0, data.shape[0], args.batch_size):
            batch = data[permutation[i : i + args.batch_size]]
            loss = diffusion.noise_estimation_loss(model, batch)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        pbar.set_postfix(loss=f"{avg_loss:.6f}")

    # ---- save artefacts ----
    torch.save(model.state_dict(), save_dir / "model.pt")
    torch.save(
        {
            "n_steps": args.n_steps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "schedule": args.schedule,
        },
        save_dir / "config.pt",
    )
    plot_training_loss(epoch_losses, save_path=save_dir / "loss.png")
    print(f"Model & loss curve saved to {save_dir}/")


if __name__ == "__main__":
    train(parse_args())
