"""
Sampling script — generate new points from a trained DDPM model.

Usage:
    python sample.py                            # defaults
    python sample.py --n_samples 5000           # more points
    python sample.py --checkpoint outputs/model.pt
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from ddpm import (
    NoiseSchedule,
    ConditionalModel,
    GaussianDiffusion,
    SwissRollDataset,
    plot_reverse_process,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from trained DDPM")
    p.add_argument("--checkpoint", type=str, default="outputs/model.pt")
    p.add_argument("--config", type=str, default="outputs/config.pt")
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save_dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def sample(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    save_dir = Path(args.save_dir)
    device = torch.device(args.device)

    # ---- load config & model ----
    cfg = torch.load(args.config, map_location=device, weights_only=True)
    schedule = NoiseSchedule(
        n_steps=cfg["n_steps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        schedule_type=cfg["schedule"],
    ).to(device)

    model = ConditionalModel(n_steps=cfg["n_steps"]).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    diffusion = GaussianDiffusion(schedule)

    # ---- sample with full trajectory ----
    print(f"Sampling {args.n_samples} points ({cfg['n_steps']} reverse steps) ...")
    trajectory = diffusion.p_sample_loop(
        model,
        shape=(args.n_samples, 2),
        device=device,
        return_trajectory=True,
    )
    generated = trajectory[-1]
    print("Done.")

    # ---- visualise reverse process ----
    plot_reverse_process(trajectory, save_path=save_dir / "reverse_process.png")

    # ---- compare with real data ----
    real = SwissRollDataset(n_samples=args.n_samples).get_numpy()
    gen_np = generated.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pts, label in [
        (axes[0], real, "Real Swiss Roll"),
        (axes[1], gen_np, "Generated (DDPM)"),
    ]:
        ax.scatter(pts[:, 0], pts[:, 1], alpha=0.4, s=8,
                   edgecolors="white", linewidths=0.3)
        ax.set_title(label, fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Real vs. Generated Distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figures saved to {save_dir}/")


if __name__ == "__main__":
    sample(parse_args())
