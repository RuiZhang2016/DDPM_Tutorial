"""
Visualization helpers for the DDPM Swiss Roll tutorial.

Provides functions to plot:
  - The original Swiss Roll distribution
  - Forward diffusion snapshots at selected timesteps
  - Reverse denoising trajectory
  - Training loss curve
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _scatter(
    ax: plt.Axes,
    points: np.ndarray,
    title: str | None = None,
    alpha: float = 0.4,
    s: float = 8,
    color: str = "#4C72B0",
) -> None:
    """Internal helper: draw a styled scatter plot on *ax*."""
    ax.scatter(points[:, 0], points[:, 1], alpha=alpha, s=s, edgecolors="white",
               linewidths=0.3, color=color)
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def plot_swiss_roll(
    data: np.ndarray | torch.Tensor,
    save_path: str | Path | None = None,
) -> None:
    """Plot the original 2D Swiss Roll distribution."""
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter(ax, data, title="Swiss Roll Distribution")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_forward_process(
    snapshots: List[torch.Tensor],
    timesteps: Sequence[int],
    save_path: str | Path | None = None,
) -> None:
    """
    Visualise the forward diffusion at selected timesteps.

    Args:
        snapshots: List of (N, 2) tensors for each plotted timestep.
        timesteps: Corresponding timestep labels.
    """
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, snap, t in zip(axes, snapshots, timesteps):
        pts = snap.numpy() if isinstance(snap, torch.Tensor) else snap
        _scatter(ax, pts, title=f"t = {t}")

    fig.suptitle("Forward Diffusion Process  q(x_t | x_0)", fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_reverse_process(
    trajectory: List[torch.Tensor],
    steps_to_show: Sequence[int] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Visualise the reverse denoising trajectory.

    Args:
        trajectory: Full list from p_sample_loop (x_T, ..., x_0).
        steps_to_show: Indices into *trajectory* to display.
                        If None, ~6 evenly spaced snapshots are picked.
    """
    total = len(trajectory)
    if steps_to_show is None:
        steps_to_show = np.linspace(0, total - 1, min(6, total), dtype=int)

    n = len(steps_to_show)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, steps_to_show):
        pts = trajectory[idx]
        if isinstance(pts, torch.Tensor):
            pts = pts.numpy()
        remaining = total - 1 - idx
        _scatter(ax, pts, title=f"reverse step {remaining}")

    fig.suptitle("Reverse Denoising Process  p(x_{{t-1}} | x_t)", fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_loss(
    losses: List[float],
    save_path: str | Path | None = None,
) -> None:
    """Plot the training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=0.8, color="#4C72B0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
