"""
Gaussian diffusion process — forward noising and reverse denoising.

Implements the core DDPM algorithm from:
    Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.

Key formulas
------------
Forward (closed-form):
    x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * eps

Reverse (single step):
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1 - alpha_bar_t) * eps_theta)
              + sigma_t * z

Training loss (simplified):
    L = E_{t, x_0, eps} || eps - eps_theta(x_t, t) ||^2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from .model import ConditionalModel
    from .noise_schedule import NoiseSchedule


class GaussianDiffusion:
    """
    Encapsulates the forward and reverse diffusion processes.

    Args:
        schedule: A NoiseSchedule instance with precomputed betas/alphas.
    """

    def __init__(self, schedule: NoiseSchedule):
        self.schedule = schedule

    # ------------------------------------------------------------------
    # Forward process  q(x_t | x_0)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0) using the reparameterisation trick.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x0: Clean data, shape (B, D).
            t: Timestep indices, shape (B,).
            noise: Optional pre-sampled noise, shape (B, D).

        Returns:
            Noisy samples x_t, shape (B, D).
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar_t = self.schedule.alphas_bar[t].unsqueeze(1)  # (B, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def noise_estimation_loss(
        self,
        model: ConditionalModel,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the simplified DDPM loss:
            L = || eps - eps_theta(x_t, t) ||^2

        A random timestep t is sampled *per element* in the batch.

        Args:
            model: Network that predicts eps_theta(x_t, t).
            x0: Clean data batch, shape (B, D).

        Returns:
            Scalar MSE loss.
        """
        batch_size = x0.shape[0]
        t = torch.randint(0, self.schedule.n_steps, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise)
        predicted_noise = model(x_t, t)

        return (noise - predicted_noise).square().mean()

    # ------------------------------------------------------------------
    # Reverse process  p(x_{t-1} | x_t)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        model: ConditionalModel,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Single reverse step: sample x_{t-1} from p(x_{t-1} | x_t).

        x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
                  + sigma_t * z

        Args:
            model: Trained noise-prediction network.
            x_t: Current noisy samples, shape (B, D).
            t: Current timestep (scalar int).

        Returns:
            Denoised samples x_{t-1}, shape (B, D).
        """
        t_tensor = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)

        alpha_t = self.schedule.alphas[t]
        beta_t = self.schedule.betas[t]
        alpha_bar_t = self.schedule.alphas_bar[t]

        eps_theta = model(x_t, t_tensor)

        eps_coeff = beta_t / torch.sqrt(1 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - eps_coeff * eps_theta)

        if t > 0:
            sigma_t = torch.sqrt(beta_t)
            z = torch.randn_like(x_t)
            return mean + sigma_t * z
        return mean

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: ConditionalModel,
        shape: tuple,
        device: torch.device | str = "cpu",
        return_trajectory: bool = False,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Full reverse sampling chain: x_T -> x_{T-1} -> ... -> x_0.

        Args:
            model: Trained noise-prediction network.
            shape: Shape of samples to generate, e.g. (1000, 2).
            device: Device for computation.
            return_trajectory: If True, return list of all intermediate x_t.

        Returns:
            Generated samples x_0 (or full trajectory list).
        """
        x = torch.randn(shape, device=device)
        trajectory = [x.cpu()] if return_trajectory else None

        for t in reversed(range(self.schedule.n_steps)):
            x = self.p_sample(model, x, t)
            if return_trajectory:
                trajectory.append(x.cpu())

        if return_trajectory:
            return trajectory
        return x
