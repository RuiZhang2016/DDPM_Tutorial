"""
Noise schedule for DDPM.

Defines beta (variance) schedules and computes derived quantities
(alpha, alpha_bar) used throughout the diffusion process.
"""

import torch


class NoiseSchedule:
    """
    Manages the noise schedule for a diffusion process.

    The schedule controls how much noise is added at each timestep.
    Supports linear and sigmoid schedule types.

    Attributes:
        n_steps: Total number of diffusion steps.
        betas: Variance schedule at each timestep, shape (n_steps,).
        alphas: 1 - betas, shape (n_steps,).
        alphas_bar: Cumulative product of alphas, shape (n_steps,).
    """

    def __init__(
        self,
        n_steps: int = 100,
        beta_start: float = 1e-5,
        beta_end: float = 1e-2,
        schedule_type: str = "sigmoid",
    ):
        self.n_steps = n_steps
        self.betas = self._create_schedule(n_steps, beta_start, beta_end, schedule_type)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _create_schedule(
        n_steps: int,
        beta_start: float,
        beta_end: float,
        schedule_type: str,
    ) -> torch.Tensor:
        """
        Create the beta schedule.

        Args:
            n_steps: Number of diffusion steps.
            beta_start: Minimum beta value.
            beta_end: Maximum beta value.
            schedule_type: "sigmoid" or "linear".

        Returns:
            Beta values of shape (n_steps,).
        """
        if schedule_type == "sigmoid":
            betas = torch.sigmoid(torch.linspace(-6, 6, n_steps))
            betas = betas * (beta_end - beta_start) + beta_start
        elif schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, n_steps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        return betas

    def to(self, device: torch.device) -> "NoiseSchedule":
        """Move all tensors to the specified device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        return self
