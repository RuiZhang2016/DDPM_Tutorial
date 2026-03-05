"""
Neural network architecture for DDPM noise prediction.

Uses a simple MLP with time-step conditioning via learned embeddings.
For 2D data (like Swiss Roll), a lightweight architecture suffices —
no U-Net required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    """
    Linear layer conditioned on a discrete timestep.

    Instead of directly passing the timestep as an input feature,
    each timestep learns a multiplicative scaling vector (gamma) via
    an embedding table.  This allows the same layer weights to behave
    differently at every diffusion step.

    forward(x, t) -> gamma_t * (Wx + b)
    """

    def __init__(self, in_features: int, out_features: int, num_steps: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.embed = nn.Embedding(num_steps, out_features)
        self.embed.weight.data.uniform_()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        gamma = self.embed(t)
        return gamma.view(-1, out.size(-1)) * out


class ConditionalModel(nn.Module):
    """
    Simple MLP that predicts the noise epsilon_theta(x_t, t).

    Architecture: 4 conditional-linear layers (2 -> 128 -> 128 -> 128 -> 2)
    with softplus activations.  The output dimension equals the data
    dimension (2 for Swiss Roll).
    """

    def __init__(self, n_steps: int, data_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.lin1 = ConditionalLinear(data_dim, hidden_dim, n_steps)
        self.lin2 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin3 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin4 = nn.Linear(hidden_dim, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = F.softplus(self.lin1(x, t))
        x = F.softplus(self.lin2(x, t))
        x = F.softplus(self.lin3(x, t))
        return self.lin4(x)
