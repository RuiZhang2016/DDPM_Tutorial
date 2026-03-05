"""
Swiss Roll dataset utilities for the DDPM tutorial.

The Swiss Roll is a classic 2D manifold that serves as an excellent toy
distribution for testing generative models.  Its spiral structure is
non-trivial enough to verify that the diffusion model has genuinely
learned the data distribution rather than merely memorising noise.
"""

import torch
import numpy as np
from sklearn.datasets import make_swiss_roll


class SwissRollDataset:
    """
    Generates and wraps a 2D Swiss Roll point cloud.

    The 3D Swiss Roll from sklearn is projected onto the (x, z) plane
    and rescaled to roughly unit variance.

    Args:
        n_samples: Number of data points.
        noise: Standard deviation of Gaussian noise added to the roll.
        scale: Divisor applied after projection (controls spread).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        noise: float = 1.0,
        scale: float = 10.0,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.noise = noise
        self.scale = scale

        points_3d, _ = make_swiss_roll(n_samples, noise=noise, random_state=seed)
        # Project to 2D (x, z) and rescale
        self.data = torch.tensor(
            points_3d[:, [0, 2]] / scale,
            dtype=torch.float32,
        )

    def get_tensor(self) -> torch.Tensor:
        """Return the full dataset as a (n_samples, 2) tensor."""
        return self.data

    def sample_batch(self, batch_size: int) -> torch.Tensor:
        """Sample a random mini-batch from the dataset."""
        indices = torch.randint(0, self.n_samples, (batch_size,))
        return self.data[indices]

    def get_numpy(self) -> np.ndarray:
        """Return data as a NumPy array for plotting."""
        return self.data.numpy()
