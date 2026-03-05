# DDPM from Scratch: 2D Swiss Roll Tutorial

A minimal, educational implementation of **Denoising Diffusion Probabilistic Models (DDPM)** on a 2D Swiss Roll distribution — built entirely from scratch with PyTorch.

> 📖 [中文版 README](README_CN.md)

## Overview

This project implements the core DDPM algorithm from [Ho et al., 2020](https://arxiv.org/abs/2006.11239) using a simple 2D toy dataset (Swiss Roll). It is designed as a hands-on tutorial that covers:

1. **Physical Intuition** — What diffusion really does to a data distribution
2. **Mathematical Formulation** — The key equations behind forward/reverse processes
3. **Engineering Implementation** — Clean, modular PyTorch code

### Why Swiss Roll?

The Swiss Roll is a classic 2D manifold with non-trivial spiral structure. It is:
- Simple enough to train on a CPU in minutes
- Complex enough to verify the model has truly learned the distribution
- Easy to visualise at every stage of the diffusion process

## Project Structure

```
ddpm/
├── ddpm/                      # Core library
│   ├── __init__.py
│   ├── noise_schedule.py      # Beta/alpha schedule (sigmoid & linear)
│   ├── model.py               # Conditional MLP noise predictor
│   ├── diffusion.py           # Forward & reverse diffusion processes
│   ├── dataset.py             # Swiss Roll data generation
│   └── visualization.py       # Plotting utilities
├── notebooks/
│   └── ddpm_tutorial.ipynb    # Interactive step-by-step tutorial
├── train.py                   # Training script (CLI)
├── sample.py                  # Sampling / generation script (CLI)
├── requirements.txt           # Python dependencies
├── README.md                  # English documentation
└── README_CN.md               # Chinese documentation
```

## Quick Start

**Requirements:** Python 3.12

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This trains the DDPM on a Swiss Roll dataset (10k points, 100 diffusion steps, 1000 epochs). Outputs are saved to `outputs/`.

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--n_steps` | 100 | Number of diffusion steps |
| `--epochs` | 1000 | Training epochs |
| `--batch_size` | 128 | Mini-batch size |
| `--lr` | 1e-3 | Learning rate |
| `--schedule` | sigmoid | Noise schedule type (`sigmoid` or `linear`) |
| `--device` | cpu | Device (`cpu` or `cuda`) |

### 3. Generate samples

```bash
python sample.py
```

Loads the trained model and generates new samples via the full reverse chain. Produces comparison plots in `outputs/`.

### 4. Interactive notebook

```bash
jupyter notebook notebooks/ddpm_tutorial.ipynb
```

The notebook walks through every concept with inline code, equations, and visualisations.

## Key Equations

### Forward Process (adding noise)

$$q(x_t \mid x_0) = \mathcal{N}\big(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t) I\big)$$

Reparameterised:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

### Reverse Process (removing noise)

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \varepsilon_\theta(x_t, t) \right) + \sigma_t z$$

### Training Loss

$$L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \varepsilon}\big[\| \varepsilon - \varepsilon_\theta(x_t, t) \|^2\big]$$

## Model Architecture

For 2D data, a lightweight MLP with **timestep conditioning** is sufficient:

```
Input (2D) → ConditionalLinear(128) → Softplus
           → ConditionalLinear(128) → Softplus
           → ConditionalLinear(128) → Softplus
           → Linear(2) → Output (predicted noise)
```

Each `ConditionalLinear` layer learns a per-timestep scaling vector via an embedding table, enabling the same weights to behave differently at each diffusion step.

## Results

After training for ~1000 epochs on CPU (a few minutes), the model generates points that closely match the Swiss Roll distribution.

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS 2020.
- Sohl-Dickstein, J., et al. (2015). [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585). ICML 2015.
- Weng, L. (2021). [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) Lil'Log.
- 刘一懂. (2025, April 9). 【手撕Diffusion-01】DDPM 原理精讲：物理直觉 [Web log post]. 小红书. https://www.xiaohongshu.com/explore/67f62bdc000000001d0079cb
## License

MIT
