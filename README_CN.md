# DDPM 从零实现：二维瑞士卷教程

基于 PyTorch 从零实现 **去噪扩散概率模型（DDPM）**，以二维瑞士卷分布为例 — 一份最小化、教学向的完整教程。

> 📖 [English README](README.md)

## 概述

本项目实现了 [Ho et al., 2020](https://arxiv.org/abs/2006.11239) 提出的 DDPM 核心算法，使用简单的二维瑞士卷（Swiss Roll）玩具数据集。教程从三个层面展开：

1. **物理直觉** — 扩散过程对数据分布做了什么
2. **数学推导** — 正向/逆向过程的关键公式
3. **工程实现** — 清晰、模块化的 PyTorch 代码

### 为什么选择瑞士卷？

瑞士卷是经典的二维流形，具有非平凡的螺旋结构：
- 足够简单，CPU 上几分钟即可训练完成
- 足够复杂，能验证模型是否真正学到了数据分布
- 便于在扩散过程的每一步进行可视化

## 项目结构

```
ddpm/
├── ddpm/                      # 核心库
│   ├── __init__.py
│   ├── noise_schedule.py      # Beta/Alpha 噪声调度（sigmoid 和 linear）
│   ├── model.py               # 条件 MLP 噪声预测器
│   ├── diffusion.py           # 正向与逆向扩散过程
│   ├── dataset.py             # 瑞士卷数据生成
│   └── visualization.py       # 可视化工具
├── notebooks/
│   └── ddpm_tutorial.ipynb    # 交互式分步教程
├── train.py                   # 训练脚本（命令行）
├── sample.py                  # 采样/生成脚本（命令行）
├── requirements.txt           # Python 依赖
├── README.md                  # 英文文档
└── README_CN.md               # 中文文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py
```

默认在瑞士卷数据集（1 万个点、100 步扩散、1000 个 epoch）上训练。输出保存至 `outputs/` 目录。

**可选参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--n_steps` | 100 | 扩散步数 |
| `--epochs` | 1000 | 训练轮数 |
| `--batch_size` | 128 | 小批量大小 |
| `--lr` | 1e-3 | 学习率 |
| `--schedule` | sigmoid | 噪声调度类型（`sigmoid` 或 `linear`） |
| `--device` | cpu | 设备（`cpu` 或 `cuda`） |

### 3. 生成样本

```bash
python sample.py
```

加载已训练的模型，通过完整的逆向链生成新样本。对比图保存至 `outputs/`。

### 4. 交互式 Notebook

```bash
jupyter notebook notebooks/ddpm_tutorial.ipynb
```

Notebook 以内联代码、公式和可视化的方式逐步讲解每个概念。

## 核心公式

### 正向过程（加噪）

$$q(x_t \mid x_0) = \mathcal{N}\big(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t) I\big)$$

重参数化形式：

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

### 逆向过程（去噪）

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \varepsilon_\theta(x_t, t) \right) + \sigma_t z$$

### 训练损失

$$L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \varepsilon}\big[\| \varepsilon - \varepsilon_\theta(x_t, t) \|^2\big]$$

## 模型架构

对于二维数据，只需一个轻量级 MLP 配合**时间步条件化**：

```
输入 (2D) → ConditionalLinear(128) → Softplus
           → ConditionalLinear(128) → Softplus
           → ConditionalLinear(128) → Softplus
           → Linear(2) → 输出（预测噪声）
```

每个 `ConditionalLinear` 层通过嵌入表学习一个逐时间步的缩放向量，使同一组权重在不同扩散步骤中表现不同。

## 训练效果

在 CPU 上训练约 1000 个 epoch（几分钟）后，模型生成的点能够较好地还原瑞士卷分布的螺旋结构。

## 参考文献

- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS 2020.
- Sohl-Dickstein, J., et al. (2015). [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585). ICML 2015.
- Weng, L. (2021). [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) Lil'Log.
- 刘一懂. (2025, April 9). 【手撕Diffusion-01】DDPM 原理精讲：物理直觉 [Web log post]. 小红书. https://www.xiaohongshu.com/explore/67f62bdc000000001d0079cb
## 许可证

MIT
