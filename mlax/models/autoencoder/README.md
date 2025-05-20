# Auto-Encoder

[![wandb badge](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)](https://wandb.ai/chnyutao/mlax/runs/s1nsdryv)

Auto-encoders perform non-linear dimensionality reduction by minimizing a self-reconstruction loss $\lVert x-\hat{x}\rVert^2$, where $\hat{x}=g_\phi(f_\theta(x))$, $f_\theta$ is an encoder network parametrized by $\theta$, and $g_\phi$ is a decoder network parametrized by $\phi$.

> [!Note]
> A low dimensional representation $z$ of arbitrary input $x$ can be obtained by computing $z=f_\theta(x)$.

## Experiment results

![plot](https://api.wandb.ai/files/chnyutao/mlax/s1nsdryv/media/images/plot_18760_a29d2a69ccb605299bf5.png)
