# Variational Auto-Encoder

[![wandb badge](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)](https://wandb.ai/chnyutao/mlax/runs/62ct64g6)

Variational auto-encoders [^1] (VAEs) are a class of powerful class of generative models. Despite the naming resemblence, VAEs are fundamentally different from auto-encoders, which are typically used for dimensionality reduction but not generative modeling.

Variational auto-encoders updates the parameters $(\theta,\phi)$ of the encoder $z\sim f_\theta(\cdot\mid x)$ and decoder $\hat{x}=g_\phi(z)$ by maximizing the evidence lower bound (ELBO) objective, which is the sum of a reconstruction error between $x$ and $\hat{x}$ (e.g. mean squared error) and a KL divergence term.

To sample from the data distribution $p(x)$, one can first sample a random noise $z\sim\mathcal{N}(0,I)$ and then pass it to the decoder $\hat{x}=g(z)$ to obtain a sample $\hat{x}$.

## Experiment results

![plot](https://api.wandb.ai/files/chnyutao/mlax/62ct64g6/media/images/plot_18760_0d839aad6fa7f59d942d.png)

[^1]: Kingma, Diederik P., and Max Welling. "Auto-Encoding Variational Bayes." _arXiv preprint arXiv:1312.6114_ (2013).
