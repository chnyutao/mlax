# LeNet

[![wandb badge](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)](https://wandb.ai/chnyutao/mlax/runs/tuey07vz)

LeNet[^1], named after the Turing Award Laureate Yann Lecun, is a series of convolutional neural networks (CNNs) designed for handwritten digit recognition. It is one of the earliest and most classic convolutional neural networks, and laid the groundwork for many modern deep neural network architectures.

> [!Note]
> As a practice, here we are implementing the convolution and pooling layers manually with `jax`. One can always turn to well-maintained neural network libraries for convenience and _probably_ better performance.

## Discussion

There are two weight & bias runs associated with the LeNet model:
- One ([link](https://wandb.ai/chnyutao/mlax/runs/islkz711)) uses the vanilla LeNet-5 network with sigmoid activations and 3e-2 learning rate, and achieved ~98.81% predictive accuracy on the test set after 10 epochs.
- The other ([link](https://wandb.ai/chnyutao/mlax/runs/tuey07vz)) replaces sigmoid activations with relu activations and uses 1e-4 learning rate, and achieved ~95.38% predictive accuracy on the test set after 10 epochs.

While LeNet with sigmoid activations performs slightly better (within 10 epochs), we found that in practice it is much more fragile to train. One to need make the learning rate much larger (3e-2 $\gg$ 1e-4) to counter the _vanishing gradient problem_ of sigmoid activations, yet the larger learning rate often results in oscilliating loss curves.

> ### Vanishing Gradient Problem
> Consider the sigmoid activation function $\sigma(x)= 1/(1+\exp({-x}))$ and its first-order derivative $\sigma^\prime(x)=\sigma(x)(1-\sigma(x))$, whose maximum is achieve at $x=0$:
>
> $$\max\sigma^\prime(x)=\sigma(0)(1-\sigma(0))=0.25.$$
>
> Now since there are four sigmoid activations in LeNet, all gradients w.r.t. the parameters in the first convolutional layer will be scaled down by a factor of (at least) $0.25^4\approx0.0039$.

## Experiment results

Experiments are conducted on the MNIST[^1] dataset to predict the label (0-9) given an handwritten digit image. The final predictive accuracy on the test set was ~98.81% (sigmoid) and ~95.38% (relu).

![plot](https://api.wandb.ai/files/chnyutao/mlax/islkz711/media/images/plot_9390_4dd8daf5a88a68de1ccc.png)

[^1]: LeCun, Yann, et al. "Gradient-based learning applied to document recognition." _Proceedings of the IEEE_ 86.11 (1998): 2278-2324.
