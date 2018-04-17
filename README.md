# SimGAN_PyTorch
A Simple Pytorch Implementation of Apple's [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/abs/1612.07828)

This code is based on [AlexHex7/SimGAN_pytorch](https://github.com/AlexHex7/SimGAN_pytorch)(PyTorch) and inspired by [wayaai/SimGAN](https://github.com/wayaai/SimGAN)(Keras&Tensorflow).

I try to re-organize AlexHex7's code, fix some bugs and speed up the training procedure.

# Time Cost

![Time Cost](https://raw.githubusercontent.com/automan000/SimGAN_PyTorch/master/images/time_cost.png)

With \[Batch Size=128, k_d=1k_g=50\], it takes about 12 hours to finish 10000 training steps.

# Current Results

Note that my parameters and optimizers are different from the original work.

## Results after 10000 iteration (About 1/4 of total steps in the original work)

![Result1](https://raw.githubusercontent.com/automan000/SimGAN_PyTorch/master/images/current_results.png)
