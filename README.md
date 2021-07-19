# Differentiable-Augmentation-for-GANs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z79pxwwnq377AFWdem7efJLMCC_UI3lY?usp=sharing)[![Visualize in WandB](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg)](https://wandb.ai/kad99kev/DiffAug)

---

* Minimal implementation of Differentiable Augmentation for Data-Efficient GAN Training. 
* Implemented with Deep Convolution Generative Adversarial Network and Self-Attention Generative Adversarial Network. 
* Training visualized with Weights and Biases. The augmentations have not been implemented from scratch but instead a library called [Kornia](https://github.com/kornia/kornia) a differentiable computer vision library for PyTorch is used.
* For a list of options that can be passed to the ```main.py``` file, [click here](https://github.com/kad99kev/Differentiable-Augmentation-for-GANs/blob/master/utils/parse.py).

## References
1. [Differentiable Augmentation for Data-Efficient GAN Training (Paper)](https://arxiv.org/pdf/2006.10738.pdf)
2. [Official repository](https://github.com/mit-han-lab/data-efficient-gans)
3. [Differentiable augmentation for GANs (using Kornia)](https://youtu.be/J97EM3Clyys)
