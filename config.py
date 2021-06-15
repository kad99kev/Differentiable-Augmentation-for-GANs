from dataclasses import dataclass


@dataclass
class Config:
    name: str  # Name of the experiment
    augment: bool = True  # If True, augmentations are applied
    batch_size: int = 16  # Batch Size
    beta_1: float = 0.5  # Adam optimizer hyperparameter
    beta_2: float = 0.999  # Adam optimizer hyperparameter
    device: str = "cpu"  # Device to use
    eval_freq: int = 400  # Generate generator images every `eval_freq` epochs
    latent_dim: int = 100  # Dimensions of the random noise
    lr: float = 0.0002  # Learning rate
    ndf: int = 32  # Number of discriminator feature maps after first convolution
    ngf: int = 32  # Number of generator feature maps before last tranpose convolution
    epochs: int = 200  # Number of training epochs
    mosaic_size: int = 10  # Size of the rectangular mosaic
    prob: float = 0.9  # Probability of applying an augmentation