import pathlib
from numpy import dtype

from sklearn.utils import shuffle
from utils import DatasetImages
import wandb
from dataclasses import dataclass
from datetime import datetime

import kornia.augmentation as K
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dcgan import Generator, Discriminator, init_weights_
from tqdm import tqdm


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


config = Config(name="DiffAug with DCGAN")

image_size = 128

# Additional parameters
device = torch.device(config.device)
mosaic_kwargs = {"nrow": config.mosaic_size, "normalize": True}
n_mosaic_cells = config.mosaic_size * config.mosaic_size
sample_showcase_idx = 0  # Will be used to demonstrate the augmentations

augment_module = torch.nn.Sequential(
    K.RandomAffine(degrees=0, translate=(1 / 8, 1 / 8), p=config.prob),
    K.RandomErasing((0.0, 0.5), p=config.prob),
)

# Initialize generator and discriminator
generator = Generator(latent_dim=config.latent_dim, ngf=config.ngf)
discriminator = Discriminator(
    ndf=config.ndf, augement_module=augment_module if config.augment_module else None
)

# Shift models to device
generator.to(device)
discriminator.to(device)

# Initialize weights
generator.apply(init_weights_)
discriminator.apply(init_weights_)

# Configure DataLoader
data_path = pathlib.Path("data")
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# Dataset
dataset = DatasetImages(data_path, transform=transforms)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# Optimizers
optim_G = torch.optim.Adam(
    generator.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2)
)
optim_D = torch.optim.Adam(
    discriminator.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2)
)

# Loss
adversarial_loss = torch.nn.BCELoss()

# Wandb
wandb.init(config=config, project="DiffAug")

# Log true data
wandb.log(
    {
        "true_data": make_grid(
            torch.stack([dataset[i] for i in range(n_mosaic_cells)]), **mosaic_kwargs
        )
    }
)

# Log augmented data
batch_showcase = dataset[sample_showcase_idx][None, ...].repeat(n_mosaic_cells, 1, 1, 1)
batch_showcase_aug = discriminator.augment_module(batch_showcase)
wandb.log({"augmented_data": make_grid(batch_showcase_aug, **mosaic_kwargs)})

# Prepare evaluation noise
z_eval = torch.randn(n_mosaic_cells, config.latent_dim).to(device)

# Training Loop
for epoch in tqdm(range(config.epochs)):
    for i, images in enumerate(dataloader):
        n_samples, *_ = images.shape
        batches_done = epoch * len(dataloader) * i

        # Adversarial ground truths
        valid = 0.9 * torch.ones(n_samples, 1, device=device, dtype=torch.float32)
        fake = torch.zeros(n_samples, 1, device=device, dtype=torch.float32)

        # Training D
        optim_D.zero_grad()

        # D loss on real
        real_images = images.to(device)
        d_out = discriminator(real_images)
        real_loss = adversarial_loss(d_out, valid)
        real_loss.backward()

        # D loss on fake
        z = torch.randn(n_samples, config.latent_dim).to(device)
        generated_images = generator(z)
        d_fake_out = discriminator(generated_images.detach())

        fake_loss = adversarial_loss(d_fake_out, fake)
        fake_loss.backward()

        optim_D.step()

        # Training D
        optim_G.zero_grad()

        # G loss
        d_fake_out2 = discriminator(generated_images)
        g_loss = adversarial_loss(d_fake_out2, valid)

        g_loss.backward()
        optim_G.step()
