import pathlib
import wandb

import kornia.augmentation as K
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from dcgan import Generator, Discriminator, init_weights_
from utils import DatasetImages
from parse import parse_args


def main() -> None:
    config = parse_args()

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
        ndf=config.ndf,
        augement_module=augment_module if config.augment else None,
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

    # Outputs
    output_path = pathlib.Path("outputs") / config.name
    output_path.mkdir(exist_ok=True, parents=True)

    # Wandb
    wandb.init(config=config, project="DiffAug", name=config.name)

    # Log true data
    wandb.log(
        {
            "true_data": make_grid(
                torch.stack([dataset[i] for i in range(n_mosaic_cells)]),
                **mosaic_kwargs
            )
        }
    )

    # Log augmented data
    batch_showcase = dataset[sample_showcase_idx][None, ...].repeat(
        n_mosaic_cells, 1, 1, 1
    )
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

            # Logging
            if batches_done % 50 == 0:
                wandb.log(
                    {
                        "d_out": d_out.mean().item(),
                        "d_fake_out": d_fake_out.mean().item(),
                        "d_fake_out2": d_fake_out2.mean().item,
                        "discriminator_loss": (real_loss + fake_loss).item(),
                        "generator_loss": g_loss.item(),
                    },
                    step=epoch,
                )

            if epoch % config.eval_freq == 0 and i == 0:
                generator.eval()
                discriminator.eval()

                # Geenrate fake images
                g_eval_images = generator(z_eval)

                # Generate mosaic
                wandb.log(
                    {"fake": make_grid(g_eval_images.data, **mosaic_kwargs)}, step=epoch
                )

                # Save checkpoint
                torch.save(generator, output_path / "model.pt")

                # Switch back to training
                generator.train()
                discriminator.train()


if __name__ == "__main__":
    main()