import torch
import torch.nn as nn


def init_weights_(module) -> None:
    """
    Initialize weight by sampling from a normal distribution.
    This operation modifies the weights in place.

    Arguments:
        module (nn.Module): Module with trainable weights.
    """
    cls_name = module.__class__.__name__

    if cls_name in {"Conv2d", "ConvTranspose2d"}:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

    elif cls_name == "BatchNorm2d":
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class SelfAttention(nn.Module):
    """
    Self Attention Layer.

    Arguements:
        input_dim (int): Input dimensions.
    """

    def __init__(self, input_dim) -> None:
        super().__init__()

        self.query_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=input_dim // 2, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=input_dim // 2, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=input_dim, kernel_size=1
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward Pass

        Arguments:
            x: Input vector.

        Returns:
            torch.Tensor: Generated images of shape `(n_samples, 3, 128, 128)`.
        """
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)

        attn = self.softmax(energy)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attn.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out


class Generator(nn.Module):
    """
    Generator Network.

    Arguments:
        latent_dim (int): Dimensions of the input noise.
        ngf (int): Number of generator filters.

    Attributes:
        main (torch.Sequential): The actual network that is composed on `ConvTranspose2d`, `BatchNorm2d` and `ReLU` blocks.
    """

    def __init__(self, latent_dim: int, ngf=64) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(latent_dim, ngf * 16, 4, 1, 0, bias=False)
            ),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # (ngf * 16) x 4 x 4
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf * 8) x 8 x 8
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf * 4) x 16 x 16
            SelfAttention(ngf * 4),
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf * 4) x 32 x 32
            SelfAttention(ngf * 2),
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf * 2) x 64 x 64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 3 x 128 x 128
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass

        Arguments:
            inputs: Input noise of shape `(n_samples, latent_dim)`.

        Returns:
            torch.Tensor: Generated images of shape `(n_samples, 3, 128, 128)`.
        """
        x = inputs.reshape(*inputs.shape, 1, 1)  # (n_samples, latent_dim, 1, 1)
        return self.main(x)


class Discriminator(nn.Module):
    """
    Discriminator Network.

    Arguments:
        ndf (int): Number of discriminator filters.
        augment_module (nn.Module or None): If provided it represents the Kornia module that perfroms differntiable augmentation of the images.

    Attributes:
        augment_module (nn.Module): If the input parameter `augment_module` is provided, then it is the same thing. If not, then it is just an identity mapping.
    """

    def __init__(self, ndf=16, augment_module=None) -> None:
        super().__init__()
        self.main = nn.Sequential(
            # 3 x 128 x 128
            nn.utils.spectral_norm(nn.Conv2d(3, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 64 x 64
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 2) x 32 x 32
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(ndf * 4),
            # (ndf * 4) x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(ndf * 8),
            # (ndf * 8) x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 1 x 1 x 1
        )

        if augment_module is not None:
            self.augment_module = augment_module
        else:
            self.augment_module = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass.

        Arguments:
            inputs (torch.Tensor): Input images of shape` (n_samples, 3, 128, 128)`.

        Returns:
            torch.Tensor: Classification ouputs of shape `(n_samples, 1)`.
        """
        if self.training:
            x = self.augment_module(inputs)
        x = self.main(x)  # (n_samples, 1, 1, 1)
        x = x.squeeze()[:, None]  # (n_samples, 1)
        return x
