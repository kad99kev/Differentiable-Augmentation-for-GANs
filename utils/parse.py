import argparse


def parse_args(argv=None) -> argparse.Namespace:
    # Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", help="Name of the experiment")
    parser.add_argument(
        "-a",
        "--augment",
        action="store_true",
        help="If True, augmentations are applied",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="dcgan",
        choices=["dcgan", "sagan"],
        help="Type of model architecture",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Adam optimizer hyperparameter"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Adam optimizer hyperparameter"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=400,
        help="Generate generator images every `eval_freq` epochs",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=100, help="Dimensions of the random noise"
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning Rate")
    parser.add_argument(
        "--ndf",
        type=int,
        default=32,
        help="Number of discriminator feature maps after first convolution",
    )
    parser.add_argument(
        "--ngf",
        type=int,
        default=32,
        help="Number of generator feature maps before last tranpose convolution",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=200, help=" Number of training epochs"
    )
    parser.add_argument(
        "--mosaic-size", type=int, default=10, help="Size of the rectangular mosaic"
    )
    parser.add_argument(
        "-p",
        "--prob",
        type=float,
        default=0.9,
        help="Probability of applying an augmentation",
    )

    args = parser.parse_args(argv)
    print(args)
    return args