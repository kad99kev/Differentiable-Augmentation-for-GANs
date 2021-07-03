import pathlib
from torch.utils.data import Dataset
from PIL import Image


class DatasetImages(Dataset):
    """
    Dataset loading photos from a given path.

    Arguments:
        path (pathlib.Path): Path to the folder containing all images.
        transform (None or callable): The transforms to be applied

    Attributes:
        all_paths (list): List of all paths to the images.
    """

    def __init__(self, path: pathlib.Path, transform=None) -> None:
        super().__init__()

        self.all_paths = sorted([p for p in path.iterdir() if p.suffix == ".jpg"])
        self.transform = transform

    def __len__(self) -> int:
        """
        Compute length of the dataset.
        """
        return len(self.all_paths)

    def __getitem__(self, idx) -> Image:
        """
        Get a single item.
        """
        image = Image.open(self.all_paths[idx])

        if self.transform:
            image = self.transform(image)

        return image
