"""Loading Images Module:
This module defines a PyTorch Dataset for loading a collection of images from a directory or a single image.

Dependencies:
    - pathlib
    - PIL (Pillow)
    - torch.utils.data.Dataset

Usage:
    Instantiate the MixedImagesDataset class to create a PyTorch Dataset for loading images.
    Specify the path to a directory containing images
    or the path to a single image. Optionally, provide a transform function to apply to each image.

Example:
    dataset = MixedImagesDataset('/path/to/images', transform=transforms.ToTensor())
    image_path, image = dataset[0]

Author:
    Dr. Aleksei Krasnov
    a.krasnov@digital-science.com
    Date: December 4, 2023
"""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class MixedImagesDataset(Dataset):
    def __init__(self, path_or_dir, transform=None):
        """
        A PyTorch Dataset for loading a collection of images from a directory or a single image.

        Parameters:
        - path_or_dir (str or Path): Path to a directory containing images or a path to a single image.
        - transform (callable, optional): A function/transform to apply to each image.
        """
        self.path_or_dir = Path(path_or_dir) if isinstance(path_or_dir, str) else path_or_dir
        self.transform = transform
        self.image_paths = self.get_image_paths()

    def get_image_paths(self):
        """
        Get a list of image paths based on whether the input is a directory or a single image.

        Returns:
        - List[Path]: List of image paths.
        """
        image_paths = []
        if isinstance(self.path_or_dir, list):
            for path in self.path_or_dir:
                path = Path(path)
                if path.is_dir():
                    image_paths.extend([img_path for img_path in path.glob('[!.]*') if self.is_image(img_path)])
                elif self.is_image(path):
                    image_paths.append(path)
                else:
                    raise ValueError(f"Invalid path or directory provided: {path}. File is not an image")
        elif self.path_or_dir.is_dir():
            image_paths = [img_path for img_path in self.path_or_dir.glob('[!.]*') if self.is_image(img_path)]
        elif self.is_image(self.path_or_dir):
            image_paths = [self.path_or_dir]
        else:
            raise ValueError("Invalid path or directory provided. File is not an image")
        return image_paths

    def __len__(self):
        """
        Get the number of images in the dataset.

        Returns:
        - int: Number of images.
        """
        return len(self.image_paths)

    @staticmethod
    def is_image(path):
        """
        Check if a file is a valid image file.

        Parameters:
        - path (Path): Path to the image file.

        Returns:
        - bool: True if the file is a valid image, False otherwise.
        """
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except (IOError, SyntaxError) as e:
            print(f"Invalid image file: {path}. Error: {e}")
            return False

    def __getitem__(self, idx):
        """
        Get the image and its path at a specific index.

        Parameters:
        - idx (int): Index of the image.

        Returns:
        - Tuple[str, PIL.Image.Image]: A tuple containing the image path (as a string) and the image.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return str(img_path), image
