import os
from pathlib import Path
from typing import Union, List, Dict

import torch
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from kaggle.api.kaggle_api_extended import KaggleApi

class DogBreedImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        num_workers: int = 0,
        batch_size: int = 32,
        train_val_test_split: List[float] = [0.8, 0.1, 0.1],
        pin_memory: bool = False,
        dataset_name: str = "khushikhushikhushi/dog-breed-image-dataset",
        image_size: List[int] = [224, 224],
        augmentation: Dict[str, bool] = {"horizontal_flip": True},
        normalization: Dict[str, List[float]] = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_workers = self._get_num_workers(num_workers)
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.pin_memory = pin_memory
        self.dataset_name = dataset_name
        self.image_size = tuple(image_size)
        self.augmentation = augmentation
        self.normalization = normalization
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Define the image transformations for training and validation
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()

        self.dataset_path = self.data_dir / 'dataset'

    def _get_num_workers(self, default: int) -> int:
        """Automatically determine number of workers based on available device."""
        if torch.cuda.is_available():
            return min(default, os.cpu_count() * 2)
        else:
            return 0
        
    def _create_train_transform(self):
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalization["mean"], std=self.normalization["std"])
        ]
        if self.augmentation.get("horizontal_flip", False):
            transform_list.insert(1, transforms.RandomHorizontalFlip())
        return transforms.Compose(transform_list)

    def _create_val_transform(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalization["mean"], std=self.normalization["std"])
        ])

    def prepare_data(self):
        """Download images if not already present."""
        if self.dataset_path.exists() and any(self.dataset_path.iterdir()):
            print(f"Dataset already exists in {self.dataset_path}. Skipping download.")
            return

        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Download the dataset from Kaggle
        api.dataset_download_files(self.dataset_name, path=self.data_dir, unzip=True)

        print(f"Dataset downloaded and extracted in {self.data_dir}")

    def setup(self, stage: str = None):
        """Setup datasets based on the stage (fit/test)."""
        print(f"Setting up data from: {self.dataset_path}")

        full_dataset = ImageFolder(root=self.dataset_path, transform=self.train_transform)
        print(f"Number of classes detected: {len(full_dataset.classes)}")
        print(f"Classes: {full_dataset.classes}")
        print(f"Total number of samples: {len(full_dataset)}")

        # Split the dataset into train, validation, and test sets
        total_size = len(full_dataset)
        train_size = int(self.train_val_test_split[0] * total_size)
        val_size = int(self.train_val_test_split[1] * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size, test_size]
        )

        # Apply appropriate transforms
        self.train_dataset.dataset.transform = self.train_transform
        self.val_dataset.dataset.transform = self.val_transform
        self.test_dataset.dataset.transform = self.val_transform

        print(f"Training samples: {len(self.train_dataset)}, Validation samples: {len(self.val_dataset)}, Test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)
        print(f"Train dataloader created with {len(loader.dataset)} samples")
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
        print(f"Validation dataloader created with {len(loader.dataset)} samples")
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
        print(f"Test dataloader created with {len(loader.dataset)} samples")
        return loader

# Example usage
if __name__ == "__main__":
    datamodule = DogBreedImageDataModule(
        data_dir="data",  # Default data directory
        batch_size=32,
        num_workers=0  # Will be set automatically in the constructor
    )
    
    # Prepare data (download dataset)
    datamodule.prepare_data()
    
    # Setup (create train/val/test splits)
    datamodule.setup(stage="fit")
    
    # Access data loaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    print("DataModule setup complete. Ready for model training.")
    print(f"Train loader has {len(train_loader.dataset)} samples.")
    print(f"Validation loader has {len(val_loader.dataset)} samples.")
    print(f"Test loader has {len(test_loader.dataset)} samples." if test_loader.dataset else "No test dataset loaded.")