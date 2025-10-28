"""
CIFAR-10 dataset that returns coordinate-value pairs for coordinate-based models.

Key differences from standard dataset:
- Returns (x, y) coordinates and RGB values as separate tensors
- Supports arbitrary query resolutions
- Dense supervision on all pixels by default
"""
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np


class CIFAR10CoordinateDataset(Dataset):
    """
    CIFAR-10 dataset returning coordinate-value pairs.

    Args:
        root: Data directory
        train: Training or test split
        input_ratio: Ratio of sparse input pixels (e.g., 0.2 = 20%)
        target_ratio: Ratio of target pixels for supervision (1.0 = dense, all pixels)
        image_size: Image resolution (default 32)
        download: Whether to download dataset
        seed: Random seed for reproducibility
    """
    def __init__(self, root='./data', train=True, input_ratio=0.2, target_ratio=1.0,
                 image_size=32, download=False, seed=42):
        super().__init__()

        # Load CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),  # [0, 1] range
        ])

        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )

        self.image_size = image_size
        self.num_pixels = image_size * image_size
        self.input_ratio = input_ratio
        self.target_ratio = target_ratio
        self.seed = seed

        # Number of input and target pixels
        self.num_input = int(self.num_pixels * input_ratio)
        self.num_target = int(self.num_pixels * target_ratio)

        # Create coordinate grid [0, 1] x [0, 1]
        y = torch.linspace(0, 1, image_size)
        x = torch.linspace(0, 1, image_size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self.coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (H*W, 2)

        # Fixed random generator for reproducible splits
        self.rng = np.random.RandomState(seed)

        # Cache for fixed instance masks
        self.input_masks = {}
        self.target_masks = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]  # (3, 32, 32), label

        # Flatten image
        image_flat = image.view(3, -1).T  # (H*W, 3)

        # Get fixed random masks for this instance
        if idx not in self.input_masks:
            # Generate masks once and cache
            all_indices = np.arange(self.num_pixels)

            # Input mask (sparse observations)
            input_indices = self.rng.choice(all_indices, self.num_input, replace=False)
            self.input_masks[idx] = input_indices

            # Target mask (supervision pixels)
            if self.target_ratio == 1.0:
                # Dense supervision: all pixels
                target_indices = all_indices
            else:
                # Sparse supervision: random subset
                # Exclude input pixels from targets for non-overlapping masks
                remaining = np.setdiff1d(all_indices, input_indices)
                target_indices = self.rng.choice(
                    remaining,
                    min(self.num_target, len(remaining)),
                    replace=False
                )
            self.target_masks[idx] = target_indices

        input_indices = self.input_masks[idx]
        target_indices = self.target_masks[idx]

        # Get coordinate-value pairs
        input_coords = self.coords[input_indices]      # (N_input, 2)
        input_values = image_flat[input_indices]       # (N_input, 3)

        target_coords = self.coords[target_indices]    # (N_target, 2)
        target_values = image_flat[target_indices]     # (N_target, 3)

        return {
            'input_coords': input_coords,
            'input_values': input_values,
            'target_coords': target_coords,
            'target_values': target_values,
            'full_image': image,  # For evaluation
            'label': label,
            'idx': idx
        }


def get_cifar10_coordinate_dataloader(root='./data', train=True, batch_size=64,
                                      input_ratio=0.2, target_ratio=1.0,
                                      image_size=32, num_workers=4, download=False, seed=42):
    """
    Create CIFAR-10 coordinate dataloader.

    Args:
        root: Data directory
        train: Training or test split
        batch_size: Batch size
        input_ratio: Ratio of sparse input pixels
        target_ratio: Ratio of target pixels (1.0 for dense supervision)
        image_size: Image resolution
        num_workers: Number of data loading workers
        download: Whether to download dataset
        seed: Random seed

    Returns:
        DataLoader
    """
    dataset = CIFAR10CoordinateDataset(
        root=root,
        train=train,
        input_ratio=input_ratio,
        target_ratio=target_ratio,
        image_size=image_size,
        download=download,
        seed=seed
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
