"""
CIFAR-10 dataset wrapper for masked conditioning experiments.
Returns (image, idx) tuples to preserve sample_id for mask reproducibility.
"""
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 wrapper that returns (image, sample_id) tuples.

    Args:
        root: Data directory (auto-downloads if needed)
        train: Whether to use train split
        augment_horizontal_flip: Whether to apply random horizontal flips
        download: Whether to download data if not present
    """
    def __init__(self, root='./data', train=True, augment_horizontal_flip=False, download=True):
        self.train = train

        # Build transforms
        transform_list = []
        if augment_horizontal_flip and train:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.ToTensor())  # Converts to [0,1]

        # Map to [-1, 1] as per prompt
        transform_list.append(transforms.Lambda(lambda x: x * 2 - 1))

        self.transform = transforms.Compose(transform_list)

        # Load CIFAR-10
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            transform=self.transform,
            download=download
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns (image, sample_id) where sample_id is the dataset index."""
        image, _ = self.dataset[idx]  # Ignore class label
        return image, idx


def get_cifar10_dataloader(root='./data', train=True, batch_size=128,
                            augment_horizontal_flip=False, num_workers=4,
                            shuffle=None, download=True):
    """
    Convenience function to create CIFAR-10 dataloader.

    Returns:
        DataLoader yielding (images, indices) with shape (B, 3, 32, 32), (B,)
    """
    dataset = CIFAR10Dataset(
        root=root,
        train=train,
        augment_horizontal_flip=augment_horizontal_flip,
        download=download
    )

    # Default shuffle behavior
    if shuffle is None:
        shuffle = train

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train  # Drop last incomplete batch during training
    )
