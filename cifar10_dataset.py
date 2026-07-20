"""
Dataset wrapper around torchvision's CIFAR-10 that returns (image, class_label) pairs
normalized to [-1, 1], which is a convenient convention for generative models whose
reconstructions/outputs are compared directly in this range.
"""

import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

CIFAR10_BASE_FOLDER = "cifar-10-batches-py"


def _resolve_cifar_root(root, max_search_depth=2):
    """
    torchvision.datasets.CIFAR10 expects `root` to directly contain a
    'cifar-10-batches-py' folder. Pre-attached dataset mounts (e.g. Kaggle's
    /kaggle/input/<dataset-slug>/cifar-10-batches-py/...) bury it one level
    deeper under a slug you can't predict in advance, so this searches a
    few levels down and returns the first directory whose immediate child
    is 'cifar-10-batches-py'. Falls back to `root` unchanged if nothing is
    found, so behavior for a normal local ./data root is unaffected.
    """
    if os.path.isdir(os.path.join(root, CIFAR10_BASE_FOLDER)):
        return root

    if not os.path.isdir(root):
        return root

    frontier = [root]
    for _ in range(max_search_depth):
        next_frontier = []
        for current_dir in frontier:
            try:
                entries = os.listdir(current_dir)
            except OSError:
                continue
            for entry in entries:
                candidate = os.path.join(current_dir, entry)
                if not os.path.isdir(candidate):
                    continue
                if os.path.isdir(os.path.join(candidate, CIFAR10_BASE_FOLDER)):
                    return candidate
                next_frontier.append(candidate)
        frontier = next_frontier

    return root


class CIFAR10Dataset(Dataset):
    """
    Loads CIFAR-10 images together with their class label.

    Args:
        root (str): directory to download/read CIFAR-10 from. If this directory
            doesn't directly contain a 'cifar-10-batches-py' folder, a shallow
            search is performed under it to find one (handles pre-attached
            dataset mounts like Kaggle's /kaggle/input/<slug>/ layout).
        train (bool): True for the 50k training split, False for the 10k test split.
        image_side_length (int): resize target (CIFAR-10 is already 32x32, but this
            keeps the dataset flexible if IMAGE_SIDE_LENGTH in configs.yml ever changes).
        augment (bool): whether to apply light data augmentation (random horizontal flip).
            Typically True for training, False for evaluation/FID real-image sampling.
        download (bool): whether to download CIFAR-10 if not already present in root.
            Should be False when pointed at a read-only mount (e.g. Kaggle input data);
            torchvision will raise if it can't find the data and download=False.
    """

    def __init__(self, root="./data", train=True, image_side_length=32, augment=True, download=True):
        super().__init__()

        transform_list = []

        if image_side_length != 32:
            transform_list.append(transforms.Resize((image_side_length, image_side_length)))

        if augment and train:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        transform_list.append(transforms.ToTensor())  # -> [0, 1], shape (3, H, W)
        # Normalize [0, 1] -> [-1, 1]
        transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        self.transform = transforms.Compose(transform_list)

        resolved_root = _resolve_cifar_root(root)

        self.base_dataset = datasets.CIFAR10(
            root=resolved_root,
            train=train,
            download=download,
            transform=self.transform,
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, class_id = self.base_dataset[idx]
        # image: (3, H, W) float tensor in [-1, 1]
        # class_id: python int in [0, 9] -> convert to tensor for consistent batching/typing
        return image, torch.tensor(class_id, dtype=torch.long)


def denormalize(images):
    """Utility: convert a batch of images from [-1, 1] back to [0, 1] for saving/viewing."""
    return (images.clamp(-1, 1) + 1.0) / 2.0


if __name__ == "__main__":
    # Quick smoke test (won't run automatically during training; just for manual sanity checking)
    ds = CIFAR10Dataset(root="./data", train=True, download=True)
    img, label = ds[0]
    print("Image shape:", img.shape, "dtype:", img.dtype, "range:", img.min().item(), img.max().item())
    print("Label:", label.item())
    print("Dataset size:", len(ds))
