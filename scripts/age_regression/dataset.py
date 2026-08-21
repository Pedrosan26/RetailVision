"""
dataset.py

PyTorch Dataset that reads an age-regression manifest CSV (image path,
continuous age) and applies the same augmentation philosophy documented in
the classifiers' (flip safe, no vertical flip, mild rotation, mild
color jitter that doesn't distort skin tone) via torchvision transforms.
"""

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import IMGSZ

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class AgeRegressionDataset(Dataset):
    def __init__(self, manifest_csv, transform) -> None:
        """Load a manifest CSV (path, age, gender) and store the given image transform."""
        self.rows = pd.read_csv(manifest_csv)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of images in this split."""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (transformed image tensor, age as a float32 scalar tensor)."""
        row = self.rows.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        age = torch.tensor(float(row["age"]), dtype=torch.float32)
        return image, age