"""Dataset loaders for MVTec-AD-style folders.

<root>/<category>/train/good/*.png
<root>/<category>/test/<good|defect_type>/*.png
<root>/<category>/ground_truth/<defect_type>/<name>_mask.png
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # [0,1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class NormalDataset(Dataset):
    """Defect-free images only (used for training / validation)."""

    def __init__(self, paths, size=256, augment=False):
        self.paths, self.size, self.augment = list(paths), size, augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = pil_to_tensor(Image.open(self.paths[i]), self.size)
        if self.augment and random.random() < 0.5:
            x = torch.flip(x, dims=[2])                       # horizontal flip
        return x


class TestDataset(Dataset):
    """Official test split: returns image, mask, label (0 good / 1 defect), path, defect type."""

    def __init__(self, category_root, size=256):
        self.size = size
        root = Path(category_root)
        self.items = []
        for p in list_images(root / "test"):
            defect = p.parent.name
            label = 0 if defect == "good" else 1
            mask = None
            if label:
                cand = root / "ground_truth" / defect / f"{p.stem}_mask.png"
                if not cand.exists():
                    cand = root / "ground_truth" / defect / f"{p.stem}.png"
                mask = cand if cand.exists() else None
            self.items.append((p, label, mask, defect))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, label, mask_path, defect = self.items[i]
        x = pil_to_tensor(Image.open(p), self.size)
        if mask_path is not None:
            m = Image.open(mask_path).convert("L").resize((self.size, self.size), Image.NEAREST)
            m = (torch.from_numpy(np.asarray(m, dtype=np.float32)) > 127).float().unsqueeze(0)
        else:
            m = torch.zeros(1, self.size, self.size)
        return x, m, label, str(p), defect


def split_normal_train(category_root, val_split=0.1, seed=42):
    paths = list_images(Path(category_root) / "train" / "good")
    if not paths:
        raise FileNotFoundError(
            f"No training images in {category_root}/train/good - check data_root/category in the config."
        )
    rng = random.Random(seed)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * val_split))
    return paths[n_val:], paths[:n_val]
