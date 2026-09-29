"""DomainNet test split loader."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def get_class_from_path(image_path: str | Path) -> str:
    return Path(image_path).parent.name


class DomainNetDataset(Dataset):
    def __init__(self, style: str, root_path: str, split: str = "test.txt", image_transform=None) -> None:
        self.style = style
        self.root = Path(root_path)
        self.image_transform = image_transform
        split_file = self.root / f"{style}_{split}"

        self.samples: list[tuple[str, int]] = []
        with split_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                path, label = line.rsplit(" ", 1)
                self.samples.append((path, int(label)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        relative_path, label = self.samples[index]
        image_path = self.root / relative_path
        image = Image.open(image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        class_name = get_class_from_path(image_path)
        return {
            "image": image,
            "label": label,
            "class_name": class_name,
            "prompt": f"a photo of {class_name}",
        }
