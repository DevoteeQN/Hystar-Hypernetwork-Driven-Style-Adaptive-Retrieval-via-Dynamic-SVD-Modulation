"""Minimal DSR datasets required for Hystar training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class _DSRBase(Dataset):
    def __init__(self, root_path: str, json_path: str, image_transform=None) -> None:
        self.root = Path(root_path)
        metadata_path = Path(json_path)
        if not metadata_path.is_absolute():
            metadata_path = self.root / metadata_path
        with metadata_path.open("r", encoding="utf-8") as handle:
            self.samples = json.load(handle)
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.samples)

    def _image(self, relative_path: Path):
        image = Image.open(relative_path).convert("RGB")
        return self.image_transform(image) if self.image_transform else image


class DSRTrainDataset(_DSRBase):
    """Natural-image target paired with one randomly selected query style."""

    def __init__(
        self,
        root_path: str,
        json_path: str = "train.json",
        image_transform=None,
        styles: tuple[str, ...] = ("sketch", "mosaic", "art"),
    ) -> None:
        super().__init__(root_path, json_path, image_transform)
        self.styles = styles

    def __getitem__(self, index: int):
        item = self.samples[index]
        image_name = item["image"]
        style = self.styles[np.random.randint(0, len(self.styles))]
        natural = self._image(self.root / "images" / image_name)
        styled = self._image(self.root / style / image_name)
        return {"ori": natural, "pair": styled, "style": style, "idx": index}


class DSRStyleTestDataset(_DSRBase):
    def __init__(self, style: str, root_path: str, json_path: str = "test.json", image_transform=None) -> None:
        super().__init__(root_path, json_path, image_transform)
        self.style = style

    def __getitem__(self, index: int):
        item = self.samples[index]
        image_name = item["image"]
        return {
            "ori": self._image(self.root / "images" / image_name),
            "pair": self._image(self.root / self.style / image_name),
            "idx": index,
        }


class DSRTextTestDataset(_DSRBase):
    def __getitem__(self, index: int):
        item = self.samples[index]
        caption_path = self.root / "text" / item["caption"]
        caption = caption_path.read_text(encoding="utf-8").splitlines()[0]
        return {
            "text": caption,
            "pair": self._image(self.root / "images" / item["image"]),
            "idx": index,
        }
