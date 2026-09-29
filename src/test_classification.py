"""Zero-shot DomainNet classification with Hystar image features and CLIP text prototypes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.DomainNet import DomainNetDataset
from src.model.re_model import HystarCLIP, HystarConfig
from src.utils.utils import setup_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hystar on DomainNet classification")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--style", required=True, choices=["real", "clipart", "sketch", "painting", "quickdraw", "infograph"])
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--dino-repo", default="facebookresearch/dinov2")
    parser.add_argument("--dino-model", default="dinov2_vitb14")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_class_list(root: str) -> list[str]:
    real_dir = Path(root) / "real"
    if not real_dir.is_dir():
        raise FileNotFoundError(f"Expected DomainNet class folders under {real_dir}")
    return sorted(path.name for path in real_dir.iterdir() if path.is_dir())


@torch.no_grad()
def evaluate(model, loader, class_list, device):
    model.eval()
    prompts = [f"a photo of {name}" for name in class_list]
    text_tokens = model.tokenizer(prompts).to(device)
    text_features = F.normalize(model.encode_text(text_tokens), dim=-1)
    class_to_id = {name: i for i, name in enumerate(class_list)}

    correct = 0
    total = 0
    for batch in tqdm(loader, desc="classification"):
        images = batch["image"].to(device, non_blocking=True)
        labels = torch.tensor([class_to_id[name] for name in batch["class_name"]], device=device)
        image_features = F.normalize(model.encode_image(images), dim=-1)
        predictions = (image_features @ text_features.t()).argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return correct / total


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    model = HystarCLIP(
        HystarConfig(
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            dino_repo=args.dino_repo,
            dino_model=args.dino_model,
        )
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=False)
    model = model.to(args.device)

    dataset = DomainNetDataset(args.style, args.dataset_root, image_transform=model.pre_process_val)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    accuracy = evaluate(model, loader, get_class_list(args.dataset_root), args.device)
    print(f"Accuracy: {100 * accuracy:.2f}%")


if __name__ == "__main__":
    main()
