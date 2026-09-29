"""Zero-shot category retrieval on DomainNet."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.DomainNet import DomainNetDataset
from src.model.re_model import HystarCLIP, HystarConfig
from src.utils.utils import setup_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hystar on DomainNet retrieval")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--style", required=True, choices=["clipart", "sketch", "painting", "quickdraw", "infograph"])
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--dino-repo", default="facebookresearch/dinov2")
    parser.add_argument("--dino-model", default="dinov2_vitb14")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def extract_image_features(model, loader, device):
    features, labels = [], []
    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        feature = F.normalize(model.encode_image(images), dim=-1)
        features.append(feature)
        labels.append(batch["label"])
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def evaluate(model, gallery_loader, query_loader, device):
    model.eval()
    gallery_features, gallery_labels = extract_image_features(model, gallery_loader, device)
    query_features, query_labels = extract_image_features(model, query_loader, device)
    gallery_labels = gallery_labels.to(device)
    query_labels = query_labels.to(device)

    top1_correct = 0
    top5_correct = 0
    for start in range(0, query_features.size(0), 512):
        q = query_features[start : start + 512]
        q_labels = query_labels[start : start + 512]
        similarity = q @ gallery_features.t()
        topk = similarity.topk(min(5, gallery_features.size(0)), dim=1).indices
        retrieved = gallery_labels[topk]
        top1_correct += (retrieved[:, 0] == q_labels).sum().item()
        top5_correct += (retrieved == q_labels.unsqueeze(1)).any(dim=1).sum().item()

    n = query_features.size(0)
    return top1_correct / n, top5_correct / n


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

    gallery = DomainNetDataset("real", args.dataset_root, image_transform=model.pre_process_val)
    query = DomainNetDataset(args.style, args.dataset_root, image_transform=model.pre_process_val)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    r1, r5 = evaluate(
        model,
        DataLoader(gallery, **loader_kwargs),
        DataLoader(query, **loader_kwargs),
        args.device,
    )
    print(f"Top-1: {100 * r1:.2f}%")
    print(f"Top-5: {100 * r5:.2f}%")


if __name__ == "__main__":
    main()
