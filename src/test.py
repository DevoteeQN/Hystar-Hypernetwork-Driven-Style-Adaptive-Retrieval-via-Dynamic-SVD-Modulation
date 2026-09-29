"""Evaluate Hystar on paired DSR retrieval."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.DSR import DSRStyleTestDataset, DSRTextTestDataset
from src.model.re_model import HystarCLIP, HystarConfig
from src.utils.utils import setup_seed, topk_paired_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hystar on DSR")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--test-json", default="test.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query-type", choices=["style", "text"], default="style")
    parser.add_argument("--style", default="sketch", help="e.g. art, sketch, mosaic")
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--dino-repo", default="facebookresearch/dinov2")
    parser.add_argument("--dino-model", default="dinov2_vitb14")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: HystarCLIP, loader: DataLoader, device: str, query_type: str) -> tuple[float, float]:
    model.eval()
    weighted_r1 = 0.0
    weighted_r5 = 0.0
    total = 0

    for batch in tqdm(loader, desc="eval"):
        if query_type == "text":
            natural = batch["pair"].to(device, non_blocking=True)
            tokens = model.tokenizer(batch["text"]).to(device, non_blocking=True)
            query_features = F.normalize(model.encode_text(tokens), dim=-1)
            target_features = F.normalize(model.encode_image(natural), dim=-1)
        else:
            natural = batch["ori"].to(device, non_blocking=True)
            styled = batch["pair"].to(device, non_blocking=True)
            query_features = F.normalize(model.encode_image(styled), dim=-1)
            target_features = F.normalize(model.encode_image(natural), dim=-1)

        similarity = query_features @ target_features.t()
        batch_size = similarity.size(0)
        weighted_r1 += topk_paired_accuracy(similarity, 1) * batch_size
        weighted_r5 += topk_paired_accuracy(similarity, 5) * batch_size
        total += batch_size

    return weighted_r1 / total, weighted_r5 / total


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    config = HystarConfig(
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        dino_repo=args.dino_repo,
        dino_model=args.dino_model,
    )
    model = HystarCLIP(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=False)
    model = model.to(args.device)

    if args.query_type == "text":
        dataset = DSRTextTestDataset(args.dataset_root, args.test_json, model.pre_process_val)
    else:
        dataset = DSRStyleTestDataset(args.style, args.dataset_root, args.test_json, model.pre_process_val)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    r1, r5 = evaluate(model, loader, args.device, args.query_type)
    print(f"Top-1: {100 * r1:.2f}%")
    print(f"Top-5: {100 * r5:.2f}%")


if __name__ == "__main__":
    main()
