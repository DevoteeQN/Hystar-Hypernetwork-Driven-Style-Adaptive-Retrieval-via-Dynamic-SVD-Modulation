"""Train the compact Hystar-CLIP reference implementation on DSR."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.DSR import DSRTrainDataset
from src.model.re_model import HystarCLIP, HystarConfig
from src.utils.utils import count_trainable_parameters, setup_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hystar on DSR")
    parser.add_argument("--dataset-root", required=True, help="Path to the DSR dataset root")
    parser.add_argument("--train-json", default="train.json")
    parser.add_argument("--output-dir", default="outputs/hystar_clip")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint to resume/load")
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai", help="OpenCLIP pretrained tag or local checkpoint path")
    parser.add_argument("--dino-repo", default="facebookresearch/dinov2")
    parser.add_argument("--dino-model", default="dinov2_vitb14")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--static-lr", type=float, default=1e-3)
    parser.add_argument("--dynamic-lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--gamma", type=float, default=80.0)
    parser.add_argument("--ot-lambda", type=float, default=1.0)
    parser.add_argument("--sinkhorn-iters", type=int, default=50)
    return parser.parse_args()


def build_optimizer(model: HystarCLIP, static_lr: float, dynamic_lr: float) -> torch.optim.Optimizer:
    static_params = []
    dynamic_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".attn." in name:
            dynamic_params.append(parameter)
        elif ".mlp." in name:
            static_params.append(parameter)

    if not static_params or not dynamic_params:
        raise RuntimeError(
            f"Failed to identify modulation parameters: static={len(static_params)}, dynamic={len(dynamic_params)}"
        )
    return torch.optim.Adam(
        [
            {"params": static_params, "lr": static_lr},
            {"params": dynamic_params, "lr": dynamic_lr},
        ]
    )


def train_one_epoch(model, dataloader, optimizer, device: str, grad_accum: int) -> float:
    model.train()
    model.gram_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    sample_count = 0

    for step, batch in enumerate(tqdm(dataloader, desc="train", leave=False)):
        natural = batch["ori"].to(device, non_blocking=True)
        styled = batch["pair"].to(device, non_blocking=True)

        natural_style = model.extract_style(natural)
        styled_style = model.extract_style(styled)
        natural_features = model.encode_image(natural, natural_style)
        styled_features = model.encode_image(styled, styled_style)

        loss = model.retrieval_loss(styled_features, natural_features)
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = natural.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HystarConfig(
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        temperature=args.temperature,
        gamma=args.gamma,
        ot_lambda=args.ot_lambda,
        sinkhorn_iters=args.sinkhorn_iters,
        dino_repo=args.dino_repo,
        dino_model=args.dino_model,
    )
    model = HystarCLIP(config).to(args.device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)

    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    dataset = DSRTrainDataset(
        root_path=args.dataset_root,
        json_path=args.train_json,
        image_transform=model.pre_process_train,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    optimizer = build_optimizer(model, args.static_lr, args.dynamic_lr)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, optimizer, args.device, args.grad_accum)
        print(f"Epoch {epoch + 1:03d}/{args.epochs:03d} | loss={loss:.6f}")
        torch.save(model.state_dict(), output_dir / "last.pt")
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), output_dir / "best.pt")


if __name__ == "__main__":
    main()
