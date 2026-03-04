#!/usr/bin/env python3
"""
Train Faster R-CNN FROM SCRATCH (random init) using your CSV-format boxes.

Expected CSV columns:
  - image_file_name
  - category_name
  - bbox_x, bbox_y, bbox_w, bbox_h   (pixel coords in original image space)

Example:
  python train_frcnn_from_csv_scratch.py \
    --train-images /path/to/train_images \
    --train-csv /path/to/train.csv \
    --val-images /path/to/val_images \
    --val-csv /path/to/val.csv \
    --num-classes 70 \
    --output-path /path/to/out \
    --epochs 50 --batch-size 4 --lr 0.002 --ngpus 1
"""

import os, argparse, time, json, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.transforms import functional as F


# -----------------------
# Label map (deterministic)
# -----------------------
def build_label_map_from_csv(csv_path: str, expected_classes: int | None = None):
    labels = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.add(row["category_name"].strip())
    labels = sorted(labels)

    if expected_classes is not None and len(labels) != expected_classes:
        print(f"WARNING: found {len(labels)} unique category_name values in {csv_path} "
              f"(expected {expected_classes}).")

    # detection labels: 0 is background; classes are 1..C
    label_to_idx = {lab: i + 1 for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


# -----------------------
# Dataset
# -----------------------
class DetectionBBoxCSVDataset(torch.utils.data.Dataset):
    """
    Returns:
      image: FloatTensor [3,H,W] in [0,1]
      target: dict with keys:
        - boxes: FloatTensor [N,4] in XYXY pixel coords
        - labels: LongTensor [N] in 1..C
        - image_id: LongTensor [1]
        - area: FloatTensor [N]
        - iscrowd: LongTensor [N] zeros
    """
    def __init__(self, images_root: str, csv_path: str, label_to_idx: dict,
                 resize_short: int | None = 512, max_size: int | None = 768):
        self.images_root = Path(images_root)
        self.csv_path = Path(csv_path)
        self.label_to_idx = label_to_idx
        self.resize_short = resize_short
        self.max_size = max_size

        img_to_anns = defaultdict(list)
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["image_file_name"]
                cat = row["category_name"].strip()
                if cat not in self.label_to_idx:
                    continue

                x = float(row["bbox_x"])
                y = float(row["bbox_y"])
                w = float(row["bbox_w"])
                h = float(row["bbox_h"])
                x1, y1, x2, y2 = x, y, x + w, y + h
                img_to_anns[img_name].append((self.label_to_idx[cat], [x1, y1, x2, y2]))

        self.items = []
        missing = 0
        for img_name, anns in img_to_anns.items():
            p = self.images_root / img_name
            if p.exists():
                self.items.append((p, anns))
            else:
                missing += 1

        if len(self.items) == 0:
            raise RuntimeError(f"No matched images found in {self.images_root} from {self.csv_path}")
        if missing:
            print(f"Warning: {missing} image_file_name rows had no matching file in {self.images_root}")

    def __len__(self):
        return len(self.items)

    def _maybe_resize(self, img: Image.Image, boxes: torch.Tensor):
        if self.resize_short is None:
            return img, boxes

        w, h = img.size
        short = min(h, w)
        long = max(h, w)

        scale = self.resize_short / float(short)
        if self.max_size is not None and (long * scale) > self.max_size:
            scale = self.max_size / float(long)

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        if boxes.numel() > 0:
            boxes = boxes * torch.tensor([scale, scale, scale, scale], dtype=torch.float32)
        return img, boxes

    def __getitem__(self, idx):
        img_path, ann_list = self.items[idx]
        img = Image.open(img_path).convert("RGB")

        labels = torch.tensor([a[0] for a in ann_list], dtype=torch.long)
        boxes = torch.tensor([a[1] for a in ann_list], dtype=torch.float32)

        # clamp + filter invalid (original size)
        W, H = img.size
        if boxes.numel() > 0:
            boxes[:, 0].clamp_(0, W); boxes[:, 2].clamp_(0, W)
            boxes[:, 1].clamp_(0, H); boxes[:, 3].clamp_(0, H)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]

        # optional resize
        img, boxes = self._maybe_resize(img, boxes)

        # tensorize
        img_t = F.to_tensor(img)  # [0,1]

        # target dict
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if boxes.numel() > 0 else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.long),
            "area": area,
            "iscrowd": iscrowd,
        }
        return img_t, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# -----------------------
# Model (FROM SCRATCH)
# -----------------------
def build_frcnn_random_init(num_classes: int):
    # Random init for everything
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    # Replace head for your classes (+ background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        num_classes=num_classes + 1
    )
    return model


@torch.no_grad()
def evaluate_loss(model, loader, device):
    # torchvision detection returns losses in train() mode
    model.train()
    losses = []
    for images, targets in loader:
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values()).item()
        losses.append(loss)
    return float(np.mean(losses)) if losses else float("nan")


def train_one_epoch(model, optimizer, loader, device, epoch, print_every=50):
    model.train()
    running = []
    t0 = time.time()

    for it, (images, targets) in enumerate(loader):
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running.append(loss.item())

        if (it + 1) % print_every == 0:
            dt = time.time() - t0
            print(f"epoch {epoch} iter {it+1}/{len(loader)} "
                  f"loss={np.mean(running[-print_every:]):.4f} time={dt:.1f}s")
            t0 = time.time()

    return float(np.mean(running)) if running else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_images", required=True)
    parser.add_argument("--val_csv", required=True)

    parser.add_argument("--num_classes", type=int, required=True)   # foreground classes
    parser.add_argument("--output_path", required=True)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)

    # from-scratch often wants smaller LR than finetune
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--resize-short", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=768)

    args = parser.parse_args()

    outdir = Path(args.output_path)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.ngpus > 0 else "cpu")
    print("torch:", torch.__version__, "| torchvision:", torchvision.__version__)
    print("device:", device)

    # Label map from TRAIN CSV only
    label_to_idx, idx_to_label = build_label_map_from_csv(args.train_csv, expected_classes=args.num_classes)
    with open(outdir / "label_map.json", "w") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, indent=2)
    print("Saved label map:", outdir / "label_map.json")

    train_ds = DetectionBBoxCSVDataset(args.train_images, args.train_csv, label_to_idx,
                                       resize_short=args.resize_short, max_size=args.max_size)
    val_ds   = DetectionBBoxCSVDataset(args.val_images, args.val_csv, label_to_idx,
                                       resize_short=args.resize_short, max_size=args.max_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=detection_collate, pin_memory=(device.type == "cuda")
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=detection_collate, pin_memory=(device.type == "cuda")
    )

    model = build_frcnn_random_init(num_classes=args.num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from {args.resume} (start_epoch={start_epoch}, best_val={best_val:.4f})")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_every=50)
        val_loss = evaluate_loss(model, val_loader, device)

        print(f"\nEPOCH {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}\n")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
            "args": vars(args),
        }
        torch.save(ckpt, outdir / "latest.pth.tar")

        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val"] = best_val
            torch.save(ckpt, outdir / "best.pth.tar")
            print(f"[BEST] saved best.pth.tar (val_loss={best_val:.4f})")

    print("Done.")


if __name__ == "__main__":
    main()