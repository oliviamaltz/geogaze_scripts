#!/usr/bin/env python3
"""
B2: CORnet backbone (spatial feature map) + DETR-style set prediction head
    - Learned object queries (K)
    - Transformer decoder
    - Predict (C+1) class logits (includes NO_OBJECT)
    - Predict boxes in cx,cy,w,h normalized [0,1]
    - Hungarian matching + set loss (cls + L1 + GIoU)

CSV format expected:
  - image_file_name
  - category_name
  - bbox_x, bbox_y, bbox_w, bbox_h   (pixel coords in original image space)

Usage example:
  python train_cornet_detr_b2.py \
    --train_images /path/to/train_images \
    --train_csv /path/to/train.csv \
    --val_images /path/to/val_images \
    --val_csv /path/to/val.csv \
    --output_path /path/to/out \
    --num_classes 70 \
    --num_queries 10 \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --backbone_lr 1e-5 \
    --ngpus 1 \
    --model Z \
    --feature_layer IT

Notes:
- Requires: torch, torchvision, PIL
- Hungarian matching: tries scipy; if missing, uses a greedy fallback (works but worse).
"""

import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import csv
from PIL import Image

# -------------------------
# Optional Hungarian matching (scipy)
# -------------------------
def _try_import_scipy():
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment
    except Exception:
        return None

linear_sum_assignment = _try_import_scipy()


# -------------------------
# GPU selection (your style)
# -------------------------
def set_gpus(n=1):
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'
    ), check=True, stdout=subprocess.PIPE).stdout
    gpus = pandas.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    gpus = gpus[gpus['memory.total [MiB]'] > 10000]
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpus['index'].iloc[:n]])

# -------------------------
# Label map (deterministic)
# -------------------------
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
    label_to_idx = {lab: i for i, lab in enumerate(labels)}  # 0..C-1
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    return label_to_idx, idx_to_label

# -------------------------
# Box helpers
# -------------------------
def xyxy_to_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
    # xyxy: [...,4]
    x1, y1, x2, y2 = xyxy.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w  = (x2 - x1).clamp(min=1e-6)
    h  = (y2 - y1).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)

def cxcywh_to_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = cxcywh.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

def generalized_box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: [N,4], boxes2: [M,4] in xyxy, normalized 0..1
    returns: [N,M] GIoU
    """
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)                               # [N,M,2]
    inter = wh[..., 0] * wh[..., 1]                           # [N,M]

    area1 = box_area_xyxy(boxes1)[:, None]                    # [N,1]
    area2 = box_area_xyxy(boxes2)[None, :]                    # [1,M]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    # Enclosing box
    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]

    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
    return giou

# -------------------------
# Dataset (your CSV, but returns normalized boxes)
# -------------------------
class DetectionBBoxCSVDataset(torch.utils.data.Dataset):
    """
    Returns:
      image: Tensor [3,224,224] normalized for CORnet
      target: dict with:
        - boxes_xyxy: FloatTensor [N,4] normalized to 0..1
        - labels: LongTensor [N] in 0..C-1
    """
    def __init__(self, images_root, csv_path, label_to_idx, transform, max_objects=20):
        self.images_root = Path(images_root)
        self.csv_path = Path(csv_path)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.max_objects = max_objects

        img_to_anns = defaultdict(list)
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["image_file_name"]
                cat_name = row["category_name"].strip()
                if cat_name not in self.label_to_idx:
                    continue

                x = float(row["bbox_x"])
                y = float(row["bbox_y"])
                w = float(row["bbox_w"])
                h = float(row["bbox_h"])
                x1, y1, x2, y2 = x, y, x + w, y + h
                img_to_anns[img_name].append((self.label_to_idx[cat_name], [x1, y1, x2, y2]))

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

    def __getitem__(self, idx):
        img_path, ann_list = self.items[idx]
        im_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = im_pil.size

        # Optional cap: keep largest boxes by area
        if self.max_objects is not None and len(ann_list) > self.max_objects:
            ann_list = sorted(
                ann_list,
                key=lambda a: (a[1][2]-a[1][0])*(a[1][3]-a[1][1]),
                reverse=True
            )[: self.max_objects]

        # Convert to normalized xyxy w.r.t original size, then clamp to [0,1]
        labels = []
        boxes = []
        for lab, (x1, y1, x2, y2) in ann_list:
            x1n = max(0.0, min(1.0, x1 / orig_w))
            x2n = max(0.0, min(1.0, x2 / orig_w))
            y1n = max(0.0, min(1.0, y1 / orig_h))
            y2n = max(0.0, min(1.0, y2 / orig_h))
            if x2n <= x1n or y2n <= y1n:
                continue
            labels.append(lab)
            boxes.append([x1n, y1n, x2n, y2n])

        labels_t = torch.tensor(labels, dtype=torch.long)
        boxes_t = torch.tensor(boxes, dtype=torch.float32)

        # Image -> model tensor (224x224 + normalize) (does NOT change normalized boxes)
        img_t = self.transform(im_pil)

        return img_t, {"boxes_xyxy": boxes_t, "labels": labels_t, "image_path": str(img_path)}

def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# -------------------------
# 2D sine positional encoding (DETR-style)
# -------------------------
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        B, _, H, W = x.shape
        device = x.device

        y_embed = torch.linspace(0, 1, H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.linspace(0, 1, W, device=device).unsqueeze(0).repeat(H, 1)

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x), dim=2)  # [H,W,2*num_pos_feats]
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,2F,H,W]
        return pos

# -------------------------
# CORnet feature extractor (hook a spatial layer)
# -------------------------
class CornetSpatialBackbone(nn.Module):
    """
    Runs CORnet and captures a spatial feature map from a chosen layer (e.g., 'IT' or 'V4').
    We use a forward hook on that module.
    """
    def __init__(self, cornet_model: nn.Module, feature_layer: str = "IT"):
        super().__init__()
        self.model = cornet_model
        self.feature_layer = feature_layer
        self._feat = None

        # Find module
        # CORnet commonly has attributes V1, V2, V4, IT, decoder.
        if not hasattr(self.model, feature_layer):
            raise ValueError(f"CORnet model has no attribute '{feature_layer}'. Try one of: V1,V2,V4,IT")
        layer_mod = getattr(self.model, feature_layer)

        def hook_fn(module, inp, out):
            # out is expected to be [B,C,H,W]
            self._feat = out

        layer_mod.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.model(x)
        if self._feat is None:
            raise RuntimeError("Feature hook did not capture output. Check feature_layer.")
        return self._feat

# -------------------------
# DETR-lite head
# -------------------------
class CornetDETR(nn.Module):
    def __init__(
        self,
        cornet_base: nn.Module,
        num_classes: int,
        num_queries: int = 10,
        feature_layer: str = "IT",
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.no_object_class = num_classes  # index C (so total C+1)

        self.backbone = CornetSpatialBackbone(cornet_base, feature_layer=feature_layer)

        # We don't know backbone channel count until we run once. So we set proj lazily.
        self.proj = None
        self.d_model = d_model

        self.pos_embed = PositionEmbeddingSine(num_pos_feats=d_model // 2)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)

        # Heads: class + box
        self.class_head = nn.Linear(d_model, num_classes + 1)  # includes NO_OBJECT
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
        )

    def _ensure_proj(self, feat: torch.Tensor):
        if self.proj is None:
            in_ch = feat.shape[1]
            self.proj = nn.Conv2d(in_ch, self.d_model, kernel_size=1).to(feat.device)
            self.add_module("input_proj", self.proj)

    def forward(self, x: torch.Tensor):
        """
        Returns:
          pred_logits: [B,Q,C+1]
          pred_boxes:  [B,Q,4] in cxcywh normalized [0,1]
        """
        feat = self.backbone(x)              # [B,C,H,W]
        self._ensure_proj(feat)

        src = self.proj(feat)               # [B,d,H,W]
        pos = self.pos_embed(src)           # [B,d,H,W]

        B, d, H, W = src.shape
        # Flatten to tokens
        src_tokens = src.flatten(2).permute(0, 2, 1)  # [B,HW,d]
        pos_tokens = pos.flatten(2).permute(0, 2, 1)  # [B,HW,d]

        # Queries
        q = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B,Q,d]

        # Transformer decoder attends to src
        hs = self.decoder(tgt=q, memory=src_tokens + pos_tokens)  # [B,Q,d]

        logits = self.class_head(hs)                              # [B,Q,C+1]
        boxes = torch.sigmoid(self.box_head(hs))                  # [B,Q,4] cxcywh in 0..1
        return {"pred_logits": logits, "pred_boxes": boxes}

# -------------------------
# Matching + Loss (DETR style)
# -------------------------
def hungarian_matcher(pred_logits, pred_boxes, tgt_labels, tgt_boxes_xyxy,
                      cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
    """
    pred_logits: [Q,C+1] (raw logits)
    pred_boxes:  [Q,4] cxcywh (0..1)
    tgt_labels:  [N] in 0..C-1
    tgt_boxes_xyxy: [N,4] xyxy (0..1)

    returns:
      matched_pred_idx: [M]
      matched_tgt_idx:  [M]
    """
    Q, Cp1 = pred_logits.shape
    N = tgt_labels.shape[0]
    if N == 0:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)

    # classification cost: -P(class)
    prob = pred_logits.softmax(-1)  # [Q,C+1]
    cost_cls = -prob[:, tgt_labels] # [Q,N]

    # bbox cost: L1 between xyxy
    pred_xyxy = cxcywh_to_xyxy(pred_boxes).clamp(0, 1)  # [Q,4]
    cost_l1 = torch.cdist(pred_xyxy, tgt_boxes_xyxy, p=1)  # [Q,N]

    # giou cost
    giou = generalized_box_iou_xyxy(pred_xyxy, tgt_boxes_xyxy)  # [Q,N]
    cost_g = -giou

    C = cost_class * cost_cls + cost_bbox * cost_l1 + cost_giou * cost_g  # [Q,N]
    C_cpu = C.detach().cpu().numpy()

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(C_cpu)
        return torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)

    # Greedy fallback (not optimal, but works without scipy)
    # Iteratively pick best remaining pair
    C_work = C.clone()
    pred_used = torch.zeros(Q, dtype=torch.bool, device=C.device)
    tgt_used = torch.zeros(N, dtype=torch.bool, device=C.device)
    pairs = []
    for _ in range(min(Q, N)):
        # mask used
        C_masked = C_work.clone()
        C_masked[pred_used, :] = 1e9
        C_masked[:, tgt_used] = 1e9
        val, idx = C_masked.view(-1).min(0)
        if val.item() >= 1e8:
            break
        pi = idx // N
        ti = idx % N
        pred_used[pi] = True
        tgt_used[ti] = True
        pairs.append((pi.item(), ti.item()))
    if len(pairs) == 0:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    row_ind = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    col_ind = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    return row_ind, col_ind

class SetCriterion(nn.Module):
    def __init__(self, num_classes, no_object_weight=0.1,
                 cost_class=1.0, cost_bbox=5.0, cost_giou=2.0,
                 loss_bbox=5.0, loss_giou=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.no_object = num_classes  # index C
        self.no_object_weight = no_object_weight

        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou

        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou

        # class weights: downweight NO_OBJECT
        weight = torch.ones(num_classes + 1)
        weight[self.no_object] = no_object_weight
        self.register_buffer("ce_weight", weight)

    def forward(self, outputs, targets):
        """
        outputs: dict with pred_logits [B,Q,C+1], pred_boxes [B,Q,4] cxcywh
        targets: list of dict with boxes_xyxy [N,4], labels [N]
        """
        pred_logits = outputs["pred_logits"]
        pred_boxes  = outputs["pred_boxes"]
        B, Q, Cp1 = pred_logits.shape

        total_ce = 0.0
        total_l1 = 0.0
        total_giou = 0.0
        n_targets = 0

        for b in range(B):
            tgt_labels = targets[b]["labels"]
            tgt_xyxy   = targets[b]["boxes_xyxy"]
            n_targets += tgt_labels.shape[0]

            # match
            pi, ti = hungarian_matcher(
                pred_logits[b], pred_boxes[b],
                tgt_labels, tgt_xyxy,
                cost_class=self.cost_class, cost_bbox=self.cost_bbox, cost_giou=self.cost_giou
            )

            # --- Classification loss ---
            # initialize all queries as NO_OBJECT
            target_classes = torch.full((Q,), self.no_object, dtype=torch.long, device=pred_logits.device)
            if pi.numel() > 0:
                target_classes[pi.to(pred_logits.device)] = tgt_labels[ti.to(tgt_labels.device)].to(pred_logits.device)

            ce = F.cross_entropy(pred_logits[b], target_classes, weight=self.ce_weight)
            total_ce += ce

            # --- Box losses on matched only ---
            if pi.numel() > 0:
                p_boxes = pred_boxes[b, pi.to(pred_boxes.device)]                 # cxcywh
                p_xyxy  = cxcywh_to_xyxy(p_boxes).clamp(0, 1)
                t_xyxy  = tgt_xyxy[ti.to(tgt_xyxy.device)].to(pred_boxes.device)  # xyxy

                l1 = F.l1_loss(p_xyxy, t_xyxy, reduction="mean")
                giou = generalized_box_iou_xyxy(p_xyxy, t_xyxy).diag()
                giou_loss = (1.0 - giou).mean()

                total_l1 += l1
                total_giou += giou_loss

        # normalize by batch (and avoid div by 0)
        denom = max(B, 1)
        losses = {
            "loss_ce": total_ce / denom,
            "loss_bbox": total_l1 / denom,
            "loss_giou": total_giou / denom,
        }
        total = losses["loss_ce"] + self.loss_bbox * losses["loss_bbox"] + self.loss_giou * losses["loss_giou"]
        losses["loss_total"] = total
        losses["n_targets"] = n_targets
        return losses

# -------------------------
# Train / Val loops
# -------------------------
def run_epoch(model, criterion, loader, optimizer, device, train: bool, print_every=50):
    if train:
        model.train()
    else:
        model.eval()

    meters = {"loss_total": [], "loss_ce": [], "loss_bbox": [], "loss_giou": []}
    t0 = time.time()

    for it, (images, targets) in enumerate(loader):
        images = torch.stack(images, dim=0).to(device)

        # move target tensors
        t_list = []
        for t in targets:
            t_list.append({
                "boxes_xyxy": t["boxes_xyxy"].to(device),
                "labels": t["labels"].to(device),
                "image_path": t.get("image_path", ""),
            })

        with torch.set_grad_enabled(train):
            outputs = model(images)
            losses = criterion(outputs, t_list)

            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss_total"].backward()
                optimizer.step()

        for k in ["loss_total", "loss_ce", "loss_bbox", "loss_giou"]:
            meters[k].append(float(losses[k].detach().cpu()))

        if (it + 1) % print_every == 0:
            dt = time.time() - t0
            print(f"{'train' if train else 'val'} iter {it+1}/{len(loader)} "
                  f"loss={np.mean(meters['loss_total'][-print_every:]):.4f} "
                  f"(ce={np.mean(meters['loss_ce'][-print_every:]):.4f}, "
                  f"l1={np.mean(meters['loss_bbox'][-print_every:]):.4f}, "
                  f"giou={np.mean(meters['loss_giou'][-print_every:]):.4f}) "
                  f"time={dt:.1f}s")
            t0 = time.time()

    return {k: float(np.mean(v)) if len(v) else float("nan") for k, v in meters.items()}

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="CORnet + DETR-lite detector (B2)")
    parser.add_argument('--train_images', required=True)
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_images', required=True)
    parser.add_argument('--val_csv', required=True)

    parser.add_argument('--num_classes', required=True, type=int)
    parser.add_argument('--output_path', required=True)

    parser.add_argument('--model', choices=['Z', 'R', 'RT', 'S'], default='Z')
    parser.add_argument('--times', default=5, type=int)
    parser.add_argument('--feature_layer', default='IT', help="Which CORnet layer to tap: V1,V2,V4,IT")

    parser.add_argument('--num_queries', default=10, type=int)
    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--num_decoder_layers', default=4, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)

    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--backbone_lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    parser.add_argument('--ngpus', default=0, type=int)
    parser.add_argument('--resume', default=None)

    # loss weights
    parser.add_argument('--no_object_weight', default=0.1, type=float)
    parser.add_argument('--cost_class', default=1.0, type=float)
    parser.add_argument('--cost_bbox', default=5.0, type=float)
    parser.add_argument('--cost_giou', default=2.0, type=float)
    parser.add_argument('--loss_bbox', default=5.0, type=float)
    parser.add_argument('--loss_giou', default=2.0, type=float)

    args = parser.parse_args()

    if args.ngpus > 0:
        set_gpus(args.ngpus)

    device = torch.device("cuda" if torch.cuda.is_available() and args.ngpus > 0 else "cpu")
    print("device:", device)
    print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
    if linear_sum_assignment is None:
        print("NOTE: scipy not found; using greedy matching fallback. Install scipy for best results.")

    outdir = Path(args.output_path)
    outdir.mkdir(parents=True, exist_ok=True)

    # CORnet import path
    CORN_NET_REPO = Path("/zpool/vladlab/active_drive/omaltz/git_repos/CORnet")
    import sys
    sys.path.insert(0, str(CORN_NET_REPO))
    import cornet
    print("Imported cornet from:", cornet.__file__)

    # label map from TRAIN csv
    label_to_idx, idx_to_label = build_label_map_from_csv(args.train_csv, expected_classes=args.num_classes)
    with open(outdir / "label_map.json", "w") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, indent=2)
    print("Saved label map:", outdir / "label_map.json")

    # image transform (CORnet-style)
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    train_ds = DetectionBBoxCSVDataset(args.train_images, args.train_csv, label_to_idx, transform, max_objects=50)
    val_ds   = DetectionBBoxCSVDataset(args.val_images, args.val_csv, label_to_idx, transform, max_objects=50)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate
    )

    # build CORnet base
    model_letter = args.model.lower()
    base_ctor = getattr(cornet, f'cornet_{model_letter}')
    if model_letter == 'r':
        cornet_base = base_ctor(pretrained=False, map_location=device, times=args.times)
    else:
        cornet_base = base_ctor(pretrained=False, map_location=device)
    # unwrap DataParallel if needed
    if hasattr(cornet_base, "module"):
        cornet_base = cornet_base.module

    print("cornet_base type:", type(cornet_base))
    print("top-level has:", {k: hasattr(cornet_base, k) for k in ["V1","V2","V4","IT","decoder"]})

    # show close matches (helpful if names differ)
    print("some attrs:", [a for a in dir(cornet_base) if a.lower() in ["v1","v2","v4","it","decoder"]])

    # DETR model
    model = CornetDETR(
        cornet_base=cornet_base,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        feature_layer=args.feature_layer,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
    ).to(device)

    criterion = SetCriterion(
        num_classes=args.num_classes,
        no_object_weight=args.no_object_weight,
        cost_class=args.cost_class, cost_bbox=args.cost_bbox, cost_giou=args.cost_giou,
        loss_bbox=args.loss_bbox, loss_giou=args.loss_giou,
    ).to(device)

    # optimizer with separate LR for backbone vs head (standard DETR trick)
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay
    )

    start_epoch = 0
    best_val = float("inf")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_val={best_val:.4f})")

    for epoch in range(start_epoch, args.epochs):
        print("\n" + "="*100)
        print(f"EPOCH {epoch}/{args.epochs-1}")

        train_stats = run_epoch(model, criterion, train_loader, optimizer, device, train=True, print_every=50)
        val_stats   = run_epoch(model, criterion, val_loader, optimizer, device, train=False, print_every=50)

        print(f"train: {train_stats}")
        print(f"val:   {val_stats}")

        ckpt = {
            "epoch": epoch,
            "best_val": best_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, outdir / "latest.pth.tar")

        if val_stats["loss_total"] < best_val:
            best_val = val_stats["loss_total"]
            ckpt["best_val"] = best_val
            torch.save(ckpt, outdir / "best.pth.tar")
            print(f"[BEST] Saved best.pth.tar (val_loss_total={best_val:.4f})")

    print("Done.")

if __name__ == "__main__":
    main()