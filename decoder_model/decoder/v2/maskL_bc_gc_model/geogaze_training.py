#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLAD Lab pair stimuli — decoder-only binary segmentation .

Assumes:
  images: pair_<MID>_<ID>.png           under PAIRS_DIR
  masks : mask(L|R)_<MID>_<ID>.png      under MASKS_DIR
"""

import os, re, random, math
import numpy as np
from PIL import Image
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

# -------------------------
# HARD-CODED PATHS (
# -------------------------
PAIRS_DIR  = '/zpool/vladlab/data_drive/geogaze_data/pairs'
MASKS_DIR  = '/zpool/vladlab/data_drive/geogaze_data/left_masks'
OUTPUT_DIR = '/zpool/vladlab/active_drive/omaltz/scripts/geogaze/weights_bc_gc_left2'

# -------------------------
# CLI (kept for training knobs only)
# -------------------------
parser = argparse.ArgumentParser("Decoder-only segmentation (PNG-only)")
parser.add_argument('--mask_side',  type=str, choices=['L','R'], default='L')
parser.add_argument('--pair_mids',  type=str, default='bc_gc', help='CSV of MIDs to include')

parser.add_argument('--arch',   type=str, default='mobilenetv3_small_050.lamb_in1k')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('-b','--batch_size', type=int, default=32)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--val_split', type=float, default=0.2)
parser.add_argument('--rand_seed', type=int, default=1)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--step_size', type=int, default=10)

parser.add_argument('--pos_weight', type=float, default=0.0, help='BCE pos weight (0=off)')
parser.add_argument('--threshold',  type=float, default=0.5)
parser.add_argument('--no_amp', action='store_true')
args = parser.parse_args()
if args.mask_side == 'L':
    MASKS_DIR = '/zpool/vladlab/data_drive/geogaze_data/left_masks'
else:
    MASKS_DIR = '/zpool/vladlab/data_drive/geogaze_data/right_masks'


# -------------------------
# Setup
# -------------------------
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(args.rand_seed)

# -------------------------
# File discovery (PNG-only)
# -------------------------
allowed_mids = [m.strip() for m in args.pair_mids.split(',') if m.strip()]
mid_union = '|'.join(re.escape(m) for m in allowed_mids)
PAIR_RE = re.compile(rf'^pair_({mid_union})_(\d+)\.png$')

def mask_path(side, mid, id_):
    p = os.path.join(MASKS_DIR, f'mask{side}_{mid}_{id_}.png')
    return p if os.path.isfile(p) else None



def collect_items(side):
    items, skipped = [], 0
    for fn in os.listdir(PAIRS_DIR):
        m = PAIR_RE.match(fn)
        if not m: continue
        mid, id_ = m.group(1), m.group(2)
        img_p = os.path.join(PAIRS_DIR, fn)
        msk_p = mask_path(side, mid, id_)
        if msk_p is None:
            skipped += 1
            continue
        items.append((img_p, msk_p, mid, id_))
    items.sort(key=lambda x: (x[2], int(x[3])))
    if skipped: print(f'[WARN] Skipped {skipped} image(s) with no matching mask{side}.')
    print(f'[INFO] Found {len(items)} (image, mask{side}) pairs across MIDs: {allowed_mids}')
    return items

# -------------------------
# Dataset (ImageNet norm, binary masks)
# -------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def load_img(path):
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2,0,1)
    return (t - IMAGENET_MEAN) / IMAGENET_STD

def load_mask(path):
    m = np.asarray(Image.open(path).convert('L'), dtype=np.uint8)
    m = (m == 0).astype(np.float32)   
    return torch.from_numpy(m).unsqueeze(0)


class PairMaskDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        img_p, msk_p, mid, id_ = self.items[i]
        return load_img(img_p), load_mask(msk_p), f'{mid}_{id_}'

# -------------------------
# Model (timm encoder + 1x1 decoder), decoder-only training
# -------------------------
class OneLayerDecoder(nn.Module):
    def __init__(self, in_ch): super().__init__(); self.proj = nn.Conv2d(in_ch,1,1)
    def forward(self, feat, out_hw):
        logits = self.proj(feat)
        return F.interpolate(logits, size=out_hw, mode='bilinear', align_corners=False)

class EncoderHead(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, features_only=True, out_indices=(-1,))
        in_ch = self.encoder.feature_info[-1]['num_chs']
        self.head = OneLayerDecoder(in_ch)
    def forward(self, x):
        feat = self.encoder(x)[0]
        H, W = x.shape[-2:]
        return self.head(feat, (H, W))

def freeze_encoder(model):
    for p in model.encoder.parameters(): p.requires_grad = False
    model.encoder.eval()

# -------------------------
# Metrics / Loss
# -------------------------
def dice_coef(pred_bin, target, eps=1e-6):
    inter = (pred_bin * target).sum(dim=(1,2,3))
    union = pred_bin.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2*inter + eps) / (union + eps)).mean()

def iou_coef(pred_bin, target, eps=1e-6):
    inter = (pred_bin * target).sum(dim=(1,2,3))
    union = pred_bin.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
    return ((inter + eps) / (union + eps)).mean()

# -------------------------
# Data
# -------------------------
all_items = collect_items(args.mask_side)
assert len(all_items) > 0, "No (image, mask) pairs found."

rng = random.Random(args.rand_seed)
shuf = all_items[:]; rng.shuffle(shuf)
n_val = max(1, int(len(shuf) * args.val_split))
val_items = sorted(shuf[:n_val], key=lambda x: (x[2], int(x[3])))
trn_items = sorted(shuf[n_val:], key=lambda x: (x[2], int(x[3])))

print(f'[INFO] Train: {len(trn_items)}  Val: {len(val_items)}')

trainloader = DataLoader(PairMaskDataset(trn_items), batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)
valloader   = DataLoader(PairMaskDataset(val_items), batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

# -------------------------
# Build model, optimizer, scheduler, loss
# -------------------------
model = EncoderHead(args.arch).to(device)

# Freeze encoder permanently (params + BN stats)
freeze_encoder(model)

# Initialize decoder to a small foreground prior (5%)
nn.init.zeros_(model.head.proj.weight)
nn.init.constant_(model.head.proj.bias, math.log(0.05 / 0.95))

# Optimizer updates ONLY the decoder
optimizer = torch.optim.SGD(model.head.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)

bce = nn.BCEWithLogitsLoss(
    pos_weight=(torch.tensor(args.pos_weight, device=device) if args.pos_weight > 0 else None)
)

scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp) and device.type == 'cuda')

# -------------------------
# Train / Validate
# -------------------------
best_dice = -float('inf')
print('starting training...')
for epoch in range(1, args.epochs + 1):
    # ---- Train ----
    model.train()
    model.encoder.eval()  # keep BN frozen
    train_loss_sum, n_train = 0.0, 0

    for imgs, masks, _ in trainloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(not args.no_amp) and device.type == 'cuda'):
            logits = model(imgs)
            loss = bce(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item() * imgs.size(0)
        n_train += imgs.size(0)

    scheduler.step()
    train_loss = train_loss_sum / max(1, n_train)

    # ---- Validate ----
    model.eval()
    val_loss_sum, n_val = 0.0, 0
    dices, ious = [], []
    with torch.no_grad():
        for imgs, masks, _ in valloader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(not args.no_amp) and device.type == 'cuda'):
                logits = model(imgs)
                loss = bce(logits, masks)
                preds = (torch.sigmoid(logits) > args.threshold).float()
            val_loss_sum += loss.item() * imgs.size(0)
            n_val += imgs.size(0)
            dices.append(dice_coef(preds, masks).item())
            ious.append(iou_coef(preds, masks).item())

    val_loss = val_loss_sum / max(1, n_val)
    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_iou  = float(np.mean(ious))  if ious else 0.0

    print(f'Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_loss:.6f} | Dice {mean_dice:.3f} | IoU {mean_iou:.3f}')

    # ---- Save best ----
    if mean_dice > best_dice:
        best_dice = mean_dice
        tag = f'{args.arch}_mask{args.mask_side}_{"-".join(allowed_mids)}'
        path = os.path.join(OUTPUT_DIR, f'{tag}_best.pth')
        torch.save({'epoch': epoch, 'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_dice': best_dice}, path)
