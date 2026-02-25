#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Left/Right binary segmentation for VLAD Lab pair stimuli.

- Flexible naming:
  images: pair_<MID>_<ID>.<ext>
  masks:  maskL_<MID>_<ID>.<ext>   or   maskR_<MID>_<ID>.<ext>

- <MID> can be any of:
  bc_gc,gc_bc,bs_gs,gs_bs,gs_gc,bs_bc,bc_bs,gc_gs   (configurable via --pair_mids)

- Train exactly like your original script (SGD, StepLR, checkpoints), but outputs masks.
- Mixed precision + GPU by default. Images/masks assumed already at target size (no resizing).
"""

import os, re, argparse, shutil, random
from glob import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser(description='Binary segmentation (left or right) for VLAD pair stimuli')

# Data roots (folders contain many files; we filter by exact patterns)
parser.add_argument('--pairs_dir', type=str, default='/zpool/vladlab/data_drive/geogaze_data/pairs',
                    help='Folder containing images named pair_<MID>_<ID>.<ext>')
parser.add_argument('--masks_dir', type=str, default='/zpool/vladlab/data_drive/geogaze_data/left_masks',
                    help='Folder containing masks named maskL_<MID>_<ID>.<ext> or maskR_<MID>_<ID>.<ext>')

# Naming controls
parser.add_argument('--mask_side', type=str, choices=['L','R'], default='L',
                    help='Which side mask to use for training: L (left) or R (right)')
parser.add_argument('--pair_mids', type=str, default='bc_gc',
                    help='Comma-separated list of middle tokens (MIDs) to include, e.g. "bc_gc,gc_bc,bs_gs"')

# Training / output (kept aligned with your original script)
parser.add_argument('-o', '--output_path', type=str, default='/zpool/vladlab/active_drive/omaltz/scripts/weights',
                    help='Where to store checkpoints')
parser.add_argument('--arch', default='mobilenetv3_small_050.lamb_in1k',
                    help='timm backbone name')
parser.add_argument('--epochs', default=30, type=int, help='total epochs')
parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size')
parser.add_argument('--workers', default=8, type=int, help='data loading workers')
parser.add_argument('--rand_seed', default=1, type=int, help='random seed')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint to resume')
parser.add_argument('--val_split', default=0.2, type=float, help='fraction for validation split')

# Optim & sched (same defaults)
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate (SGD)')
parser.add_argument('--step_size', default=10, type=int, help='epochs between LR step')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)

# Loss tweak
parser.add_argument('--pos_weight', default=0.0, type=float,
                    help='positive class weight for BCE (0 = disabled)')

# Freezing & AMP
parser.add_argument('--freeze_epochs', default=0, type=int,
                    help='epochs to freeze encoder (train head only). 0 disables.')
parser.add_argument('--threshold', default=0.5, type=float, help='prob threshold for metrics')
parser.add_argument('--no_amp', action='store_true', help='disable mixed precision')

args = parser.parse_args()

# -------------------------
# Setup
# -------------------------
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_path, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(args.rand_seed)

# -------------------------
# Naming / discovery
# -------------------------
# Build regex to match allowed MIDs
allowed_mids = [m.strip() for m in args.pair_mids.split(',') if m.strip()]
mid_union = '|'.join(re.escape(m) for m in allowed_mids)
PAIR_RE = re.compile(rf'^pair_({mid_union})_(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)$', re.IGNORECASE)

def _mask_candidates(mask_side, mid, id_):
    base = f'mask{mask_side}_{mid}_{id_}'
    return [f'{base}.{ext}' for ext in ('png','jpg','jpeg','bmp','tif','tiff',
                                        'PNG','JPG','JPEG','BMP','TIF','TIFF')]

def _find_mask(masks_dir, mask_side, mid, id_):
    for name in _mask_candidates(mask_side, mid, id_):
        p = os.path.join(masks_dir, name)
        if os.path.isfile(p):
            return p
    return None

def collect_items(pairs_dir, masks_dir, mask_side):
    """
    Returns list of (img_path, mask_path, mid, id_str)
    Only keeps items where BOTH image and corresponding mask exist.
    """
    items = []
    skipped = 0
    for fn in os.listdir(pairs_dir):
        m = PAIR_RE.match(fn)
        if not m:
            continue
        mid = m.group(1)
        id_ = m.group(2)
        img_path = os.path.join(pairs_dir, fn)
        mask_path = _find_mask(masks_dir, mask_side, mid, id_)
        if mask_path is None:
            skipped += 1
            continue
        items.append((img_path, mask_path, mid, id_))
    items.sort(key=lambda x: (x[2], int(x[3])))  # sort by MID, then numeric ID
    if skipped:
        print(f'[WARN] Skipped {skipped} image(s) with no matching mask{mask_side}.')
    print(f'[INFO] Found {len(items)} usable (image, mask{mask_side}) pairs across MIDs: {allowed_mids}')
    return items

# -------------------------
# Dataset
# -------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _open_rgb(path):
    return Image.open(path).convert('RGB')

def _img_to_tensor_norm(img_pil):
    arr = np.asarray(img_pil, dtype=np.float32) / 255.0  # HWC
    t = torch.from_numpy(arr).permute(2,0,1)             # CHW
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(3,1,1)
    return (t - mean) / std

def _open_mask_bin(path):
    m = Image.open(path).convert('L')
    m = np.asarray(m, dtype=np.uint8)
    m = (m > 0).astype(np.float32)
    return torch.from_numpy(m).unsqueeze(0)  # [1,H,W]

class PairMaskDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, mid, id_ = self.items[idx]
        img = _img_to_tensor_norm(_open_rgb(img_path))
        mask = _open_mask_bin(mask_path)
        return img, mask, f'{mid}_{id_}'

# -------------------------
# Model: encoder + 1-layer decoder
# -------------------------
class OneLayerDecoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, feat, out_hw):
        logits_low = self.proj(feat)  # [B,1,h,w]
        return F.interpolate(logits_low, size=out_hw, mode='bilinear', align_corners=False)

class EncoderHead(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, features_only=True, out_indices=(-1,))
        in_ch = self.encoder.feature_info[-1]['num_chs']
        self.head = OneLayerDecoder(in_ch)

    def forward(self, x):
        feat = self.encoder(x)[0]           # [B,C,h,w]
        H, W = x.shape[-2:]
        return self.head(feat, (H, W))      # [B,1,H,W]

# -------------------------
# Loss & Metrics
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
# Checkpointing (mirrors your cadence)
# -------------------------
def save_checkpoint(state, is_best, epoch, filename_prefix, out_dir):
    ckpt = os.path.join(out_dir, f'{filename_prefix}_checkpoint_{args.rand_seed}.pth.tar')
    torch.save(state, ckpt)
    if (epoch == 1) or (epoch % 5 == 0):
        shutil.copyfile(ckpt, os.path.join(out_dir, f'{filename_prefix}_{epoch}_{args.rand_seed}.pth.tar'))
    if is_best:
        shutil.copyfile(ckpt, os.path.join(out_dir, f'{filename_prefix}_best_{args.rand_seed}.pth.tar'))

# -------------------------
# Build dataset & split
# -------------------------
all_items = collect_items(args.pairs_dir, args.masks_dir, args.mask_side)
assert len(all_items) > 0, "No matching (image, mask) pairs found. Check --pair_mids, --mask_side, and folders."

rng = random.Random(args.rand_seed)
shuffled = all_items[:]
rng.shuffle(shuffled)

n_val = max(1, int(len(shuffled) * args.val_split))
val_items = sorted(shuffled[:n_val], key=lambda x: (x[2], int(x[3])))
train_items = sorted(shuffled[n_val:], key=lambda x: (x[2], int(x[3])))

print(f'[INFO] Train: {len(train_items)}  Val: {len(val_items)}')

train_ds = PairMaskDataset(train_items)
val_ds   = PairMaskDataset(val_items)

trainloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)
valloader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

# -------------------------
# Model / Optim / Sched / Loss
# -------------------------
model = EncoderHead(args.arch).to(device)

def set_encoder_trainable(m, trainable: bool):
    for p in m.encoder.parameters():
        p.requires_grad = trainable

if args.freeze_epochs > 0:
    set_encoder_trainable(model, False)
else:
    set_encoder_trainable(model, True)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)

if args.pos_weight > 0.0:
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
else:
    bce = nn.BCEWithLogitsLoss()

scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp) and device.type == 'cuda')

# -------------------------
# Resume
# -------------------------
start_epoch = 1
best_dice = -np.inf
if args.resume and os.path.isfile(args.resume):
    print(f"=> loading checkpoint '{args.resume}'")
    ck = torch.load(args.resume, map_location='cpu')
    start_epoch = ck.get('epoch', 1)
    best_dice = ck.get('best_metric', -np.inf)
    model.load_state_dict(ck['state_dict'])
    optimizer.load_state_dict(ck['optimizer'])
    print(f"=> loaded (epoch {start_epoch})")
elif args.resume:
    print(f"=> no checkpoint found at '{args.resume}'")

# -------------------------
# Train / Validate
# -------------------------
print('starting training...')
for epoch in range(start_epoch, args.epochs + 1):
    # Unfreeze encoder after freeze period
    if epoch == args.freeze_epochs + 1:
        set_encoder_trainable(model, True)

    # ----- Train -----
    model.train()
    train_loss_sum, n_train = 0.0, 0
    for imgs, masks, _ in trainloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(not args.no_amp) and device.type == 'cuda'):
            logits = model(imgs)         # [B,1,H,W]
            loss = bce(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item() * imgs.size(0)
        n_train += imgs.size(0)

    scheduler.step()
    train_loss = train_loss_sum / max(1, n_train)

    # ----- Validate -----
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
                probs = torch.sigmoid(logits)
                preds = (probs > args.threshold).float()

            val_loss_sum += loss.item() * imgs.size(0)
            n_val += imgs.size(0)
            dices.append(dice_coef(preds, masks).item())
            ious.append(iou_coef(preds, masks).item())

    val_loss = val_loss_sum / max(1, n_val)
    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_iou  = float(np.mean(ious))  if ious else 0.0

    print(f'Epoch: {epoch:03d}\tTrain Loss: {train_loss:.6f}\tVal Loss: {val_loss:.6f}\tDice: {mean_dice:.3f}\tIoU: {mean_iou:.3f}')

    # ----- Save -----
    model_tag = f'{args.arch}_mask{args.mask_side}_{"-".join(allowed_mids)}'
    is_best = mean_dice > best_dice
    best_dice = max(best_dice, mean_dice)

    save_checkpoint({
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_metric': best_dice,
        'optimizer': optimizer.state_dict(),
    }, is_best, epoch, model_tag, args.output_path)
