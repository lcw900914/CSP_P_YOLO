"""
CSPPartial-YOLO 訓練腳本
論文設定：
  - Optimizer: SGD, momentum=0.9, weight_decay=0.0005
  - LR: 0.006, CosineDecay 300 epochs + LinearWarmup 10 epochs
  - Batch: 6, GPU: 1
"""

import os
import sys
import time
import argparse
import math
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.csp_partial_yolo import CSPPartialYOLO
from datasets.dota_dataset import build_dataloader
from eval import evaluate


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def cosine_lr(epoch: int, total_epochs: int,
              base_lr: float, warmup_epochs: int = 10,
              min_lr: float = 1e-5) -> float:
    """CosineDecay with LinearWarmup"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    total_cls  = 0.0
    total_reg  = 0.0
    total_dfl  = 0.0
    n_batches  = 0

    for i, (imgs, gt_bboxes, gt_labels, _) in enumerate(loader):
        imgs = imgs.to(device)
        gt_bboxes = [b.to(device) for b in gt_bboxes]
        gt_labels = [l.to(device) for l in gt_labels]

        optimizer.zero_grad()
        with autocast():
            losses = model(imgs, gt_bboxes, gt_labels)
        loss = losses['loss']

        scaler.scale(loss).backward()
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_cls  += losses['loss_cls'].item()
        total_reg  += losses['loss_reg'].item()
        total_dfl  += losses['loss_dfl'].item()
        n_batches  += 1

        if (i + 1) % 50 == 0:
            print(f'  [Epoch {epoch+1}] iter {i+1}/{len(loader)} '
                  f'loss={loss.item():.4f} '
                  f'(cls={losses["loss_cls"].item():.3f} '
                  f'reg={losses["loss_reg"].item():.3f} '
                  f'dfl={losses["loss_dfl"].item():.3f})')

    n = max(n_batches, 1)
    return {
        'loss': total_loss / n,
        'cls':  total_cls  / n,
        'reg':  total_reg  / n,
        'dfl':  total_dfl  / n,
    }


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'loss':       loss,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='D:/cspyolo/data/dota/train',
                        help='前處理後的訓練資料目錄')
    parser.add_argument('--val_dir',   default='D:/cspyolo/data/dota/val',
                        help='前處理後的驗證資料目錄')
    parser.add_argument('--output',    default='D:/cspyolo/checkpoints')
    parser.add_argument('--epochs',    type=int,   default=300)
    parser.add_argument('--batch',     type=int,   default=6)
    parser.add_argument('--lr',        type=float, default=0.006)
    parser.add_argument('--warmup',    type=int,   default=10)
    parser.add_argument('--workers',   type=int,   default=4)
    parser.add_argument('--resume',    default='',  help='resume from checkpoint')
    parser.add_argument('--use_ca',    action='store_true', default=True,
                        help='Exp4=True(default), Exp2=False')
    parser.add_argument('--val_freq',  type=int,   default=10,
                        help='每幾個 epoch 計算一次 val mAP（0=停用）')
    parser.add_argument('--score_thr', type=float, default=0.05)
    parser.add_argument('--nms_thr',   type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── 模型 ──────────────────────────────────────────────────
    model = CSPPartialYOLO(num_classes=4).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model params: {total_params:.2f}M')

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
    )
    scaler = GradScaler()

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')

    # ── DataLoader ────────────────────────────────────────────
    train_loader = build_dataloader(
        args.train_dir, batch_size=args.batch,
        augment=True,  num_workers=args.workers,
    )
    val_loader = None
    if args.val_freq > 0 and args.val_dir:
        val_loader = build_dataloader(
            args.val_dir, batch_size=args.batch,
            augment=False, num_workers=args.workers,
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss  = float('inf')
    best_mAP   = 0.0
    log_path   = out_dir / 'train_log.txt'

    # ── 訓練迴圈 ──────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        # 更新學習率
        lr = cosine_lr(epoch, args.epochs, args.lr, args.warmup)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        t0 = time.time()
        metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch
        )
        elapsed = time.time() - t0

        log_line = (
            f'Epoch [{epoch+1:03d}/{args.epochs}] '
            f'lr={lr:.6f} '
            f'loss={metrics["loss"]:.4f} '
            f'(cls={metrics["cls"]:.3f} '
            f'reg={metrics["reg"]:.3f} '
            f'dfl={metrics["dfl"]:.3f}) '
            f'time={elapsed:.1f}s'
        )
        print(log_line)
        with open(log_path, 'a') as f:
            f.write(log_line + '\n')

        # 每 10 epoch 存一次
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, metrics['loss'],
                out_dir / f'epoch_{epoch+1:03d}.pt'
            )

        # 存最佳（loss）
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            save_checkpoint(
                model, optimizer, epoch, metrics['loss'],
                out_dir / 'best_model.pt'
            )

        # ── 週期性 val mAP ────────────────────────────────────
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            print(f'  [Val mAP] Epoch {epoch+1} ...')
            val_results = evaluate(model, val_loader, device,
                                   args.score_thr, args.nms_thr)
            mAP = val_results['mAP']
            map_line = (f'  [Val mAP] Epoch {epoch+1}: mAP@0.5={mAP*100:.2f}%  '
                        + '  '.join(f'{n}={v*100:.1f}%'
                                    for n, v in val_results['AP_per_class'].items()))
            print(map_line)
            with open(log_path, 'a') as f:
                f.write(map_line + '\n')

            if mAP > best_mAP:
                best_mAP = mAP
                save_checkpoint(
                    model, optimizer, epoch, metrics['loss'],
                    out_dir / 'best_model_map.pt'
                )
                print(f'  [Val mAP] New best mAP={best_mAP*100:.2f}%, saved best_model_map.pt')
            model.train()

    print(f'Training done. Best loss: {best_loss:.4f}  Best mAP: {best_mAP*100:.2f}%')


if __name__ == '__main__':
    main()
