import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import CocoDetection
from sklearn.model_selection import KFold
import os
import argparse
import wandb
import numpy as np
from pycocotools import mask as coco_mask

from models.baseline import UNetFP16

# --- Collate function for COCO ---
def coco_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

# --- Mask/Target Parsing Utilities ---
def parse_segmentation_masks(targets, height, width, num_classes=80):
    batch_masks = []
    for anns in targets:
        mask = torch.zeros((num_classes, height, width), dtype=torch.float32)
        for ann in anns:
            if 'segmentation' in ann:
                category_id = ann['category_id'] - 1
                rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
                m = torch.tensor(coco_mask.decode(rle), dtype=torch.float32)
                m = m if m.ndim == 2 else m.any(dim=-1)
                mask[category_id] = torch.max(mask[category_id], m)
        batch_masks.append(mask)
    return torch.stack(batch_masks)

def parse_instance_masks(targets, height, width):
    batch_masks = []
    for anns in targets:
        mask = torch.zeros((len(anns), height, width), dtype=torch.float32)
        for i, ann in enumerate(anns):
            if 'segmentation' in ann:
                rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
                m = torch.tensor(coco_mask.decode(rle), dtype=torch.float32)
                m = m if m.ndim == 2 else m.any(dim=-1)
                mask[i] = m
        batch_masks.append(mask)
    return batch_masks  # variable-length list

def parse_detection_heatmap(targets, height, width, num_classes=80):
    batch_heatmaps = []
    for anns in targets:
        heatmap = torch.zeros((num_classes, height, width), dtype=torch.float32)
        for ann in anns:
            cat = ann['category_id'] - 1
            x, y, w, h = ann['bbox']
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            if 0 <= cx < width and 0 <= cy < height:
                heatmap[cat, cy, cx] = 1.0
        batch_heatmaps.append(heatmap)
    return torch.stack(batch_heatmaps)

# --- Telemetry Logging ---
def log_telemetry(model, loss_value, step):
    if not np.isfinite(loss_value):
        print(f"⚠️ Skipping telemetry logging due to non-finite loss at step {step}")
        return
    wandb.log({"loss": loss_value}, step=step)
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights = param.detach().cpu().numpy().flatten()
            grads = param.grad.detach().cpu().numpy().flatten() if param.grad is not None else None
            if np.isfinite(weights).all():
                wandb.log({f"weights/{name}": wandb.Histogram(weights)}, step=step)
            if grads is not None and np.isfinite(grads).all():
                wandb.log({f"grads/{name}": wandb.Histogram(grads)}, step=step)

# --- Epoch Training ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, step_start, task):
    model.train()
    running_loss = 0.0
    step = step_start
    for images, targets in dataloader:
        images = images.to(device, dtype=torch.float16)
        outputs = model(images)
        height, width = outputs.shape[2:]

        if task == "segmentation":
            gt = parse_segmentation_masks(targets, height, width).to(device, dtype=torch.float16)
        elif task == "detection":
            gt = parse_detection_heatmap(targets, height, width).to(device, dtype=torch.float16)
        elif task == "instance":
            print("Instance segmentation training is not supported in dense loss setting.")
            continue
        else:
            raise ValueError("Unsupported task.")

        optimizer.zero_grad()
        loss = criterion(outputs, gt)

        if not torch.isfinite(loss):
            print(f"⚠️ Skipping update due to non-finite loss at step {step}")
            continue

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        log_telemetry(model, loss.item(), step)
        step += 1
    return running_loss / len(dataloader), step

def validate(model, dataloader, criterion, device, step_start, task):
    model.eval()
    val_loss = 0.0
    step = step_start
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, dtype=torch.float16)
            outputs = model(images)
            height, width = outputs.shape[2:]

            if task == "segmentation":
                gt = parse_segmentation_masks(targets, height, width).to(device, dtype=torch.float16)
            elif task == "detection":
                gt = parse_detection_heatmap(targets, height, width).to(device, dtype=torch.float16)
            elif task == "instance":
                continue
            else:
                raise ValueError("Unsupported task.")

            loss = criterion(outputs, gt)

            if not torch.isfinite(loss):
                print(f"⚠️ Skipping validation step due to non-finite loss at step {step}")
                continue

            val_loss += loss.item()
            wandb.log({"val/loss": loss.item()}, step=step)
            step += 1
    return val_loss / len(dataloader), step

# --- Main Function ---
def main(args):
    wandb.init(project="unet-fp16-coco", config=vars(args))
    device = torch.device("cuda")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = CocoDetection(root=args.coco_root,
                            annFile=args.ann_file,
                            transform=transform)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/5")
        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=coco_collate_fn, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=coco_collate_fn, pin_memory=True)

        model = UNetFP16().to(device).half()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        step = 0
        for epoch in range(args.epochs):
            train_loss, step = train_one_epoch(model, train_loader, criterion, optimizer, device, step, args.task)
            val_loss, step = validate(model, val_loader, criterion, device, step, args.task)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "train/avg_loss": train_loss, "val/avg_loss": val_loss}, step=step)

        torch.save(model.state_dict(), f"unet_fold{fold + 1}.pth")

# --- Argument Parsing ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_root', type=str,  help='Path to COCO images folder', default="coco2017/images/train2017")
    parser.add_argument('--ann_file', type=str,  help='Path to COCO annotation file', default="coco2017/annotations/instances_train2017.json")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--task', type=str, choices=['segmentation', 'instance', 'detection'], default='segmentation')
    args = parser.parse_args()
    main(args)