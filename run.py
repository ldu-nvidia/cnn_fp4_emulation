# --- Imports ---
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
from torchvision.transforms.functional import resize, InterpolationMode
from pycocotools import mask as coco_mask
from torch.cuda.amp import autocast, GradScaler

from models.baseline import UNetFP16  # <- make sure this supports out_channels

# --- Collate function for COCO ---
def coco_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

# --- Mask/Target Parsing Utilities ---
def parse_segmentation_masks(targets, height, width, num_classes):
    batch_masks = []
    for anns in targets:
        mask = torch.zeros((num_classes, height, width), dtype=torch.float32)
        for ann in anns:
            if 'segmentation' in ann:
                category_id = ann['category_id']
                if category_id >= num_classes:
                    continue
                rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
                m = torch.tensor(coco_mask.decode(rle), dtype=torch.float32)
                if m.ndim == 3:
                    m = m.any(dim=-1)
                m = m.unsqueeze(0)
                m = resize(m, size=(height, width), interpolation=InterpolationMode.NEAREST)
                m = m.squeeze(0)
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
    return batch_masks

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
def train_one_epoch(model, dataloader, criterion, optimizer, device, step_start, task, num_classes, scaler):
    model.train()
    running_loss = 0.0
    step = step_start
    for images, targets in dataloader:
        images = images.to(device)
        print("this is step: ", step + 1)
        #print("this is input image batch: ", images)

        with autocast(dtype=torch.float16):
            outputs = model(images)
            #print("this is model output: ", outputs)
            height, width = outputs.shape[2:]

            if task == "segmentation":
                gt = parse_segmentation_masks(targets, height, width, num_classes).to(device)
            elif task == "detection":
                gt = parse_detection_heatmap(targets, height, width).to(device)
            elif task == "instance":
                print("Instance segmentation training is not supported in dense loss setting.")
                continue
            else:
                raise ValueError("Unsupported task.")

            if outputs.shape[1] != gt.shape[1]:
                raise ValueError(f"Mismatch: model output channels = {outputs.shape[1]}, target = {gt.shape[1]}")

            loss = criterion(outputs, gt)
            print("this is calculated loss: ", loss)

        if not torch.isfinite(loss):
            print(f"⚠️ Skipping update due to non-finite loss at step {step}")
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        log_telemetry(model, loss.item(), step)
        step += 1
    return running_loss / len(dataloader), step

def validate(model, dataloader, criterion, device, step_start, task, num_classes):
    model.eval()
    val_loss = 0.0
    step = step_start
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            with autocast(dtype=torch.float16):
                outputs = model(images)
                height, width = outputs.shape[2:]

                if task == "segmentation":
                    gt = parse_segmentation_masks(targets, height, width, num_classes).to(device)
                elif task == "detection":
                    gt = parse_detection_heatmap(targets, height, width).to(device)
                elif task == "instance":
                    continue
                else:
                    raise ValueError("Unsupported task.")

                if outputs.shape[1] != gt.shape[1]:
                    raise ValueError(f"Mismatch: model output channels = {outputs.shape[1]}, target = {gt.shape[1]}")

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
    wandb.init(project="unet-fp16-coco", name = "trial: getting right logging", config=vars(args))
    device = torch.device("cuda:1")
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

        def get_max_category_id(dataset, sample_size=1000):
            max_cat = 0
            for i in range(min(len(dataset), sample_size)):
                _, anns = dataset[i]
                if anns:
                    max_cat = max(max_cat, max(ann['category_id'] for ann in anns))
            return max_cat + 1

        num_classes = get_max_category_id(dataset)
        print(f"[Fold {fold+1}] Dataset-wide num_classes = {num_classes}")

        model = UNetFP16(in_channels=3, out_channels=num_classes).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler = GradScaler()

        step = 0
        for epoch in range(args.epochs):
            train_loss, step = train_one_epoch(model, train_loader, criterion, optimizer, device, step, args.task, num_classes, scaler)
            val_loss, step = validate(model, val_loader, criterion, device, step, args.task, num_classes)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "train/avg_loss": train_loss, "val/avg_loss": val_loss}, step=step)

        torch.save(model.state_dict(), f"unet_fold{fold + 1}.pth")

# --- Argument Parsing ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_root', type=str, help='Path to COCO images folder', default="coco2017/images/train2017")
    parser.add_argument('--ann_file', type=str, help='Path to COCO annotation file', default="coco2017/annotations/instances_train2017.json")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--task', type=str, choices=['segmentation', 'instance', 'detection'], default='segmentation')
    args = parser.parse_args()
    main(args)
