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
from pycocotools import mask as coco_mask
from torch.amp import autocast, GradScaler
import scipy.stats
import json
from config import configs
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from models.baseline import UNetFP16  # <- make sure this supports out_channels
from visualize import plot_grid_heatmaps, plot_interactive_3d
from utils import rename, parse_segmentation_masks, parse_instance_masks, parse_detection_heatmap, coco_collate_fn, count_parameters, get_max_category_id, combined_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        assert logits.ndim == 4, f"logits must be 4D, got {logits.shape}"
        assert targets.ndim == 4, f"targets must be 4D (one-hot), got {targets.shape}"
        assert logits.shape == targets.shape, f"Shape mismatch: logits {logits.shape}, targets {targets.shape}"

        probs = F.softmax(logits, dim=1)
        targets = targets.float()

        intersection = torch.sum(probs * targets, dim=(2, 3))
        union = torch.sum(probs + targets, dim=(2, 3))

        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()

class reduced_precision_trainer():
    def __init__(self, args):
        self.layer_stats = []
        self.args = args

    def log_telemetry(self, model, loss_value, step):
        if not np.isfinite(loss_value):
            print(f"⚠️ Skipping telemetry logging due to non-finite loss at step {step}")
            return
        wandb.log({"loss": loss_value}, step=step)
        if step % self.args.logf != 0:
            return
        stat_dict = {}
        for name, param in model.named_parameters():
            new_name = rename(name)
            if param.requires_grad and 'bias' not in new_name and 'weight' in new_name:
                w = param.detach().cpu().float().numpy().flatten()
                if np.isfinite(w).all():
                    stat_dict[new_name + "/mean"] = np.mean(w)
                    stat_dict[new_name + "/std"] = np.std(w)
                    stat_dict[new_name + "/kurtosis"] = scipy.stats.kurtosis(w)
                    wandb.log({new_name: wandb.Histogram(w)}, step=step)
        stat_dict['step'] = step
        self.layer_stats.append(stat_dict)

    def save_layer_stats(self, output_path="layer_stats.json"):
        os.makedirs("plots", exist_ok=True)
        with open("plots/" + output_path, "w") as f:
            json.dump(self.layer_stats, f)
        print("✅ Layer stats saved to plot folder")

    def finalize_and_visualize(self):
        if not self.layer_stats:
            return

        self.save_layer_stats()
        keys = sorted(k for k in self.layer_stats[0].keys() if k != 'step')
        steps = [stat['step'] for stat in self.layer_stats]
        tensor = np.zeros((len(steps), len(keys) // 3, 3))
        layer_names = []
        for i, k in enumerate(keys):
            base = k.rsplit('/', 1)[0]
            if base not in layer_names:
                layer_names.append(base)
        stat_map = {'mean': 0, 'std': 1, 'kurtosis': 2}
        for i, entry in enumerate(self.layer_stats):
            for k, v in entry.items():
                if k == 'step': continue
                base, stat = k.rsplit('/', 1)
                tensor[i, layer_names.index(base), stat_map[stat]] = v
        plot_grid_heatmaps(tensor, layer_names, list(stat_map.keys()), self.args.logf)
        plot_interactive_3d(tensor, layer_names, list(stat_map.keys()))

    def train_one_epoch(self, model, dataloader, ce_criterion, dice_crition, optimizer, device, step_start, task, num_classes, scaler):
        model.train()
        running_loss = 0.0
        step = step_start
        for images, targets in dataloader:
            images = images.to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
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

                loss = combined_loss(outputs, gt, ce_criterion, dice_crition)

            if not torch.isfinite(loss):
                print(f"⚠️ Skipping update due to non-finite loss at step {step}")
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if self.args.enable_logging:
                self.log_telemetry(model, loss.item(), step)
            step += 1
        return running_loss / len(dataloader), step

    def validate(self, model, dataloader, ce_criterion, dice_crition, device, step_start, task, num_classes):
        model.eval()
        val_loss = 0.0
        step = step_start
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                with autocast(device_type='cuda', dtype=torch.float16):
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

                    loss = combined_loss(outputs, gt, ce_criterion, dice_crition)
                    if not torch.isfinite(loss):
                        print(f"⚠️ Skipping validation step due to non-finite loss at step {step}")
                        continue

                    val_loss += loss.item()
                    wandb.log({"val/loss": loss.item()}, step=step)
                    step += 1
        
        if self.args.enable_logging and self.args.visualize_val:
            self.log_visual_predictions(model, dataloader, num_samples=4, device=device, num_classes=num_classes)
        return val_loss / len(dataloader), step

    def log_visual_predictions(self, model, dataloader, num_samples, device, num_classes):
        model.eval()
        class_colors = plt.cm.get_cmap("tab20", num_classes)
        samples_logged = 0
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu()
                height, width = preds.shape[1:]

                gt_masks = parse_segmentation_masks(targets, height, width, num_classes).argmax(1).cpu()

                for i in range(min(len(images), num_samples - samples_logged)):
                    img_np = TF.to_pil_image(images[i].cpu())
                    pred_mask = preds[i].numpy()
                    gt_mask = gt_masks[i].numpy()

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(img_np)
                    axes[0].set_title("Input Image")
                    axes[1].imshow(img_np)
                    axes[1].imshow(pred_mask, alpha=0.5, cmap='jet')
                    axes[1].set_title("Predicted Mask")
                    axes[2].imshow(img_np)
                    axes[2].imshow(gt_mask, alpha=0.5, cmap='jet')
                    axes[2].set_title("Ground Truth Mask")
                    for ax in axes: ax.axis("off")
                    fig.tight_layout()
                    wandb.log({f"vis/sample_{samples_logged}": wandb.Image(fig)})
                    plt.close(fig)
                    samples_logged += 1
                    if samples_logged >= num_samples:
                        return

    def run(self):
        if self.args.enable_logging: 
            wandb.init(project="unet-fp16-coco", name = "smaller model, new loss", config=vars(self.args))
        device = torch.device("cuda:1")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        dataset = CocoDetection(root=self.args.coco_root,
                                annFile=self.args.ann_file,
                                transform=transform)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"Fold {fold + 1}/5")
            train_subset = Subset(dataset, train_idx.tolist())
            val_subset = Subset(dataset, val_idx.tolist())

            train_loader = DataLoader(train_subset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, collate_fn=coco_collate_fn, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=self.args.batch_size, shuffle=False, num_workers=4, collate_fn=coco_collate_fn, pin_memory=True)

            num_classes = get_max_category_id(dataset) + 1
            print(f"[Fold {fold+1}] Dataset-wide num_classes = {num_classes}")

            model = UNetFP16(in_channels=3, out_channels=num_classes).to(device)
            print("size of the model: ", count_parameters(model))
            ce_criterion = nn.CrossEntropyLoss()
            dice_crition = DiceLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-5)
            scaler = GradScaler(device='cuda')

            step = 0
            for epoch in range(self.args.epochs):
                train_loss, step = self.train_one_epoch(model, train_loader, ce_criterion, dice_crition, optimizer, device, step, self.args.task, num_classes, scaler)
                val_loss, step = self.validate(model, val_loader, ce_criterion, dice_crition, device, step, self.args.task, num_classes)
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                if self.args.enable_logging:
                    wandb.log({"epoch": epoch + 1, "train/avg_loss": train_loss, "val/avg_loss": val_loss}, step=step)
                if self.args.debug:
                    break

            torch.save(model.state_dict(), f"unet_fold{fold + 1}.pth")
            if self.args.debug:
                break

        self.finalize_and_visualize()

if __name__ == '__main__':
    args = configs()
    runner = reduced_precision_trainer(args)
    runner.run()
