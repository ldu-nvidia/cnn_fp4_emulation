# --- Imports ---
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import CocoDetection
from sklearn.model_selection import KFold
import os
import argparse
import wandb
import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import scipy.stats
import json
from config import configs
from dataclasses import asdict
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from models.baseline import UNetFP16 
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
    def __init__(self, config):
        self.layer_stats_w = []
        self.layer_stats_grad = []
        self.config = config

    def log_telemetry(self, model, loss_value, step):
        if step % self.config.logf != 0:
            return
        
        stat_dict_w, stat_dict_grad, stat_dict_a = {}, {}, {}
        if not np.isfinite(loss_value):
            print(f"⚠️ Skipping telemetry logging due to non-finite loss at step {step}")
            return
        wandb.log({"loss": loss_value}, step=step)

        for name, param in model.named_parameters():
            new_name = rename(name)
            if param.requires_grad and 'bias' not in new_name and 'weight' in new_name:
                w = param.detach().cpu().float().numpy().flatten()
                if np.isfinite(w).all():
                    stat_dict_w[new_name + "/mean"] = np.mean(w)
                    stat_dict_w[new_name + "/std"] = np.std(w)
                    if np.std(w) > 1e-6:
                        stat_dict_w[new_name + "/kurtosis"] = scipy.stats.kurtosis(w)
                    else:
                        stat_dict_w[new_name + "/kurtosis"] = 0.0  
                    wandb.log({new_name: wandb.Histogram(w)}, step=step)

            if param.requires_grad and param.grad is not None and 'bias' not in new_name and 'weight' in new_name:
                w = param.grad.detach().cpu().float().numpy().flatten()
                if np.isfinite(w).all():
                    stat_dict_grad[new_name + "/mean"] = np.mean(w)
                    stat_dict_grad[new_name + "/std"] = np.std(w)
                    if np.std(w) > 1e-6:
                        stat_dict_grad[new_name + "/kurtosis"] = scipy.stats.kurtosis(w)
                    else:
                        stat_dict_grad[new_name + "/kurtosis"] = 0.0
                    wandb.log({new_name: wandb.Histogram(w)}, step=step)

        stat_dict_w['step'], stat_dict_grad['step'] = step, step
        self.layer_stats_w.append(stat_dict_w)
        self.layer_stats_grad.append(stat_dict_grad)

    def save_layer_stats(self):
        output_paths=["weights_layer_stats.json", "grads_layer_stats.json"]
        os.makedirs("plots", exist_ok=True)
        for outpath, layer_stats in zip(output_paths, [self.layer_stats_w, self.layer_stats_grad]):
            with open("plots/" + outpath, "w") as f:
                serializable_stats = [
                    {k: float(v) if hasattr(v, 'item') and hasattr(v, 'dtype') else v for k, v in entry.items()}
                    for entry in layer_stats
                ]
                json.dump(serializable_stats, f)
                assert outpath in os.listdir('plots/')
        print("✅ Layer stats saved to plot folder")

    def finalize_and_visualize(self):
        assert len(self.layer_stats_w) != 0 and len(self.layer_stats_w) != 0, "weight or grad stats are empty"
        self.save_layer_stats()
        for layer_stats, type in zip([self.layer_stats_w, self.layer_stats_grad], ['weights', 'grads']):
            keys = sorted(k for k in layer_stats[0].keys() if k != 'step')
            steps = [stat['step'] for stat in layer_stats]
            tensor = np.zeros((len(steps), len(keys) // 3, 3))
            layer_names = []
            for i, k in enumerate(keys):
                base = k.rsplit('/', 1)[0]
                if base not in layer_names:
                    layer_names.append(base)
            stat_map = {'mean': 0, 'std': 1, 'kurtosis': 2}
            for i, entry in enumerate(layer_stats):
                for k, v in entry.items():
                    if k == 'step': continue
                    base, stat = k.rsplit('/', 1)
                    tensor[i, layer_names.index(base), stat_map[stat]] = v
            plot_grid_heatmaps(tensor, layer_names, list(stat_map.keys()), self.config, type)
            plot_interactive_3d(tensor, layer_names, list(stat_map.keys()), self.config, type)

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
                    gt = parse_segmentation_masks(targets, height, width, num_classes, self.category_id_to_class_idx).to(device)
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
            if self.config.enable_logging:
                self.log_telemetry(model, loss.item(), step)
            step += 1
            if step == 100:
                break
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
                        gt = parse_segmentation_masks(targets, height, width, num_classes, self.category_id_to_class_idx).to(device)
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
                    if step == step_start + 20:
                        break
        
        if self.config.enable_logging and self.config.visualize_val:
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

                gt_masks = parse_segmentation_masks(targets, height, width, num_classes, self.category_id_to_class_idx).argmax(1).cpu()

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
        if self.config.enable_logging: 
            wandb.init(project="unet-fp16-coco", name = "smaller model, new loss", config=asdict(self.config))
        device = torch.device("cuda:1")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        dataset = CocoDetection(root=self.config.coco_root,
                                annFile=self.config.ann_file,
                                transform=transform)
        
        coco = COCO(self.config.ann_file)
        category_ids = sorted(coco.getCatIds())  # just get all category IDs
        self.category_id_to_class_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
        num_classes = len(self.category_id_to_class_idx)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"Fold {fold + 1}/5")
            train_subset = Subset(dataset, train_idx.tolist())
            val_subset = Subset(dataset, val_idx.tolist())

            train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True, num_workers=4, collate_fn=coco_collate_fn, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False, num_workers=4, collate_fn=coco_collate_fn, pin_memory=True)

            print(f"[Fold {fold+1}] Dataset-wide num_classes = {num_classes}")

            model = UNetFP16(in_channels=3, out_channels=num_classes).to(device)
            print("size of the model: ", count_parameters(model))
            ce_criterion = nn.CrossEntropyLoss()
            dice_crition = DiceLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=1e-5)
            scaler = GradScaler(device='cuda')

            step = 0
            for epoch in range(self.config.epochs):
                train_loss, step = self.train_one_epoch(model, train_loader, ce_criterion, dice_crition, optimizer, device, step, self.config.task, num_classes, scaler)
                val_loss, step = self.validate(model, val_loader, ce_criterion, dice_crition, device, step, self.config.task, num_classes)
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                if self.config.enable_logging:
                    wandb.log({"epoch": epoch + 1, "train/avg_loss": train_loss, "val/avg_loss": val_loss}, step=step)
                if self.config.debug:
                    break

            torch.save(model.state_dict(), f"unet_fold{fold + 1}.pth")
            if self.config.debug:
                break

        self.finalize_and_visualize()

if __name__ == '__main__':
    config = configs()
    runner = reduced_precision_trainer(config)
    runner.run()
