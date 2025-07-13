# --- Imports ---
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import CocoDetection
from sklearn.model_selection import KFold
import os, random
import argparse
import wandb
import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import scipy.stats
import json
from data import CocoSegmentationDataset
from config import configs
from dataclasses import asdict
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.baseline import UNetFP16 
from utils import rename, parse_segmentation_masks, parse_instance_masks, parse_detection_heatmap, \
    coco_collate_fn, count_parameters, get_max_category_id, combined_loss, visualize_instance_batch, \
    get_max_instances_from_annotations, log_visual_predictions_to_file, check_mask, save_visual_predictions, \
        get_coco, get_max_instances_from_annotations, plot_grid_heatmaps, plot_interactive_3d
from loss import DiceLoss, instance_loss

SEED = 8        # pick any integer you like ────────────────┐
torch.manual_seed(SEED)       # <- torch CPU ops                │
random.seed(SEED)             # <- python.random                │
np.random.seed(SEED)          # <- NumPy                        │
torch.cuda.manual_seed(SEED)  # <- torch GPU ops (optional)     │
torch.backends.cudnn.deterministic = True   # reproducible conv │
torch.backends.cudnn.benchmark = False      # (may slow a bit)  │
# ───────────────────────────────────────────────────────────────┘

# 1.  torch.Generator to control the *shuffling* RNG
g = torch.Generator()
g.manual_seed(SEED)

# 2. worker_init_fn so *each* worker gets a deterministic, unique seed
def seed_worker(worker_id: int):
    # base_seed is the same for every worker but worker_id makes it unique
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
        directory = "plots/heatmaps/"
        os.makedirs(directory, exist_ok=True)
        for outpath, layer_stats in zip(output_paths, [self.layer_stats_w, self.layer_stats_grad]):
            with open(directory + outpath, "w") as f:
                serializable_stats = [
                    {k: float(v) if hasattr(v, 'item') and hasattr(v, 'dtype') else v for k, v in entry.items()}
                    for entry in layer_stats
                ]
                json.dump(serializable_stats, f)
                assert outpath in os.listdir(directory)
        print("✅ Layer stats saved to plot folder")

    def finalize_and_visualize(self):
        print("\n finalizing and visualizing \n")
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
            print("✅ Plots saved to plot folder")
        print("\n task finished \n")

    def train_one_epoch(
            self, model, dataloader, ce_criterion, dice_crition,
            optimizer, device, step_start, task, num_classes, scaler
    ):
        model.train()
        running_loss = 0.0
        step = step_start
        visualized = False                     # ← visualize only once

        for images, targets in dataloader:
            images = images.to(device)

            # ─── Forward pass (AMP) ─────────────────────────────────────────────────
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                h, w = outputs.shape[2:]

                if task == "semantic":
                    gt = parse_segmentation_masks(
                        targets, h, w, num_classes, self.category_id_to_class_idx
                    ).to(device)
                    loss = combined_loss(outputs, gt, ce_criterion, dice_crition)
                    assert (
                        gt.shape[-2:] == outputs.shape[-2:]
                        and gt.shape[1] == num_classes
                    ), "GT tensor shape mismatch with outputs or num_classes for semantic task"

                elif task == "detection":
                    gt = parse_detection_heatmap(targets, h, w).to(device)
                    loss = torch.zeros([], device=device)


                elif task == "instance":
                    batch_masks, _ = parse_instance_masks(
                        targets, h, w, self.category_id_to_class_idx
                    )
                    batch_masks = [m.to(device) for m in batch_masks]
                    loss = instance_loss(outputs, batch_masks)
                    assert (all(m.shape[-2:] == outputs.shape[-2:] for m in batch_masks)), "GT tensor shape mismatch with outputs for instance task"

                else:
                    raise ValueError("Unsupported task")

            # ─── Back-prop & optimiser step ───────────────────────────────────────
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=device)

            if not torch.isfinite(loss):
                print(f"⚠️  skipping non-finite loss at step {step}")
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # ─── One-off visual check after first step ────────────────────────────
            if not visualized:
                print("\n📸  Visual sanity-check after first training step\n")
                preds = torch.argmax(outputs, dim=1).cpu()
                directory = "plots/start_of_training_check/"
                os.makedirs(directory, exist_ok=True)
                save_visual_predictions(
                    images=images.cpu(),
                    targets=targets,
                    preds=preds,
                    config=self.config,
                    task=task,
                    num_classes=num_classes,
                    category_id_to_class_idx=self.category_id_to_class_idx,
                    save_prefix=directory
                )
                visualized = True                 # ← flip flag so we don’t repeat

            # ─── Book-keeping ─────────────────────────────────────────────────────
            running_loss += loss.item()
            if self.config.enable_logging:
                self.log_telemetry(model, loss.item(), step)
            step += 1

            if step == 100:                       # your early-break for debugging
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

                    if task == "semantic":
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
            assert self.config.task in ["semantic", "instance"], \
                f"Unsupported task type: {self.config.task}"
            print("\n start validating data after validation step \n")
            log_visual_predictions_to_file(
                self.config,
                model,
                dataloader,
                device=device,
                num_classes=num_classes,
                category_id_to_class_idx=self.category_id_to_class_idx
            )

        # Ensure validate always returns a tuple (val_loss, step)
        return val_loss / max(len(dataloader), 1), step

    def run(self):
        print("start initial mask check, before training")
        check_mask(self.config.seeds)

        if self.config.enable_logging: 
            wandb.init(project="unet-fp16-coco", name = "smaller model, new loss", config=asdict(self.config))
        device = torch.device("cuda:1")

        # get category to class index mapping once as global variable
        coco = get_coco(self.config.ann_file)               # loads once
        category_ids = sorted(coco.getCatIds())
        self.category_id_to_class_idx = {cat: i for i, cat in enumerate(category_ids)}

        dataset = CocoSegmentationDataset(
            img_root=self.config.coco_root,
            ann_file=self.config.ann_file,
            category_id_to_class_idx=self.category_id_to_class_idx,
            target_size=(256, 256)
        )
        
        # for semantic segmentation, we need to know the number of classes, and no num_instances
        if self.config.task == "semantic":
            num_classes = len(self.category_id_to_class_idx)
            num_instances = -1
        
        # for instance segmentation, we need to know the number of instances, and no num_classes
        if self.config.task == "instance":
            num_instances = get_max_instances_from_annotations(self.config.ann_file)
            print(f"Max number of instances per image = {num_instances}")
            num_classes = len(self.category_id_to_class_idx)  # ✅ safer, consistent

            
        kf = KFold(n_splits=5, shuffle=True, random_state=self.config.seeds)

        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
            print(f"Fold {fold + 1}/5")
            train_subset = Subset(dataset, train_idx.tolist())
            val_subset = Subset(dataset, val_idx.tolist())

            train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True, num_workers=4, 
                                      collate_fn=coco_collate_fn, pin_memory=True, generator=g, worker_init_fn=seed_worker)
            val_loader = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False, num_workers=4, 
                                    collate_fn=coco_collate_fn, pin_memory=True, generator=g, worker_init_fn=seed_worker)

            print(f"[Fold {fold+1}] Dataset-wide num_classes = {num_classes}")

            model = UNetFP16(task=self.config.task, in_channels=3, out_channels=num_classes, num_instances=num_instances).to(device)
            print("size of the model: ", count_parameters(model))
            ce_criterion = nn.CrossEntropyLoss()
            dice_crition = DiceLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=1e-5)
            scaler = GradScaler(device='cuda')

            step = 0
            for epoch in range(self.config.epochs):
                print("training, epoch: ", epoch)
                train_loss, step = self.train_one_epoch(model, train_loader, ce_criterion, dice_crition, optimizer, device, step, self.config.task, num_classes, scaler)
                print("validating, epoch: ", epoch)
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
