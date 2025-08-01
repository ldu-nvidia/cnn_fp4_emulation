# --- Imports ---
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
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
from kitchen.quantization_autograd import quantize as kitchen_quantize
from data import CocoSegmentationDataset
from config import configs
import pickle
from dataclasses import asdict
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics.functional import jaccard_index
from importlib import import_module

import shutil
from utils import (rename, parse_segmentation_masks, parse_instance_masks, parse_detection_heatmap, \
    coco_collate_fn, count_parameters, combined_loss, visualize_instance_batch, \
    get_max_instances_from_annotations, log_visual_predictions_to_file, check_mask, save_visual_predictions, \
        get_coco, get_max_instances_from_annotations, plot_grid_heatmaps, plot_interactive_3d, dice_coeff, \
            save_predictions_for_visualization, miou, instance_iou)

from loss import DiceLoss, instance_loss

SEED = 911    # pick any integer you like ────────────────┐
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

# Model factory mapping
MODEL_REGISTRY = {
    "fp16": ("models.full_precision", "UNetFP16"),
    "nvfp4": ("models.quantized_kitchen_autograd", "UNetNVFP4"),
}

def get_model_class(model_key: str):
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")
    module_name, class_name = MODEL_REGISTRY[model_key]
    try:
        module = import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_name}' for model '{model_key}': {e}")
    if not hasattr(module, class_name):
        raise AttributeError(f"Module '{module_name}' does not have class '{class_name}'")
    return getattr(module, class_name)

# 2. worker_init_fn so *each* worker gets a deterministic, unique seed
def seed_worker(worker_id: int):
    # base_seed is the same for every worker but worker_id makes it unique
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class trainer():
    def __init__(self, config):
        self.layer_stats_w_raw = []
        self.layer_stats_w_q = []
        self.layer_stats_w_deq = []
        self.layer_stats_grad = []
        self.corr_steps = []  # list of (step, correlation)
        self.config = config
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.ce_loss = None
        self.dice_loss = None
        # Pre-instantiate instance loss so we don't recreate it every batch
        self.ins_loss_fn = instance_loss()

    def log_telemetry(self, loss_value, step):
        if step % self.config.logf != 0:
            return

        stat_dict_raw, stat_dict_q, stat_dict_deq, stat_dict_grad = {}, {}, {}, {}
        raw_accum, deq_accum = [], []  # collect for correlation

        if not np.isfinite(loss_value):
            print(f"⚠️ Skipping telemetry logging due to non-finite loss at step {step}")
            return

        wandb.log({"loss": loss_value}, step=step)

        for name, param in self.model.named_parameters():
            new_name = rename(name)

            # ---- Log Weights ----
            if param.requires_grad and 'bias' not in new_name and 'weight' in new_name:
                w_fp = param.detach().cpu().float().numpy().flatten()
                if np.isfinite(w_fp).all():
                    mean = w_fp.mean().item()
                    std  = w_fp.std().item()
                    kurtosis = scipy.stats.kurtosis(w_fp)
                    stat_dict_raw[f"Weights_raw/{new_name}/mean"] = mean
                    stat_dict_raw[f"Weights_raw/{new_name}/std"] = std
                    stat_dict_raw[f"Weights_raw/{new_name}/kurtosis"] = kurtosis
                    wandb.log({f"Weights/{new_name}/hist": wandb.Histogram(w_fp, num_bins=40)}, step=step)

            # ---- Log Gradients ----
            if param.requires_grad and param.grad is not None and 'bias' not in new_name and 'weight' in new_name:
                g = param.grad.detach().cpu().float().numpy().flatten()
                if np.isfinite(g).all():
                    mean = g.mean().item()
                    std  = g.std().item()
                    kurtosis = scipy.stats.kurtosis(g)
                    stat_dict_grad[f"Gradients/{new_name}/mean"] = mean
                    stat_dict_grad[f"Gradients/{new_name}/std"] = std
                    stat_dict_grad[f"Gradients/{new_name}/kurtosis"] = kurtosis
                    wandb.log({f"Gradients/{new_name}/hist": wandb.Histogram(g, num_bins=40)}, step=step)

            # ---- Log Quantized Weights ----
            modules_dict = dict(self.model.named_modules())
            mod_prefix = name.rsplit('.', 1)[0]
            # Walk up the hierarchy until a module with qparams_w is found
            parts = mod_prefix.split('.')
            module = None
            while parts:
                candidate = '.'.join(parts)
                m = modules_dict.get(candidate, None)
                if m is not None and hasattr(m, 'qparams_w'):
                    module = m
                    break
                parts.pop()

            if module is not None:
                try:
                    # Quantized int codes and scale from kitchen quantize
                    qparams = module.qparams_w
                    block = qparams.quant_tile_shape[1]
                    flat = param.detach().view(-1)
                    pad_len = (block - (flat.numel() % block)) % block
                    if pad_len:
                        flat = torch.cat([flat, torch.zeros(pad_len, device=flat.device, dtype=flat.dtype)])
                    flat = flat.view(-1, block)
                    q_int, sx, _ = kitchen_quantize(flat, qparams, reduction_dim=-1)
                    q_int = q_int.view(-1)
                    if pad_len:
                        q_int = q_int[:-pad_len]
                        sx = sx.view(-1).repeat_interleave(block)[:-pad_len]
                    else:
                        sx = sx.view(-1).repeat_interleave(block)

                    # stats for quantized ints
                    w_q = q_int.cpu().float().numpy()
                    if np.isfinite(w_q).all():
                        stat_dict_q[f"Weights_q/{new_name}/mean"] = np.mean(w_q)
                        stat_dict_q[f"Weights_q/{new_name}/std"] = np.std(w_q)
                        stat_dict_q[f"Weights_q/{new_name}/kurtosis"] = scipy.stats.kurtosis(w_q)
                        wandb.log({f"Weights_q_int/{new_name}/hist": wandb.Histogram(w_q, num_bins=40)}, step=step)

                    # dequantized values
                    deq = (q_int.float() * sx).cpu().numpy()
                    if np.isfinite(deq).all():
                        # accumulate for correlation (keep tensor pairs aligned)
                        raw_accum.append(w_fp)
                        stat_dict_deq[f"Weights_deq/{new_name}/mean"] = np.mean(deq)
                        stat_dict_deq[f"Weights_deq/{new_name}/std"] = np.std(deq)
                        stat_dict_deq[f"Weights_deq/{new_name}/kurtosis"] = scipy.stats.kurtosis(deq)
                        wandb.log({f"Weights_deq/{new_name}/hist": wandb.Histogram(deq, num_bins=40)}, step=step)
                        deq_accum.append(deq)
                except Exception as e:
                    print(f"[telemetry] dequant logging failed for {new_name}: {e}")

        stat_dict_raw['step'] = step
        stat_dict_q['step'] = step
        stat_dict_deq['step'] = step
        stat_dict_grad['step'] = step
        self.layer_stats_w_raw.append(stat_dict_raw)
        self.layer_stats_w_q.append(stat_dict_q)
        self.layer_stats_w_deq.append(stat_dict_deq)
        self.layer_stats_grad.append(stat_dict_grad)

        # ---- Correlation between raw and dequantised ----
        if raw_accum and deq_accum and len(raw_accum)==len(deq_accum):
            raw_concat = np.concatenate(raw_accum)
            deq_concat = np.concatenate(deq_accum)
            corr = np.corrcoef(raw_concat, deq_concat)[0,1]
            self.corr_steps.append((step, float(corr)))
            wandb.log({"weights_raw_deq/corr": corr}, step=step)

    def save_layer_stats(self):
        model_key = getattr(self, 'current_model_key', 'model')
        base_dir = f"plots/heatmaps/{model_key}"
        task_prefix = self.config.task  # 'semantic' or 'instance'
        mapping = [
             ("weights/prequantize", self.layer_stats_w_raw, f"{task_prefix}_weights.json"),
             ("weights/quantized",   self.layer_stats_w_q,   f"{task_prefix}_weights.json"),
             ("weights/dequantized", self.layer_stats_w_deq, f"{task_prefix}_weights.json"),
             ("gradients",           self.layer_stats_grad,  f"{task_prefix}_gradients.json"),
        ]

        for sub, layer_stats, fname in mapping:
            out_dir = os.path.join(base_dir, sub)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)
            with open(out_path, "w") as f:
                serializable_stats = [
                    {k: float(v) if hasattr(v, 'item') and hasattr(v, 'dtype') else v for k, v in entry.items()}
                    for entry in layer_stats
                ]
                json.dump(serializable_stats, f)
        print("✅ Layer stats saved to plot folder structure under", base_dir)

    def finalize_and_visualize(self):
        print("\n finalizing and visualizing \n")
        assert len(self.layer_stats_w_raw) != 0 and len(self.layer_stats_grad) != 0, "weight or grad stats are empty"
        print("saving layer stats")
        self.save_layer_stats()
        type_list = ['weights_prequant', 'weights_quantized', 'weights_dequant', 'grads']
        subfolder_map = {
            'weights_prequant': 'prequantize',
            'weights_quantized': 'quantized',
            'weights_dequant': 'dequantized',
            'grads': ''  # empty triggers gradients folder path
        }

        for layer_stats, type in zip([self.layer_stats_w_raw, self.layer_stats_w_q, self.layer_stats_w_deq, self.layer_stats_grad], type_list):
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
            subfolder = subfolder_map[type]
            plot_grid_heatmaps(tensor, layer_names, list(stat_map.keys()), self.config, type_label=type, model_key=self.current_model_key, subfolder=subfolder)
            plot_interactive_3d(tensor, layer_names, list(stat_map.keys()), self.config, type_label=type, model_key=self.current_model_key, subfolder=subfolder)
            print("✅ Plots saved to plot folder")

        # ---- Correlation plot ----
        if self.corr_steps:
            import matplotlib.pyplot as plt
            steps, corrs = zip(*self.corr_steps)
            base = os.path.join("plots", "correlation", self.current_model_key)
            os.makedirs(base, exist_ok=True)
            plt.figure(figsize=(8,4))
            plt.plot(steps, corrs, marker='o')
            plt.title("Correlation between Raw and Dequantised Weights")
            plt.xlabel("Step")
            plt.ylabel("Pearson r")
            plt.grid(True)
            plt.savefig(os.path.join(base, "raw_deq_correlation.png"))
            plt.close()
            print("✅ Correlation plot saved")

        print("\n task finished \n")

    def compute_batch_instance_iou(self, pred_mask, gt_mask):
        # Shapes & dtypes
        assert pred_mask.ndim == 2,  "pred_mask must be HxW"
        assert gt_mask.ndim  == 3,  "gt_mask  must be NxHxW"
        assert pred_mask.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8), "pred_mask must be integer type"
        # Background (0) might legitimately be absent if the model predicted only instances in this image.
        assert gt_mask.shape[1:] == pred_mask.shape, "spatial dim mismatch"

        ious = []

        for gt in gt_mask:
            gt_bin = gt.bool()
            best_iou = 0.0

            # Loop through each unique instance label in prediction (excluding background)
            for instance_id in pred_mask.unique():
                if instance_id.item() == 0:
                    continue  # skip background

                pred_bin = (pred_mask == instance_id)
                intersection = (gt_bin & pred_bin).sum().item()
                union = (gt_bin | pred_bin).sum().item()
                if union > 0:
                    iou = intersection / union
                    best_iou = max(best_iou, iou)

            ious.append(best_iou)

        iou_avg = np.mean(ious) if ious else 0.0
        assert 0.0 <= iou_avg <= 1.0 + 1e-6, "IoU out of [0,1] range"
        return iou_avg


    def train_one_epoch(self, train_loader, device, global_step, task, num_cls):
        self.model.train()
        tr_loss, dice_sum, inst_iou_sum, batches = 0.0, 0.0, 0.0, 0
        step0 = global_step
        for imgs, tgts in train_loader:
            imgs = imgs.to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                outs = self.model(imgs)
                H, W = outs.shape[2:]

                if task == "semantic":
                    gt = parse_segmentation_masks(tgts, H, W, num_cls, self.cat2idx).to(device)
                    loss = combined_loss(outs, gt, self.ce_loss, self.dice_loss)
                    preds = outs.argmax(1)
                    gt_lbl = gt.argmax(1)
                    dice_sum += dice_coeff(preds.cpu(), gt_lbl.cpu())

                elif task == "instance":
                    masks, _ = parse_instance_masks(tgts, H, W, self.cat2idx)
                    masks = [m.to(device) for m in masks]
                    # Use the pre-built loss function
                    loss = self.ins_loss_fn(outs, masks)

                    preds = outs.argmax(1)  # [B,H,W] integer instance IDs
                    batch_iou = 0.0
                    for p, g in zip(preds, masks):
                        batch_iou += self.compute_batch_instance_iou(p, g)
                    batch_iou /= max(len(masks), 1)
                    inst_iou_sum += batch_iou

                else:
                    loss = torch.zeros([], device=device)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tr_loss += loss.item()
            global_step += 1
            batches += 1

            # 🔍 Log intermediate stats every `logf` steps
            if self.config.enable_logging and global_step % self.config.logf == 0:
                log_dict = {"train/loss": loss.item()}
                if task == "semantic":
                    log_dict["train/dice"] = dice_sum / batches
                if task == "instance":
                    log_dict["train/instance_iou"] = inst_iou_sum / batches
                wandb.log(log_dict, step=global_step)
                self.log_telemetry(loss.item(), global_step)

        avg_dice = dice_sum / max(batches, 1) if task == "semantic" else None
        avg_iou = inst_iou_sum / max(batches, 1) if task == "instance" else None
        return tr_loss / max(batches, 1), avg_dice, avg_iou, global_step

    # ------------------------ Validation --------------------------------
    def validate(self, loader, ce, dice_loss,
                dev, step0, task, num_cls):
        self.model.eval()
        print("validating")
        total_loss = dice_sum = miou_sum = inst_iou_sum = 0.0
        step    = step0
        batches = 0

        with torch.no_grad():
            for imgs, tgts in loader:
                imgs = imgs.to(dev)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outs = self.model(imgs)                         # [B,C,H,W]
                    H, W = outs.shape[2:]
                    if task == "semantic":
                        gt   = parse_segmentation_masks(
                                tgts, H, W, num_cls, self.cat2idx).to(dev)
                        batch_loss = combined_loss(outs, gt, ce, dice_loss)
                        total_loss += batch_loss.item()

                        preds   = outs.argmax(1)               # still on GPU
                        gt_lbl  = gt.argmax(1)
                        dice_sum += dice_coeff(preds.cpu(), gt_lbl.cpu())
                        miou_sum += jaccard_index(
                                        preds, gt_lbl,
                                        num_classes=num_cls,
                                        task='multiclass',
                                        average='micro'
                                    ).item()
                        batches += 1

                    elif task == "instance":
                        masks, _ = parse_instance_masks(tgts, H, W, self.cat2idx)
                        masks = [m.to(dev) for m in masks]
                        # Use the pre-built loss function
                        batch_loss = self.ins_loss_fn(outs, masks)
                        total_loss += batch_loss.item()

                        preds = outs.argmax(1)  # [B,H,W] integer instance IDs
                        batch_iou = 0.0
                        for p, g in zip(preds, masks):
                            batch_iou += self.compute_batch_instance_iou(p, g)
                        batch_iou /= max(len(masks), 1)
                        inst_iou_sum += batch_iou
                        batches += 1
                    else:   # detection
                        gt   = parse_detection_heatmap(tgts, H, W).to(dev)
                        batch_loss = torch.zeros([], device=dev)
                        total_loss += batch_loss.item()
                step += 1

        # ---- average metrics -----------------------------------------------
        loss   = total_loss / max(batches, 1)
        dice_av = dice_sum   / max(batches, 1)
        miou_av = miou_sum   / max(batches, 1)
        inst_av = inst_iou_sum / max(batches, 1)

        if self.config.enable_logging:
            if task == "semantic":
                wandb.log({"val/dice": dice_av, "val/miou": miou_av}, step=step)
            elif task == "instance":
                wandb.log({"val/instance_iou": inst_av}, step=step)
            wandb.log({"val/loss": loss}, step = step)
        return loss, step

    # ---------------------------- run() ---------------------------------
    def run(self):
        if os.path.exists("plots/"):
            shutil.rmtree("plots/")

        gpu_index = 0
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(gpu_index)

        # Loop through each requested model configuration
        for model_key in self.config.models:
            print(f"\n================ Training model: {model_key} ================\n")

            # fresh telemetry buffers per model
            self.layer_stats_w_raw.clear()
            self.layer_stats_w_q.clear()
            self.layer_stats_w_deq.clear()
            self.layer_stats_grad.clear()
            self.corr_steps.clear() # Clear correlation steps for new model

            # Initialise W&B run per model if logging enabled
            if self.config.enable_logging:
                wandb.init(project=self.config.project_name,
                           name=f"{self.config.wandb_name}_{model_key}",
                           config={**asdict(self.config), "model": model_key},
                           reinit=True)  # new run each loop

            # dataset and loaders constructed only once outside loop (reuse) ----------------
            if not hasattr(self, "_data_prepared"):
                cache_path = "cat2idx_cache.pkl"
                if os.path.exists(cache_path):
                    print("Loading cached cat2idx dictionary...")
                    with open(cache_path, "rb") as f:
                        self.cat2idx = pickle.load(f)
                else:
                    print("Computing and caching cat2idx dictionary...")
                    coco = get_coco(self.config.ann_file)
                    cat_ids = sorted(coco.getCatIds())
                    self.cat2idx = {cid: i for i, cid in enumerate(cat_ids)}
                    with open(cache_path, "wb") as f:
                        pickle.dump(self.cat2idx, f)

                dataset = CocoSegmentationDataset(
                    img_root=self.config.coco_root,
                    ann_file=self.config.ann_file,
                    category_id_to_class_idx=self.cat2idx,
                    target_size=(256, 256))

                if self.config.task == "semantic":
                    num_cls, num_inst = len(self.cat2idx), -1
                elif self.config.task == "instance":
                    num_inst = get_max_instances_from_annotations(self.config.ann_file)
                    num_cls = len(self.cat2idx)
                    print(f"Max instances / image = {num_inst}")
                else:
                    raise ValueError("Unsupported task type")

                val_frac = 0.001
                val_len = int(val_frac * len(dataset))
                train_len = len(dataset) - val_len
                train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len], generator=g)

                self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True, collate_fn=coco_collate_fn,
                    generator=g, worker_init_fn=seed_worker)

                self.val_loader = DataLoader(val_set, batch_size=self.config.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True, collate_fn=coco_collate_fn,
                    generator=g, worker_init_fn=seed_worker)

                self._data_prepared = True  # flag

            # Build model -----------------------------------------------------
            ModelCls = get_model_class(model_key)
            self.model = ModelCls(task=self.config.task, in_channels=3,
                                  out_channels=num_cls, num_instances=num_inst,
                                  scale_factor=self.config.model_scale_factor).to(device)
            count_parameters(self.model)

            # fresh optimizer & scaler
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1e-5)
            self.scaler = GradScaler(device="cuda")

            self.ce_loss = nn.CrossEntropyLoss()
            self.dice_loss = DiceLoss()

            global_step = 0
            for epoch in range(self.config.epochs):
                print(f"\nEpoch {epoch} ({model_key})")
                tr_loss, tr_dice, tr_iou, global_step = self.train_one_epoch(self.train_loader, device, global_step, self.config.task, num_cls)
                #vl_loss, global_step = self.validate(self.val_loader, self.ce_loss, self.dice_loss, device, global_step, self.config.task, num_cls)

                def fmt(v): return f"{v:.5f}" if v is not None else "-"
                #print(f"[{model_key}] epoch loss: {fmt(tr_loss)}  dice: {fmt(tr_dice)}  iou: {fmt(tr_iou)}  val loss: {fmt(vl_loss)}")

                save_predictions_for_visualization(self.model, self.val_loader, device, self.config.task, self.cat2idx, epoch, model_key)

            os.makedirs("checkpoints/", exist_ok=True)
            ckpt_path = f"checkpoints/{self.config.project_name}_{self.config.wandb_name}_{model_key}_unet_final.pth"
            torch.save(self.model.state_dict(), ckpt_path)
            self.current_model_key = model_key
            self.finalize_and_visualize()

            if self.config.enable_logging:
                wandb.finish()

        print("\nAll requested models have finished training.\n")
