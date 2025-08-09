import torch
import numpy as np
from pycocotools import mask as coco_mask
from torchvision.transforms.functional import resize, InterpolationMode, to_pil_image
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
import os
from PIL import Image
import matplotlib.cm as cm
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image

# note relu layer has no weight or bias
block_to_layer_dict = {'0': "conv2d_1st",
                       '1': 'groupNorm_1st',
                       '3': 'conv2d_2nd',
                       '4': 'groupNorm_2nd'}

# marco related to the coco dataset
_COCO_CACHE = {}

def denormalize_image(img_tensor):
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    return img_np

def visualize_instance_batch(image_np, pred_mask, gt_masks, gt_classes, num_classes):
    height, width = pred_mask.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    class_cmap = plt.cm.get_cmap('tab20', num_classes)

    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    pred_overlay = class_cmap(pred_mask / num_classes)
    axes[1].imshow(image_np)
    axes[1].imshow(pred_overlay, alpha=0.5)
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis("off")

    axes[2].imshow(image_np)
    for i, mask in enumerate(gt_masks):
        color = class_cmap(gt_classes[i] / num_classes)
        colored_mask = np.zeros((height, width, 4), dtype=np.float32)
        colored_mask[..., :3] = color[:3]
        colored_mask[..., 3] = 0.5 * mask.float().cpu().numpy()
        axes[2].imshow(colored_mask)
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0 and len(y_indices) > 0:
            x1, y1 = np.min(x_indices), np.min(y_indices)
            x2, y2 = np.max(x_indices), np.max(y_indices)
            rect = mpatches.Rectangle((float(x1), float(y1)), float(x2 - x1), float(y2 - y1),
                                      linewidth=1, edgecolor=color, facecolor='none')
            axes[2].add_patch(rect)

    axes[2].set_title("Ground Truth Instances")
    axes[2].axis("off")
    fig.tight_layout()
    return fig

def rename(old_name):
    new_name = ""
    if 'block' in old_name:
        splitted = old_name.split('.')
        layer, blk_num, type = splitted[0], splitted[-2], splitted[-1]
        if blk_num in block_to_layer_dict:
            new_name = f"{layer}.{block_to_layer_dict[blk_num]}.{type}"
        else:
            # Unknown block identifier; fall back to original name
            new_name = old_name
    else:
        new_name = old_name
    return new_name

def parse_segmentation_masks(targets, height, width, num_classes, category_id_to_class_idx):
    batch_masks = []
    for anns in targets:
        mask = torch.zeros((num_classes, height, width), dtype=torch.float32)
        for ann in anns:
            if not ann.get("segmentation"):
                continue

            coco_cat_id = ann["category_id"]
            if coco_cat_id not in category_id_to_class_idx:
                continue

            class_idx = category_id_to_class_idx[coco_cat_id]

            # --- decode at ORIGINAL resolution ------------------------------
            H0, W0 = ann.get("orig_height"), ann.get("orig_width")
            if H0 is None or W0 is None:
                raise KeyError("orig_height / orig_width not found in annotation; make sure data.py attaches them.")

            rle = coco_mask.frPyObjects(ann["segmentation"], H0, W0)
            decoded = coco_mask.decode(rle)
            if decoded.ndim == 3:                    # merge polygon parts
                decoded = np.any(decoded, axis=2)

            m_tensor = torch.from_numpy(decoded.astype(np.float32))
            # --- resize to training resolution --------------------------------
            m_tensor = TF.resize(
                m_tensor.unsqueeze(0),
                size=[height, width],
                interpolation=TF.InterpolationMode.NEAREST,
            ).squeeze(0)

            mask[class_idx] = torch.max(mask[class_idx], m_tensor)
        batch_masks.append(mask)
    return torch.stack(batch_masks)

def parse_instance_masks(targets, target_h, target_w, category_id_to_class_idx=None):
    batch_masks, batch_classes = [], []

    for anns in targets:
        inst_masks, inst_classes = [], []
        for ann in anns:
            if not ann.get("segmentation"):          # skip empty seg
                continue

            H0, W0 = ann["orig_height"], ann["orig_width"]
            rle = coco_mask.frPyObjects(ann["segmentation"], H0, W0)
            decoded = coco_mask.decode(rle)
            if decoded.ndim == 3:                    # merge parts
                decoded = np.any(decoded, axis=2)

            m = torch.from_numpy(decoded.astype(np.float32))
            m = TF.resize(m.unsqueeze(0), size=[target_h, target_w],
                          interpolation=InterpolationMode.NEAREST).squeeze(0)
            inst_masks.append(m)

            cat_id = ann["category_id"]
            cls = category_id_to_class_idx[cat_id] if category_id_to_class_idx else cat_id
            inst_classes.append(cls)

        batch_masks.append(torch.stack(inst_masks) if inst_masks
                           else torch.zeros((0, target_h, target_w)))
        batch_classes.append(inst_classes)

    return batch_masks, batch_classes

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

def coco_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable

def combined_loss(logits, targets, ce_criterion):
    """Unified loss for semantic segmentation.

    Supports two target formats:
      - one-hot: FloatTensor [B,C,H,W]
      - integer labels: LongTensor [B,H,W]
    """
    if targets.ndim == 4:
        # One-hot
        ce_targets = torch.argmax(targets, dim=1)
        ce_loss = ce_criterion(logits, ce_targets)
    elif targets.ndim == 3:
        # Integer labels
        ce_loss = ce_criterion(logits, targets)
    else:
        raise AssertionError(f"Unsupported targets shape: {targets.shape}")
    return ce_loss

def save_visual_predictions(images, targets, preds, config, task, num_classes, category_id_to_class_idx, save_prefix="train_step"):
    height, width = preds.shape[1:]
    B = images.size(0)
    os.makedirs(save_prefix, exist_ok=True)

    if task == "instance":
        for i in range(min(2, B)):
            img_np = denormalize_image(images[i])
            pred_mask = preds[i].numpy()
            inst_masks, class_ids = parse_instance_masks([targets[i]], height, width, category_id_to_class_idx)
            fig = visualize_instance_batch(img_np, pred_mask, inst_masks[0], class_ids[0], num_classes)
            fig.savefig(f"{save_prefix}instance_{i}.png")
            plt.close(fig)
    else:  # semantic
        gt_masks = parse_segmentation_masks(targets, height, width, num_classes, category_id_to_class_idx).argmax(1).cpu()
        for i in range(min(2, B)):
            img_np = TF.to_pil_image(images[i].cpu())
            pred_mask = preds[i].numpy()
            gt_mask = gt_masks[i].numpy()
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np)
            axes[0].set_title("Input Image")
            axes[1].imshow(pred_mask, alpha=0.5, cmap='jet')
            axes[1].set_title("Prediction")
            axes[2].imshow(gt_mask, alpha=0.5, cmap='jet')
            axes[2].set_title("Ground Truth")
            for ax in axes: ax.axis("off")
            fig.tight_layout()
            fig.savefig(f"{save_prefix}semantic_{i}.png")
            plt.close(fig)


def log_visual_predictions_to_file(
    config,
    model,
    dataloader,
    device,
    num_classes,
    category_id_to_class_idx,
):
    output_dir="plots/final_check/"
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    assert num_classes > 0, "num_classes must be a positive integer"
    class_colors = plt.cm.get_cmap("tab20", num_classes)

    # Use only the last batch from the dataloader
    images, targets = list(dataloader)[-1]
    assert isinstance(images, torch.Tensor)

    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)

    preds = torch.argmax(outputs, dim=1).cpu()
    height, width = preds.shape[1:]
    B = images.size(0)

    if config.task == "instance":
        for i in range(B):
            img_np = denormalize_image(images[i])
            pred_mask = preds[i].numpy()
            inst_masks, class_ids = parse_instance_masks([targets[i]], height, width, category_id_to_class_idx)
            fig = visualize_instance_batch(img_np, pred_mask, inst_masks[0], class_ids[0], num_classes)
            out_path = os.path.join(output_dir, f"instance_final_check_{i}.png")
            fig.savefig(out_path)
            print(f"Saved: {out_path}")
            plt.close(fig)

    else:  # semantic segmentation
        if isinstance(targets, list):
            gt_masks = parse_segmentation_masks(targets, height, width, num_classes, category_id_to_class_idx)
            gt_masks = gt_masks.argmax(1).cpu()
        else:
            gt_masks = targets.cpu()
        for i in range(B):
            img_np = to_pil_image(images[i].cpu())
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
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()

            out_path = os.path.join(output_dir, f"semantic_finanl_check_{i}.png")
            fig.savefig(out_path)
            print(f"Saved: {out_path}")
            plt.close(fig)
        

def check_mask(img_idx: int,
               img_root="coco2017/images/train2017",
               ann_file="coco2017/annotations/instances_train2017.json"):
    """
    Quick visual sanity-check of one COCO image + masks after resizing.

    Saves two PNGs in plots/initial_check/
        • semantic_mask_check.png  (category-ID heat-map)
        • instance_mask_check.png  (random colour per instance)
    """
    # ------------------ I/O paths --------------------------------------
    os.makedirs("plots/", exist_ok=True)
    out_dir = "plots/initial_check/"
    os.makedirs(out_dir, exist_ok=True)
    sem_path = os.path.join(out_dir, "semantic_mask_check.png")
    ins_path = os.path.join(out_dir, "instance_mask_check.png")

    # ------------------ COCO & image meta ------------------------------
    coco   = COCO(ann_file)
    img_id = coco.getImgIds()[img_idx]
    info   = coco.loadImgs(img_id)[0]              # dict with H,W & filename

    image  = Image.open(os.path.join(img_root, info["file_name"])).convert("RGB")
    W, H   = image.size
    img_np = np.array(image)

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns    = coco.loadAnns(ann_ids)

    # ------------------ allocate masks --------------------------------
    sem_mask  = np.zeros((H, W), dtype=np.uint8)   # category-id  per pixel
    inst_mask = np.zeros((H, W), dtype=np.int32)   # small ID      per pixel

    # ------------------ fill masks ------------------------------------
    for idx, ann in enumerate(anns, start=1):      # idx = 1,2,3…
        # annToMask handles polygons & RLE transparently
        binary = coco.annToMask(ann)               # ndarray H×W (0/1)
        assert binary.shape == (H, W), "Mask size mismatch"

        cat_id = ann["category_id"]
        sem_mask[binary > 0]  = cat_id
        inst_mask[binary > 0] = idx                # safe small integer

    # ------------------ save semantic visual --------------------------
    norm = sem_mask.astype(float) / max(1, sem_mask.max())
    sem_rgb = (plt.cm.get_cmap("nipy_spectral")(norm)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(sem_rgb).save(sem_path)

    # ------------------ save instance visual --------------------------
    rand_col = np.random.randint(0, 255,
                                 (inst_mask.max() + 1, 3),
                                 dtype=np.uint8)
    inst_rgb = rand_col[inst_mask]
    overlay  = (0.2 * img_np + 0.8 * inst_rgb).astype(np.uint8)
    Image.fromarray(overlay).save(ins_path)

    print(f"✅ Semantic mask saved  ➜  {sem_path}")
    print(f"✅ Instance mask saved  ➜  {ins_path}\n")

def get_coco(ann_file):
    if ann_file not in _COCO_CACHE:
        _COCO_CACHE[ann_file] = COCO(ann_file)
    return _COCO_CACHE[ann_file]

def get_max_instances_from_annotations(ann_file):
    coco = get_coco(ann_file)
    return max(len(anns) for anns in coco.imgToAnns.values())

def get_max_category_id(ann_file):
    coco = get_coco(ann_file)
    return max(coco.getCatIds())

def plot_grid_heatmaps(tensor, layer_names, stat_names, args, type_label, model_key="", subfolder=""):
    os.makedirs("plots/heatmaps/", exist_ok=True)
    # determine nested directory: weights/<subfolder> or gradients
    base = f"plots/heatmaps/{model_key}"
    if subfolder:
        base = os.path.join(base, "weights", subfolder)
    else:
        base = os.path.join(base, "gradients")
    os.makedirs(base, exist_ok=True)
    signal = "gradients" if "grad" in type_label else "weights"
    out_path = f"{base}/{args.task}_{signal}.png"
    steps = tensor.shape[0]
    fig, axs = plt.subplots(len(stat_names), 1, figsize=(15, 10 * len(stat_names)), squeeze=False)
    if model_key:
        fig.suptitle(f"Model: {model_key}", fontsize=34, y=0.97)
        plt.subplots_adjust(top=0.9)
    # Add a dedicated title on every subplot (mean, std, kurtosis)
    singular_type = type_label[:-1] if type_label.endswith('s') else type_label  # e.g., weights ➜ weight
    for i, stat in enumerate(stat_names):
        ax = axs[i, 0]
        ax.set_title(f"{singular_type.capitalize()} Stats: {stat}", fontsize=30)
        im = ax.imshow(tensor[:, :, i].T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xticks(list(range(steps)))
        ax.set_yticks(list(range(len(layer_names))))
        layer_names = [name.replace("weight", "") if "weight" in name else name for name in layer_names]
        ax.set_yticklabels(layer_names)
        plt.xlabel("Every "+ str(args.logf) +" Steps", fontsize=20)
        # per-ax label removed to avoid redundancy
        fig.colorbar(im, ax=ax)
    plt.savefig(out_path)
    plt.close()

def plot_interactive_3d(tensor, layer_names, stat_names, args, type_label, model_key="", subfolder=""):
    base = f"plots/heatmaps/{model_key}"
    if subfolder:
        base = os.path.join(base, "weights", subfolder)
    else:
        base = os.path.join(base, "gradients")
    os.makedirs(base, exist_ok=True)
    signal = "gradients" if "grad" in type_label else "weights"
    out_path = f"{base}/{args.task}_{signal}.html"
    steps, layers, stats = tensor.shape
    fig = go.Figure()
    for i in range(stats):
        z = tensor[:, :, i].T
        fig.add_trace(go.Surface(z=z, name=stat_names[i]))
    title_txt = "Layer Stats 3D" + (f" - {model_key}" if model_key else "")
    fig.update_layout(title=title_txt, scene=dict(
        xaxis_title="Step",
        yaxis_title="Layer",
        zaxis_title="Stat Value",
        yaxis=dict(tickmode='array', tickvals=list(range(len(layer_names))), ticktext=layer_names)
    ))
    fig.write_html(out_path) # or appropriate import


def get_color(index):
    # Use a fixed palette of distinguishable colors
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (64, 64, 64), (192, 192, 192)
    ]
    return COLORS[index % len(COLORS)]

def save_predictions_for_visualization(model, val_loader, device, task, cat2idx, epoch, model_key=""):
    print("saving predictions for visualization after validation step")
    model.eval()
    os.makedirs("plots/val_output", exist_ok=True)

    with torch.no_grad():
        for batch_idx, (imgs, tgts) in enumerate(val_loader):
            imgs = imgs.to(device)
            outs = model(imgs)
            preds = outs.argmax(1)

            for i in range(min(len(imgs), 4)):
                img = TF.to_pil_image(imgs[i].cpu())
                img_tensor = TF.to_tensor(img)

                H, W = outs.shape[2:]

                if task == "semantic":
                    if isinstance(tgts, list):
                        # COCO path: build from annotations
                        gt = parse_segmentation_masks([tgts[i]], H, W, len(cat2idx), cat2idx).argmax(1)[0]
                    else:
                        # MVTec path: integer mask [H,W]
                        gt = tgts[i].cpu()
                    pred = preds[i].cpu()

                    gt_masks = [(gt == k) for k in gt.unique() if k.item() > 0]
                    pred_masks = [(pred == k) for k in pred.unique() if k.item() > 0]

                    gt_colors = [get_color(idx) for idx in range(len(gt_masks))]
                    pred_colors = [get_color(idx) for idx in range(len(pred_masks))]

                    if len(gt_masks) > 0:
                        gt_overlay = draw_segmentation_masks(img_tensor.clone(), torch.stack(gt_masks), alpha=0.5, colors=gt_colors)  # type: ignore[arg-type]
                    else:
                        gt_overlay = img_tensor.clone()

                    if len(pred_masks) > 0:
                        pred_overlay = draw_segmentation_masks(img_tensor.clone(), torch.stack(pred_masks), alpha=0.5, colors=pred_colors)  # type: ignore[arg-type]
                    else:
                        pred_overlay = img_tensor.clone()

                elif task == "instance":
                    masks, _ = parse_instance_masks([tgts[i]], H, W, cat2idx)
                    masks = masks[0].cpu()  # shape [N, H, W]
                    pred = preds[i].cpu()

                    gt_masks = [m.bool() for m in masks]
                    # Skip background = 0 when visualising predictions
                    pred_instance_ids = [k for k in pred.unique() if k.item() > 0]
                    pred_masks = [(pred == k) for k in pred_instance_ids]

                    gt_colors = [get_color(idx) for idx in range(len(gt_masks))]
                    pred_colors = [get_color(idx) for idx in range(len(pred_masks))]

                    if len(gt_masks) > 0:
                        gt_overlay = draw_segmentation_masks(img_tensor.clone(), torch.stack(gt_masks), alpha=0.5, colors=gt_colors)  # type: ignore[arg-type]
                    else:
                        gt_overlay = img_tensor.clone()

                    if len(pred_masks) > 0:
                        pred_overlay = draw_segmentation_masks(img_tensor.clone(), torch.stack(pred_masks), alpha=0.5, colors=pred_colors)  # type: ignore[arg-type]
                    else:
                        pred_overlay = img_tensor.clone()

                else:
                    continue

                combined = torch.cat([gt_overlay, pred_overlay], dim=2)
                out_img = TF.to_pil_image(combined)
                suffix = f"_{model_key}" if model_key else ""
                out_path = f"plots/val_output/epoch{epoch}_img{i}{suffix}.png"
                out_img.save(out_path)
            break

def dice_coeff(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps=1e-6):
    """Mean Dice over classes present in *ground-truth* (ignores background).

    Args
    -----
    pred_mask : [B,H,W] integer tensor – model predictions (argmax).
    gt_mask   : [B,H,W] integer tensor – ground-truth class indices.
    """
    assert pred_mask.shape == gt_mask.shape, "pred/gt shapes must match"

    classes = torch.unique(gt_mask)
    dice_sum = 0.0
    valid_cls = 0
    for cls in classes:
        # skip background id if you store one (assume -1 or 255). If none, keep.
        if cls.item() < 0:
            continue
        pred_c = (pred_mask == cls)
        gt_c   = (gt_mask  == cls)
        if gt_c.sum() == 0 and pred_c.sum() == 0:
            continue
        inter = (pred_c & gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        dice_sum += (2 * inter) / (union + eps)
        valid_cls += 1

    return dice_sum / max(valid_cls, 1)


def miou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int):
    """Mean IoU over classes present in GT."""
    ious = []
    for c in range(1, num_classes):                 # skip background = 0
        gt_c   = (gt_mask == c)
        pred_c = (pred_mask == c)
        if gt_c.sum() == 0 and pred_c.sum() == 0:
            continue
        inter = torch.logical_and(gt_c, pred_c).sum()
        union = torch.logical_or (gt_c, pred_c).sum()
        if union > 0: ious.append( (inter / union).item() )
    return np.mean(ious) if ious else 0.0

def instance_iou(pred_masks, gt_masks):
    """pred_masks, gt_masks: [N,H,W] bool tensors"""
    P, G = len(pred_masks), len(gt_masks)
    iou_mat = torch.zeros((P, G))
    for i,p in enumerate(pred_masks):
        for j,g in enumerate(gt_masks):
            inter = (p & g).sum()
            union = (p | g).sum()
            iou_mat[i,j] = inter.float() / union.float().clamp(min=1)
    # best matching with Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment((-iou_mat).cpu())
    return iou_mat[row_ind, col_ind].mean().item() if len(row_ind) else 0.0
