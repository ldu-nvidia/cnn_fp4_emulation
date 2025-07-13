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
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
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
        new_name = layer + "." + block_to_layer_dict[blk_num]+ "." + type
    else:
        new_name = old_name
    return new_name

def parse_segmentation_masks(targets, height, width, num_classes, category_id_to_class_idx):
    batch_masks = []
    for anns in targets:
        mask = torch.zeros((num_classes, height, width), dtype=torch.float32)
        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                coco_cat_id = ann['category_id']
                if coco_cat_id not in category_id_to_class_idx:
                    continue
                class_idx = category_id_to_class_idx[coco_cat_id]
                rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
                decoded = coco_mask.decode(rle)
                if decoded.ndim == 3:
                    decoded = np.any(decoded, axis=2)
                m_tensor = torch.from_numpy(decoded.astype(np.float32))
                m_tensor = TF.resize(m_tensor.unsqueeze(0), size=[height, width], interpolation=TF.InterpolationMode.NEAREST)
                m_tensor = m_tensor.squeeze(0)
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

def combined_loss(logits, targets, ce_criterion, dice_criterion):
    assert targets.ndim == 4, f"Expected targets [B, C, H, W], got {targets.shape}"
    ce_targets = torch.argmax(targets, dim=1)
    ce_loss = ce_criterion(logits, ce_targets)
    dice_loss = dice_criterion(logits, targets)
    return 0.7 * ce_loss + 0.3 * dice_loss

def save_visual_predictions(images, targets, preds, config, task, num_classes, category_id_to_class_idx, save_prefix="train_step"):
    height, width = preds.shape[1:]
    B = images.size(0)

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
            fig.savefig(f"{save_prefix}_semantic_{i}.png")
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
    assert isinstance(targets, list)

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
        gt_masks = parse_segmentation_masks(targets, height, width, num_classes, category_id_to_class_idx)
        gt_masks = gt_masks.argmax(1).cpu()
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


def check_mask(img_id):
    # --- Configuration ---
    img_root = "coco2017/images/train2017"
    ann_file = "coco2017/annotations/instances_train2017.json"
    output_sem_path = "semantic_mask_check.png"
    output_inst_path = "instance_mask_check.png"

    # --- Load COCO annotations ---
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    img_id = img_ids[img_id]  # just pick the first image for demo
    img_info = coco.loadImgs(img_id)[0]

    # --- Load image ---
    img_path = os.path.join(img_root, img_info['file_name'])
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    image_np = np.array(image)

    # --- Load annotations ---
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # --- Semantic mask ---
    sem_mask = np.zeros((height, width), dtype=np.uint8)
    # --- Instance mask (color-coded by instance id) ---
    inst_mask = np.zeros((height, width), dtype=np.int32)

    for idx, ann in enumerate(anns):
        cat_id = ann["category_id"]
        inst_id = ann["id"]
        if ann.get("iscrowd", 0):
            rle = ann["segmentation"]
        else:
            rle = coco_mask.frPyObjects(ann["segmentation"], height, width)

        binary_mask = coco_mask.decode(rle)
        if binary_mask.ndim == 3:
            binary_mask = np.any(binary_mask, axis=2)

        sem_mask[binary_mask > 0] = cat_id
        inst_mask[binary_mask > 0] = inst_id  # unique per instance

    os.makedirs("plots/initial_check", exist_ok=True)
    # --- Save semantic segmentation visualization ---
    norm_sem_mask = sem_mask.astype(float)
    norm_sem_mask /= max(1, norm_sem_mask.max())  # normalize to [0,1]
    sem_color = (cm.nipy_spectral(norm_sem_mask)[:, :, :3] * 255).astype(np.uint8)
    sem_image = Image.fromarray(sem_color)
    sem_image.save("plots/initial_check/"+output_sem_path)

    # --- Save instance segmentation visualization ---
    rand_colors = np.random.randint(0, 255, (inst_mask.max() + 1, 3), dtype=np.uint8)
    inst_rgb = rand_colors[inst_mask]
    overlay = (0.2 * image_np + 0.8 * inst_rgb).astype(np.uint8)
    inst_image = Image.fromarray(overlay)
    inst_image.save("plots/initial_check/"+output_inst_path)

    print(f"✅ Saved initial semantic mask to {output_sem_path}")
    print(f"✅ Saved initial instance overlay to {output_inst_path}\n\n")



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


def plot_grid_heatmaps(tensor, layer_names, stat_names, args, type):
    os.makedirs("plots/heatmaps/", exist_ok=True)
    out_path = "plots/heatmaps/" + args.task + "_" + type + ".png"
    steps = tensor.shape[0]
    fig, axs = plt.subplots(len(stat_names), 1, figsize=(15, 10 * len(stat_names)), squeeze=False)
    # Add a dedicated title on every subplot (mean, std, kurtosis)
    singular_type = type[:-1] if type.endswith('s') else type  # e.g., weights ➜ weight
    for i, stat in enumerate(stat_names):
        ax = axs[i, 0]
        ax.set_title(f"{singular_type.capitalize()} Stats: {stat}", fontsize=30)
        im = ax.imshow(tensor[:, :, i].T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xticks(list(range(steps)))
        ax.set_yticks(list(range(len(layer_names))))
        layer_names = [name.replace("weight", "") if "weight" in name else name for name in layer_names]
        ax.set_yticklabels(layer_names)
        plt.xlabel("Every "+ str(args.logf) +" Steps")
        fig.colorbar(im, ax=ax)
    plt.savefig(out_path)
    plt.close()

def plot_interactive_3d(tensor, layer_names, stat_names, args, type):
    os.makedirs("plots/heatmaps/", exist_ok=True)
    out_path="plots/heatmaps/" + args.task + "_" + type + ".html"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    steps, layers, stats = tensor.shape
    fig = go.Figure()
    for i in range(stats):
        z = tensor[:, :, i].T
        fig.add_trace(go.Surface(z=z, name=stat_names[i]))
    fig.update_layout(title="Layer Stats 3D", scene=dict(
        xaxis_title="Step",
        yaxis_title="Layer",
        zaxis_title="Stat Value",
        yaxis=dict(tickmode='array', tickvals=list(range(len(layer_names))), ticktext=layer_names)
    ))
    fig.write_html(out_path)