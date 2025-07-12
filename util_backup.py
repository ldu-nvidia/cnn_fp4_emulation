import torch
import numpy as np
from pycocotools import mask as coco_mask
from torchvision.transforms.functional import resize, InterpolationMode
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO

# note relu layer has no weight or bias
block_to_layer_dict = {'0': "conv2d_1st",
                       '1': 'groupNorm_1st',
                       '3': 'conv2d_2nd',
                       '4': 'groupNorm_2nd'}

def visualize_instance_batch(image_np, pred_mask, gt_masks, gt_classes, num_classes):
    """
    image_np: H x W x 3 (numpy array, float32 or uint8)
    pred_mask: H x W (int numpy array)
    gt_masks: list of H x W binary numpy arrays (one per instance)
    gt_classes: list of class indices corresponding to gt_masks
    num_classes: total number of classes
    """
    height, width = pred_mask.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    class_cmap = plt.cm.get_cmap('tab20', num_classes)

    # 1. Raw image
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # 2. Predicted semantic segmentation
    pred_overlay = class_cmap(pred_mask / num_classes)
    axes[1].imshow(image_np)
    axes[1].imshow(pred_overlay, alpha=0.5)
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis("off")

    # 3. GT instance masks
    axes[2].imshow(image_np)
    for i, mask in enumerate(gt_masks):
        color = class_cmap(gt_classes[i] / num_classes)
        colored_mask = np.zeros((height, width, 4), dtype=np.float32)
        colored_mask[..., :3] = color[:3]
        colored_mask[..., 3] = 0.5 * mask.astype(np.float32)
        axes[2].imshow(colored_mask)

        # Draw bounding box around instance
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
            if 'segmentation' in ann and ann['segmentation']:  # ensure segmentation is non-empty
                coco_cat_id = ann['category_id']
                if coco_cat_id not in category_id_to_class_idx:
                    continue
                class_idx = category_id_to_class_idx[coco_cat_id]

                rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
                decoded = coco_mask.decode(rle)  # shape: (H, W) or (H, W, N)

                if decoded.ndim == 3:
                    decoded = np.any(decoded, axis=2)  # union all segments
                m_tensor = torch.from_numpy(decoded.astype(np.float32))  # [H, W]

                m_tensor = m_tensor.unsqueeze(0)
                m_tensor = TF.resize(m_tensor, size=[height, width], interpolation=TF.InterpolationMode.NEAREST)
                m_tensor = m_tensor.squeeze(0)

                mask[class_idx] = torch.max(mask[class_idx], m_tensor)

        batch_masks.append(mask)

    return torch.stack(batch_masks)  # [B, C, H, W]



def parse_instance_masks(targets, height, width, category_id_to_class_idx=None):
    batch_masks = []
    batch_classes = []

    for anns in targets:
        masks = []
        classes = []

        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                if category_id_to_class_idx is not None:
                    coco_cat_id = ann['category_id']
                    if coco_cat_id not in category_id_to_class_idx:
                        continue
                    class_idx = category_id_to_class_idx[coco_cat_id]
                else:
                    class_idx = ann['category_id']

                rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
                decoded = coco_mask.decode(rle)  # (H, W) or (H, W, N)

                if decoded.ndim == 3:
                    decoded = np.any(decoded, axis=2)

                mask_tensor = torch.from_numpy(decoded.astype(np.float32))  # [H, W]
                mask_tensor = TF.resize(mask_tensor.unsqueeze(0), size=[height, width], interpolation=InterpolationMode.NEAREST)
                masks.append(mask_tensor.squeeze(0))  # [H, W]
                classes.append(class_idx)

        if masks:
            masks_tensor = torch.stack(masks)  # [N, H, W]
        else:
            masks_tensor = torch.zeros((0, height, width), dtype=torch.float32)

        batch_masks.append(masks_tensor)
        batch_classes.append(classes)

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


# --- Collate function for COCO ---
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
def get_max_category_id(ann_file):
    from pycocotools.coco import COCO
    coco = COCO(ann_file)
    cat_ids = coco.getCatIds()
    return max(cat_ids)

def combined_loss(logits, targets, ce_criterion, dice_criterion):
    assert targets.ndim == 4, f"Expected targets [B, C, H, W], got {targets.shape}"
    ce_targets = torch.argmax(targets, dim=1)  # [B, H, W]
    ce_loss = ce_criterion(logits, ce_targets)
    dice_loss = dice_criterion(logits, targets)
    return 0.7 * ce_loss + 0.3 * dice_loss

def get_max_instances_from_annotations(ann_file):
    coco = COCO(ann_file)
    max_instances = max(len(anns) for anns in coco.imgToAnns.values())
    return max_instances


def log_visual_predictions(config, model, dataloader, num_samples, device, num_classes, category_id_to_class_idx, wandb):
    model.eval()
    assert num_classes > 0, "num_classes must be a positive integer"
    class_colors = plt.cm.get_cmap("tab20", num_classes)
    samples_logged = 0

    with torch.no_grad():
        for images, targets in dataloader:
            assert isinstance(images, torch.Tensor), "Expected images to be a torch.Tensor"
            assert isinstance(targets, list), "Expected targets to be a list of dictionaries"
            images = images.to(device)
            outputs = model(images)
            assert outputs.ndim == 4, "Model output should be 4D: [B, C or N_inst, H, W]"

            preds = torch.argmax(outputs, dim=1).cpu()
            height, width = preds.shape[1:]
            B = images.size(0)
            assert preds.shape[0] == B, "Mismatch in batch size between images and predictions"

            if config.task == "instance":
                for i in range(min(B, num_samples - samples_logged)):
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()
                    pred_mask = preds[i].numpy()

                    inst_masks, class_ids = parse_instance_masks([targets[i]], height, width, category_id_to_class_idx)
                    assert isinstance(inst_masks, list) and isinstance(class_ids, list), "Expected lists from parse_instance_masks"
                    assert inst_masks[0].shape[1:] == (height, width), "Parsed GT mask shape must match prediction size"
                    assert pred_mask.shape == (height, width), "Predicted mask should be 2D for visualization"

                    fig = visualize_instance_batch(img_np, pred_mask, inst_masks[0], class_ids[0], num_classes)
                    wandb.log({f"vis/instance_sample_{samples_logged}": wandb.Image(fig)})
                    plt.close(fig)
                    samples_logged += 1
                    if samples_logged >= num_samples:
                        return

            else:
                gt_masks = parse_segmentation_masks(targets, height, width, num_classes, category_id_to_class_idx)
                assert gt_masks.shape == (B, num_classes, height, width), "Segmentation GT shape mismatch"
                gt_masks = gt_masks.argmax(1).cpu()

                for i in range(min(B, num_samples - samples_logged)):
                    img_np = TF.to_pil_image(images[i].cpu())
                    pred_mask = preds[i].numpy()
                    gt_mask = gt_masks[i].numpy()
                    assert pred_mask.shape == (height, width), "Predicted mask should be 2D"
                    assert gt_mask.shape == (height, width), "Ground truth mask should be 2D"

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
