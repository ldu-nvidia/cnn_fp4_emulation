import torch
import numpy as np
from pycocotools import mask as coco_mask
from torchvision.transforms.functional import resize, InterpolationMode

# note relu layer has no weight or bias
block_to_layer_dict = {'0': "conv2d_1st",
                       '1': 'groupNorm_1st',
                       '3': 'conv2d_2nd',
                       '4': 'groupNorm_2nd'}

def rename(old_name):
    new_name = ""
    if 'block' in old_name:
        splitted = old_name.split('.')
        layer, blk_num, type = splitted[0], splitted[-2], splitted[-1]
        new_name = layer + "." + block_to_layer_dict[blk_num]+ "." + type
    else:
        new_name = old_name
    return new_name

# --- Mask/Target Parsing Utilities ---
def parse_segmentation_masks(targets, height, width, num_classes, category_id_to_class_idx):
    batch_masks = []
    for anns in targets:
        mask = torch.zeros((num_classes, height, width), dtype=torch.float32)
        for ann in anns:
            if 'segmentation' not in ann:
                continue

            coco_cat_id = ann['category_id']
            if coco_cat_id not in category_id_to_class_idx:
                continue

            class_idx = category_id_to_class_idx[coco_cat_id]

            rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
            m = coco_mask.decode(rle)  # shape: [height, width] or [height, width, N]

            if m.ndim == 3:
                m = np.any(m, axis=2)  # collapse multiple masks

            m_tensor = torch.tensor(m, dtype=torch.float32)
            mask[class_idx] = torch.max(mask[class_idx], m_tensor)

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

def get_max_category_id(dataset, sample_size=1000):
    max_cat = 0
    for i in range(min(len(dataset), sample_size)):
        _, anns = dataset[i]
        if anns:
            max_cat = max(max_cat, max(ann['category_id'] for ann in anns))
    return max_cat + 1

def combined_loss(logits, targets, ce_criterion, dice_criterion):
    assert targets.ndim == 4, f"Expected targets [B, C, H, W], got {targets.shape}"
    ce_targets = torch.argmax(targets, dim=1)  # [B, H, W]
    ce_loss = ce_criterion(logits, ce_targets)
    dice_loss = dice_criterion(logits, targets)
    return 0.7 * ce_loss + 0.3 * dice_loss