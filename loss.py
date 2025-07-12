import torch
import torch.nn as nn
import torch.nn.functional as F

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


def instance_loss(pred_masks, gt_masks):
    assert isinstance(pred_masks, torch.Tensor), "pred_masks must be a tensor"
    assert isinstance(gt_masks, list), "gt_masks must be a list of tensors"
    assert pred_masks.ndim == 4, "pred_masks should have shape [B, N_pred, H, W]"
    for gt in gt_masks:
        assert isinstance(gt, torch.Tensor), "Each ground truth in gt_masks must be a tensor"
        assert gt.ndim == 3, "Each gt mask should have shape [N_gt, H, W]"
        assert gt.shape[1:] == pred_masks.shape[2:], f"GT mask shape {gt.shape[1:]} must match prediction shape {pred_masks.shape[2:]}"

    loss = 0.0
    count = 0
    for pred, gt in zip(pred_masks, gt_masks):
        N_gt = gt.shape[0]
        N_pred = pred.shape[0]
        if N_gt == 0 or N_pred == 0:
            continue
        gt = gt.to(pred.device)
        losses = []
        for i in range(N_gt):
            expanded_gt = gt[i].unsqueeze(0).expand_as(pred)
            losses.append(F.binary_cross_entropy_with_logits(pred, expanded_gt))
        loss += min(losses)
        count += 1
    return loss / max(count, 1)

