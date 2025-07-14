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


class instance_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_masks, gt_masks):
        # Validate input types and shapes
        assert isinstance(pred_masks, torch.Tensor), "pred_masks must be a tensor"
        assert isinstance(gt_masks, list), "gt_masks must be a list of tensors"
        assert pred_masks.ndim == 4, f"Expected pred_masks shape [B,C,H,W], got {pred_masks.shape}"

        B, C, H, W = pred_masks.shape
        for i, gt in enumerate(gt_masks):
            assert isinstance(gt, torch.Tensor), f"gt_masks[{i}] must be a tensor"
            assert gt.ndim == 3, f"Each gt_mask[{i}] should have shape [N,H,W], got {gt.shape}"
            assert gt.shape[1:] == (H, W), f"gt_masks[{i}] spatial dims {gt.shape[1:]} != pred spatial dims {(H,W)}"

        total_loss = 0.0
        valid_samples = 0

        for idx, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
            # pred: [C,H,W], gt: [N,H,W]
            assert pred.ndim == 3, f"Expected pred shape [C,H,W], got {pred.shape}"
            N_gt = gt.shape[0]
            if N_gt == 0 or C == 0:
                continue

            # Expand gt masks to match pred channels
            # BCE expects [C,H,W] vs [C,H,W]
            # We'll compute BCE between each GT instance and all predicted logits
            gt = gt.to(pred.device)
            bce_losses = []
            for i in range(N_gt):
                gt_i = gt[i].unsqueeze(0).expand(C, H, W)  # shape [C,H,W]
                bce_loss = F.binary_cross_entropy_with_logits(pred, gt_i)
                bce_losses.append(bce_loss)

            # Take min loss across GTs for this prediction
            total_loss += torch.stack(bce_losses).min()
            valid_samples += 1

        # Normalize
        avg_loss = total_loss / max(valid_samples, 1)
        return avg_loss
