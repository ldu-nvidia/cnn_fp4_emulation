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
    """Hungarian-matched BCE loss for instance segmentation.

    Arguments
    ---------
    pred_masks : Tensor  [B, C, H, W]   – raw logits per instance-channel.
    gt_masks   : List[Tensor[N_i, H, W]] – binary GT masks for each image.

    Behaviour
    ---------
    • Computes the BCE loss for every (channel, GT) pair.
    • Uses Hungarian assignment to find the minimal-cost one-to-one matching.
      Unmatched GTs/preds are ignored (could be penalised separately).
    • Averages the matched losses over the batch and returns a scalar.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_masks: torch.Tensor, gt_masks: list[torch.Tensor]):
        # ---- Input validation -------------------------------------------------
        assert pred_masks.ndim == 4, "pred_masks must be [B,C,H,W]"
        assert isinstance(gt_masks, list), "gt_masks must be a list of tensors"

        B, C, H, W = pred_masks.shape
        assert len(gt_masks) == B, "len(gt_masks) must equal batch size"

        from scipy.optimize import linear_sum_assignment  # local import (SciPy already used elsewhere)

        device = pred_masks.device
        total_loss = 0.0
        valid_imgs = 0

        for preds_per_img, gts_per_img in zip(pred_masks, gt_masks):
            # preds_per_img: [C,H,W]; gts_per_img: [N,H,W]
            if gts_per_img.ndim != 3:
                raise ValueError("Each element in gt_masks must be [N,H,W]")

            N = gts_per_img.shape[0]
            if N == 0:
                # No GT instances – treat all predictions as background (target = 0)
                bg_target = torch.zeros_like(preds_per_img)
                loss_bg = F.binary_cross_entropy_with_logits(preds_per_img, bg_target)
                total_loss += loss_bg
                valid_imgs += 1
                continue

            # -------- pairwise BCE cost matrix (vectorised) --------------------
            # Flatten spatial dims to P = H*W for memory-efficient ops
            preds_flat = preds_per_img.view(C, -1)          # [C, P]
            gts_flat   = gts_per_img.to(device).view(N, -1) # [N, P]

            cost = torch.empty((C, N), device=device)
            # Loop over the smaller dimension to keep memory low
            if N <= C:
                # iterate over GT masks (typically fewer)
                for n in range(N):
                    tgt = gts_flat[n].unsqueeze(0).expand(C, -1)  # [C,P]
                    cost[:, n] = F.binary_cross_entropy_with_logits(
                        preds_flat, tgt, reduction="none"
                    ).mean(dim=1)
            else:
                for c in range(C):
                    pred_row = preds_flat[c].unsqueeze(0).expand(N, -1)  # [N,P]
                    cost[c] = F.binary_cross_entropy_with_logits(
                        pred_row, gts_flat, reduction="none"
                    ).mean(dim=1)

            # Hungarian assignment expects CPU numpy array
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

            matched_losses = cost[row_ind, col_ind]
            if matched_losses.numel() > 0:
                total_loss += matched_losses.mean()
                valid_imgs += 1

        return total_loss / max(valid_imgs, 1)
