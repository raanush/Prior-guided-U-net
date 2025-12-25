import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def boundary_loss(pred, target):
    # Laplacian filter
    laplace_filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                   dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
    pred_edges = F.conv2d(pred, laplace_filter, padding=1)
    target_edges = F.conv2d(target, laplace_filter, padding=1)
    return F.l1_loss(pred_edges, target_edges)


def combined_loss(pred, target, svm_weight=None, weights=(0.4, 0.4, 0.2)):
    """
    Final Q1-level loss combining:
    - Dice Loss
    - Adaptive BCE guided by SVM
    - Boundary Loss
    """
    # پد خروجی و هدف
    pred = pred.to(target.device)
    target = target.to(pred.device)

    # --- Dice Loss
    dice = dice_loss(torch.sigmoid(pred), target)

    # --- Adaptive BCE (با SVM یا بدون آن)
    if svm_weight is not None:
        if svm_weight.shape != pred.shape:
            svm_weight = F.interpolate(svm_weight, size=pred.shape[2:], mode='bilinear', align_corners=False)
        svm_weight = svm_weight.to(pred.device)

        # Adaptive weighting based on confidence
        prob = torch.sigmoid(pred)
        uncertainty = 1 - torch.abs(prob - 0.5) * 2  # uncertainty ∈ [0, 1], high near 0.5
        adaptive_weight = svm_weight * (1 - uncertainty)

        bce = F.binary_cross_entropy(prob, target, reduction='none')
        bce = (bce * adaptive_weight).mean()
    else:
        bce = F.binary_cross_entropy(torch.sigmoid(pred), target)

    # --- Boundary Loss
    bound = boundary_loss(torch.sigmoid(pred), target)

    # --- Combine with weights
    total = weights[0] * dice + weights[1] * bce + weights[2] * bound
    return total

