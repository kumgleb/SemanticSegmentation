import numpy as np
import torch


def predict_mask(model, img):
    model.eval()
    with torch.no_grad():
        mask_prd = model(img)
    mask_prd = mask_prd.argmax(dim=1)
    mask_prd = mask_prd.cpu().numpy()
    return mask_prd


def mask_to_class(mask, n_classes=3):
    """Spit masks for `n_classes` binary masks."""
    bs, w, h = mask.shape
    class_mask = np.zeros((n_classes, bs, w, h))
    for c in range(n_classes):
        idxs = mask == c
        class_mask[c, idxs] = 1
    return class_mask


def score_model(model, val_dataloader, metric, device):
    """Evaluates metric mean and std per class."""
    scores = []
    val_it = iter(val_dataloader)
    for _ in range(len(val_it)):
        x, y = next(val_it)
        x, y = x.to(device), y.to(device)
        pred_mask = predict_mask(model, x)
        y = y.cpu().numpy()
        score = metric(y, pred_mask)
        scores.append(score)
    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)
    return mean_scores, std_scores


def jaccard(gt_mask, prd_mask, smooth=0.1):
    """Calculate Jaccard score per class mean per batch."""
    cm_prd = mask_to_class(prd_mask)
    cm_gt = mask_to_class(gt_mask)

    intersection = np.logical_and(cm_prd, cm_gt).sum(axis=(2, 3))
    union = np.logical_or(cm_prd, cm_gt).sum(axis=(2, 3))

    score = (intersection + smooth) / (union + smooth)
    assert score.max() <= 1
    score = score.mean(axis=1)
    return score


def dice(gt_mask, prd_mask, smooth=0.1):
    """Calculate Dice score per class mean per batch."""
    cm_prd = mask_to_class(prd_mask)
    cm_gt = mask_to_class(gt_mask)

    intersection = np.logical_and(cm_prd, cm_gt).sum(axis=(2, 3))
    sum_ = cm_prd.sum(axis=(2, 3)) + cm_gt.sum(axis=(2, 3))

    score = (2*intersection + smooth) / (sum_ + smooth)
    assert score.max() <= 1
    score = score.mean(axis=1)
    return score


def recall(gt_mask, prd_mask, smooth=0.1):
    """Calculate recall score per class mean per batch."""

    cm_prd = mask_to_class(prd_mask)
    cm_gt = mask_to_class(gt_mask)

    TP = np.logical_and(cm_prd, cm_gt).sum(axis=(2, 3))
    FN = (cm_gt - np.logical_and(cm_prd, cm_gt)*1).sum(axis=(2, 3))

    score = (TP + smooth) / (TP + FN + smooth)
    assert score.max() <= 1
    score = score.mean(axis=1)

    return score
