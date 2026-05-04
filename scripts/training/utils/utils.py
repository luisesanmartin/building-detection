import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

dice_loss_fn = smp.losses.DiceLoss(mode='binary')


def compute_loss(logits, masks):
    masks = masks.float().unsqueeze(1)
    bce  = F.binary_cross_entropy_with_logits(logits, masks)
    dice = dice_loss_fn(logits, masks)
    return bce + dice


def compute_iou(logits, masks):
    preds = (logits.sigmoid() > 0.5).long()
    masks = masks.long().unsqueeze(1)
    tp, fp, fn, tn = smp.metrics.get_stats(preds, masks, mode='binary')
    return smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')


def run_epoch(model, loader, optimizer, device: str, train: bool):
    model.train() if train else model.eval()
    total_loss = total_iou = 0.0

    with (torch.enable_grad() if train else torch.no_grad()):
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)
            logits = model(images)
            loss   = compute_loss(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_iou  += compute_iou(logits, masks).item()

    n = len(loader)
    return total_loss / n, total_iou / n
