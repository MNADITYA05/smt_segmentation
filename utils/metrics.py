import numpy as np
import torch
import torch.nn.functional as F


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_miou(output, target, num_classes):
    """Calculate mean IoU"""
    output = F.softmax(output, dim=1)
    pred = output.argmax(dim=1)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    iou_list = []
    for i in range(num_classes):
        pred_i = pred == i
        target_i = target == i
        if target_i.sum() == 0:
            iou_i = float('nan')
        else:
            intersection = (pred_i & target_i).sum()
            union = (pred_i | target_i).sum()
            iou_i = intersection / union
        iou_list.append(iou_i)

    return np.nanmean(iou_list)