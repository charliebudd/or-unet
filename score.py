import torch

def dice_score(input, target):
    SMOOTH = 1
    intersection = torch.logical_and(input, target).sum(dim=(1, 2))
    input_count = input.sum(dim=(1, 2))
    target_count = target.sum(dim=(1, 2))
    return (2 * intersection + SMOOTH) / (input_count + target_count + SMOOTH)  

def iou_score(input, target):
    SMOOTH = 1
    intersection = torch.logical_and(input, target).sum(dim=(1, 2))
    union = torch.logical_or(input, target).sum(dim=(1, 2))
    return (intersection + SMOOTH) / (union + SMOOTH)  
