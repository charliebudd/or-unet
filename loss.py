import torch
from torch.nn import Module, AvgPool2d, Softmax2d, CrossEntropyLoss, MSELoss

class OrUnetLoss(Module):
    def __init__(self):
        super().__init__()
        self.pool = AvgPool2d(kernel_size=2)
        self.primary_loss = PrimaryLoss()
        self.secondary_loss = SecondaryLoss()
    
    def forward(self, outputs, target):

        output_count = len(outputs)

        loss_functions = [self.primary_loss] + (output_count - 1) * [self.secondary_loss]
        loss_coefficients = [1.0/2**i for i in range(output_count)]

        targets = [target]
        for _ in range(output_count - 1):
            target = self.pool(target)
            targets.append(target)

        losses = [coef * func(out, targ) for coef, func, out, targ in zip (loss_coefficients, loss_functions, outputs, targets)]
        loss = sum(losses) / sum(loss_coefficients)
        
        return loss


class PrimaryLoss(Module):
    def __init__(self):
        super().__init__()
        self.sm = Softmax2d()
        self.dice = DiceLoss()
        self.ce = CrossEntropyLoss()

    def forward(self, input, target):
        ce_loss = self.ce(input, torch.argmax(target, dim=1))
        input = self.sm(input)
        dice_loss = self.dice(input, target)
        return ce_loss + dice_loss


class SecondaryLoss(Module):
    def __init__(self):
        super().__init__()
        self.sm = Softmax2d()
        self.mse = MSELoss()
        self.soft_dice = SoftDiceLoss()

    def forward(self, input, target):
        input = self.sm(input)
        mse_loss = self.mse(input, target)
        soft_dice_loss = self.soft_dice(input, target)
        return mse_loss + soft_dice_loss


SMOOTH = 1

class DiceLoss(Module):
    def forward(self, input, target):
        intersection = (input * target).sum()
        return 1 - (2 * intersection + SMOOTH) / (input.sum() + target.sum() + SMOOTH)  

class SoftDiceLoss(Module):
    def forward(self, input, target):
        tp = torch.clamp_min(target - torch.abs(input - target), 0).sum()
        fp = torch.clamp_min(input - target, 0).sum()
        fn = torch.clamp_min(target - input, 0).sum()
        return 1 - (2 * tp + SMOOTH) / (2 * tp + fp + fn + SMOOTH)
