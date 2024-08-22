import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, smooth=1e-6):

        #Flatten label and prediction tensors
        input = input.view(-1)
        target = target.view(-1)

        intersection = (input * target).sum()
        dice = (2. * intersection + smooth) / (input.sum() + target.sum() + smooth)

        return 1-dice

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, input, target, smooth=1e-6):

        #Flatten label and prediction tensors
        input = input.view(-1)
        target = target.view(-1)

        true_positives = torch.sum(input * target)
        false_negatives = torch.sum((1-input) * target)
        false_positives = torch.sum(input * (1-target))

        tversky = (true_positives + smooth) / (true_positives + self.alpha * false_positives +
                                               (1-self.alpha) * false_negatives + smooth)

        return 1-tversky

class LagrangeLoss(nn.Module):

    def __init__(self, alpha=0.8, threshold=32):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def forward(self, input, target, smooth=1e-6):

        #Flatten label and prediction tensors
        input = input.view(-1)
        target = target.view(-1)

        true_positives = torch.sum(input * target)
        false_negatives = torch.sum((1-input) * target)
        false_positives = torch.sum(input * (1-target))

        tversky = (true_positives + smooth) / (true_positives + self.alpha * false_positives +
                                               (1-self.alpha) * false_negatives + smooth)

        if false_positives < self.threshold:
            penalty = torch.pow((false_positives - self.threshold) / self.threshold, 2)
        else:
            penalty = torch.log(1+false_positives-self.threshold)
        return (1-tversky) + penalty