import torch
import torch.nn as nn
import numpy as np

class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def masked_mae_loss(self, preds, labels, null_val=0.0):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = labels != null_val
        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds - labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    def __call__(self, preds, labels, null_val=0.0):
        return self.masked_mae_loss(preds, labels, null_val)

class LossFusion:
    def __init__(self, loss_type="huber", x_loss_weight=1.0):
        if loss_type == "huber":
            self.x_loss = nn.HuberLoss()
            self.y_loss = nn.HuberLoss()
        elif loss_type == "masked_mae":
            self.x_loss = MaskedMAELoss()
            self.y_loss = MaskedMAELoss()
        else:
            raise ValueError("No such loss type.")

        self.x_loss_weight = x_loss_weight

    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, y_preds, y_labels, x_preds=None, x_labels=None):
        y_loss = self.y_loss(y_preds, y_labels)

        if x_preds is not None and x_preds.numel() != 0:
            x_loss = self.x_loss(x_preds, x_labels) * self.x_loss_weight
            return y_loss + x_loss, y_loss, x_loss
        else:
            return y_loss, y_loss, torch.tensor(0., requires_grad=True)






