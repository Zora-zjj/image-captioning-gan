import torch
import torch.nn as nn


class RLLoss(nn.Module):

    def forward(self, rewards, props):
        loss = rewards * props
        loss = -torch.sum(loss)     # loss = -求和（rewards * props）
        return loss
