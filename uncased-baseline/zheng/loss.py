#  pylint: disable=E1101

from typing import List, Dict


import torch
from torch.nn.functional import nll_loss


class LossFn(object):

    def __init__(self, class_num: int, o_index: int,
                 balance_weight: float, device: torch.device):

        self.balance = torch.ones(class_num) * balance_weight
        self.balance[o_index] = 1.0
        self.balance = self.balance.to(device)

    def __call__(self, score: torch.FloatTensor, yb: torch.FloatTensor):

        score_ = torch.reshape(score, (-1, score.size()[-1]))
        yb_ = torch.reshape(torch.argmax(yb, dim=2), (-1,))
        # yb_ = torch.reshape(yb, (-1,))

        loss = nll_loss(score_, yb_, weight=self.balance)

        return loss

