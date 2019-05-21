#  pylint: disable=E1101

from typing import List, Dict

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.nn.functional import multilabel_soft_margin_loss


class LossFn(object):

    def __init__(self, balance_weight: float):
        self.balance_weight = balance_weight

    def __call__(self, score: torch.FloatTensor, yb_matrix: torch.LongTensor):

        output = torch.sigmoid(score)
        loss = torch.nn.functional.binary_cross_entropy(
            output, yb_matrix.to(torch.float), reduction='none')

        weight_more = yb_matrix.to(torch.float) * self.balance_weight
        weight_plain = 1.0 - yb_matrix.to(torch.float)
        weight = weight_more + weight_plain

        weighted_loss = (loss * weight).mean()

        return weighted_loss
