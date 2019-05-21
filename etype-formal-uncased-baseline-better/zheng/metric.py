from torch import Tensor
import torch

from fastai.torch_core import *
from fastai.callback import *
from fastai import *
from fastai.metrics import *


@dataclass
class F1calculation(Callback):

    mid_value: Dict = None
    o_index: int = None

    def on_epoch_begin(self, **kwargs):
        self.mid_value['true_num'] = 0
        self.mid_value['pred_num'] = 0
        self.mid_value['tp_num'] = 0

    def on_batch_end(self, last_output: Tensor, last_target: Tensor,
                     train, **kwargs):

        if train is False:
            reshaped_output_score = torch.reshape(last_output.detach().cpu(), (-1, last_output.size()[2]))
            reshaped_target_matrix = torch.reshape(last_target.detach().cpu(), (-1, last_target.size()[2]))

            ixes = torch.argmax(reshaped_output_score, dim=1)
            new_ixes = torch.stack((torch.arange(reshaped_output_score.size()[0]), ixes))

            reshaped_output_matrix = torch.zeros_like(reshaped_output_score, dtype=torch.uint8)
            reshaped_output_matrix[new_ixes.numpy().tolist()] = 1
            reshaped_output_matrix[:, -1] = 0

            reshaped_target_matrix = reshaped_target_matrix.to(torch.uint8)

            cur_true_num = reshaped_target_matrix.sum().numpy().tolist()
            cur_pred_num = reshaped_output_matrix.sum().numpy().tolist()
            cur_tp_num = torch.mul(reshaped_output_matrix, reshaped_target_matrix).sum().numpy().tolist()

            self.mid_value['true_num'] += cur_true_num
            self.mid_value['pred_num'] += cur_pred_num
            self.mid_value['tp_num'] += cur_tp_num


    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_num'] == 0:
                self.metric = 0.0
            else:
                precision = (self.mid_value['tp_num'] /
                             self.mid_value['pred_num'])
                recall = (self.mid_value['tp_num'] /
                          self.mid_value['true_num'])
                self.metric = 2 * precision * recall / (precision + recall)


@dataclass
class Precision(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_num'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_num'] /
                               self.mid_value['pred_num'])


@dataclass
class Recall(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_num'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_num'] /
                               self.mid_value['true_num'])


class ChangedAverageMetric(Callback):
    def __init__(self, func):
        self.func, self.name = func, func.__name__

    def on_epoch_begin(self, **kwargs):
        self.val, self.count = 0., 0

    def on_batch_end(self, last_output, last_target, train, **kwargs):
        if train is False:
            pass

            # last_output_ = torch.reshape(
            #     last_output, (-1, last_output.size()[-1]))
            # last_target_ = torch.reshape(last_target, (-1,))

            #
            # self.count += last_target_.size(0)
            # self.val += last_target_.size(0) * self.func(last_output_,
            #                                              last_target_).detach().item()

    def on_epoch_end(self, train, **kwargs):
        if train is False:
            # self.metric = self.val/self.count
            self.metric = 0.0


def create_metrics(o_index: int):
    accuracy_fn = ChangedAverageMetric(accuracy)

    middle_value = dict(true_num=0, pred_num=0, tp_num=0)

    f1_fn = F1calculation(middle_value, o_index)
    precision_fn = Precision(middle_value)
    recall_fn = Recall(middle_value)

    return accuracy_fn, f1_fn, precision_fn, recall_fn
