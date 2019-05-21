from torch import Tensor

from fastai.torch_core import *
from fastai.callback import *
from fastai import *


@dataclass
class FakeMetric(Callback):

    mid_value: Dict = None
    is_topk: bool = None

    def on_epoch_begin(self, **kwargs):
        self.mid_value['true_num'] = 0

        self.mid_value['pred_01'] = 0
        self.mid_value['tp_01'] = 0

        self.mid_value['pred_02'] = 0
        self.mid_value['tp_02'] = 0

        self.mid_value['pred_03'] = 0
        self.mid_value['tp_03'] = 0

        self.mid_value['pred_04'] = 0
        self.mid_value['tp_04'] = 0

        self.mid_value['pred_05'] = 0
        self.mid_value['tp_05'] = 0

        self.mid_value['pred_06'] = 0
        self.mid_value['tp_06'] = 0

        self.mid_value['pred_07'] = 0
        self.mid_value['tp_07'] = 0

        self.mid_value['pred_08'] = 0
        self.mid_value['tp_08'] = 0

        self.mid_value['pred_09'] = 0
        self.mid_value['tp_09'] = 0

    def on_batch_end(self, last_output: Tensor, last_target: Tensor,
                     train, **kwargs):

        if train is False:
            last_output = torch.exp(
                torch.nn.functional.logsigmoid(last_output))

            if self.is_topk is True:
                # topk mask
                topk_mask = torch.zeros_like(
                    last_output.view(-1, last_output.shape[-1])
                ).to(torch.uint8)

                _, ix_col = torch.topk(
                    last_output.view(-1, last_output.shape[-1]), 2, dim=1)
                ix_row = torch.stack(
                    (torch.arange(topk_mask.shape[0]),
                     torch.arange(topk_mask.shape[0])),
                    dim=1
                )
                topk_mask[ix_row, ix_col] = 1
                topk_mask = torch.reshape(
                    topk_mask, (last_output.shape[0], last_output.shape[1], -1)
                )
            else:
                topk_mask = torch.ones_like(last_output).to(torch.uint8)

            # 01
            th_mask = last_output > 0.1

            total_mask = topk_mask * th_mask

            target = last_target.to(torch.uint8)

            self.mid_value['pred_01'] += total_mask.sum().cpu().numpy().tolist()
            self.mid_value['tp_01'] += torch.mul(
                total_mask, target).sum().cpu().numpy().tolist()

            # 02
            th_mask = last_output > 0.2

            total_mask = topk_mask * th_mask

            target = last_target.to(torch.uint8)

            self.mid_value['pred_02'] += total_mask.sum().cpu().numpy().tolist()
            self.mid_value['tp_02'] += torch.mul(
                total_mask, target).sum().cpu().numpy().tolist()

            # 03
            th_mask = last_output > 0.3

            total_mask = topk_mask * th_mask

            target = last_target.to(torch.uint8)

            self.mid_value['pred_03'] += total_mask.sum().cpu().numpy().tolist()
            self.mid_value['tp_03'] += torch.mul(
                total_mask, target).sum().cpu().numpy().tolist()

            # 04
            th_mask = last_output > 0.4

            total_mask = topk_mask * th_mask

            target = last_target.to(torch.uint8)

            self.mid_value['pred_04'] += total_mask.sum().cpu().numpy().tolist()
            self.mid_value['tp_04'] += torch.mul(
                total_mask, target).sum().cpu().numpy().tolist()

            # 05
            th_mask = last_output > 0.5

            total_mask = topk_mask * th_mask

            target = last_target.to(torch.uint8)

            self.mid_value['pred_05'] += total_mask.sum().cpu().numpy().tolist()
            self.mid_value['tp_05'] += torch.mul(
                total_mask, target).sum().cpu().numpy().tolist()

            # 06
            th_mask = last_output > 0.6

            total_mask = topk_mask * th_mask

            target = last_target.to(torch.uint8)

            self.mid_value['pred_06'] += total_mask.sum().cpu().numpy().tolist()
            self.mid_value['tp_06'] += torch.mul(
                total_mask, target).sum().cpu().numpy().tolist()

            # 07
            th_mask = last_output > 0.7

            total_mask = topk_mask * th_mask

            target = last_target.to(torch.uint8)

            self.mid_value['pred_07'] += total_mask.sum().cpu().numpy().tolist()
            self.mid_value['tp_07'] += torch.mul(
                total_mask, target).sum().cpu().numpy().tolist()

            # 08
            th_mask = last_output > 0.8

            total_mask = topk_mask * th_mask

            target = last_target.to(torch.uint8)

            self.mid_value['pred_08'] += total_mask.sum().cpu().numpy().tolist()
            self.mid_value['tp_08'] += torch.mul(
                total_mask, target).sum().cpu().numpy().tolist()

            # 09
            th_mask = last_output > 0.9

            total_mask = topk_mask * th_mask

            target = last_target.to(torch.uint8)

            self.mid_value['pred_09'] += total_mask.sum().cpu().numpy().tolist()
            self.mid_value['tp_09'] += torch.mul(
                total_mask, target).sum().cpu().numpy().tolist()

            # true
            target = last_target.to(torch.uint8)
            self.mid_value['true_num'] += target.sum().cpu().numpy().tolist()

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            self.metric = 0.0


@dataclass
class Precision01(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_01'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_01'] /
                               self.mid_value['pred_01'])


@dataclass
class Recall01(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_01'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_01'] /
                               self.mid_value['true_num'])


@dataclass
class F1value01(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_01'] == 0:
                self.metric = 0.0
            else:
                precision = (self.mid_value['tp_01'] /
                             self.mid_value['pred_01'])
                recall = (self.mid_value['tp_01'] /
                          self.mid_value['true_num'])
                self.metric = 2 * precision * recall / (precision + recall)


@dataclass
class Precision02(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_02'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_02'] /
                               self.mid_value['pred_02'])


@dataclass
class Recall02(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_02'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_02'] /
                               self.mid_value['true_num'])


@dataclass
class F1value02(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_02'] == 0:
                self.metric = 0.0
            else:
                precision = (self.mid_value['tp_02'] /
                             self.mid_value['pred_02'])
                recall = (self.mid_value['tp_02'] /
                          self.mid_value['true_num'])
                self.metric = 2 * precision * recall / (precision + recall)


@dataclass
class Precision03(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_03'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_03'] /
                               self.mid_value['pred_03'])


@dataclass
class Recall03(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_03'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_03'] /
                               self.mid_value['true_num'])


@dataclass
class F1value03(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_03'] == 0:
                self.metric = 0.0
            else:
                precision = (self.mid_value['tp_03'] /
                             self.mid_value['pred_03'])
                recall = (self.mid_value['tp_03'] /
                          self.mid_value['true_num'])
                self.metric = 2 * precision * recall / (precision + recall)


@dataclass
class Precision04(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_04'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_04'] /
                               self.mid_value['pred_04'])


@dataclass
class Recall04(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_04'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_04'] /
                               self.mid_value['true_num'])


@dataclass
class F1value04(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_04'] == 0:
                self.metric = 0.0
            else:
                precision = (self.mid_value['tp_04'] /
                             self.mid_value['pred_04'])
                recall = (self.mid_value['tp_04'] /
                          self.mid_value['true_num'])
                self.metric = 2 * precision * recall / (precision + recall)


@dataclass
class Precision05(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_05'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_05'] /
                               self.mid_value['pred_05'])


@dataclass
class Recall05(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_05'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_05'] /
                               self.mid_value['true_num'])


@dataclass
class F1value05(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_05'] == 0:
                self.metric = 0.0
            else:
                precision = (self.mid_value['tp_05'] /
                             self.mid_value['pred_05'])
                recall = (self.mid_value['tp_05'] /
                          self.mid_value['true_num'])
                self.metric = 2 * precision * recall / (precision + recall)


@dataclass
class Precision06(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_06'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_06'] /
                               self.mid_value['pred_06'])


@dataclass
class Recall06(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_06'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_06'] /
                               self.mid_value['true_num'])


@dataclass
class F1value06(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_06'] == 0:
                self.metric = 0.0
            else:
                precision = (self.mid_value['tp_06'] /
                             self.mid_value['pred_06'])
                recall = (self.mid_value['tp_06'] /
                          self.mid_value['true_num'])
                self.metric = 2 * precision * recall / (precision + recall)


@dataclass
class Precision07(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_07'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_07'] /
                               self.mid_value['pred_07'])


@dataclass
class Recall07(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_07'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_07'] /
                               self.mid_value['true_num'])


@dataclass
class F1value07(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_07'] == 0:
                self.metric = 0.0
            else:
                precision = (self.mid_value['tp_07'] /
                             self.mid_value['pred_07'])
                recall = (self.mid_value['tp_07'] /
                          self.mid_value['true_num'])
                self.metric = 2 * precision * recall / (precision + recall)


@dataclass
class Precision08(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_08'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_08'] /
                               self.mid_value['pred_08'])


@dataclass
class Recall08(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_08'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_08'] /
                               self.mid_value['true_num'])


@dataclass
class F1value08(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_08'] == 0:
                self.metric = 0.0
            else:
                precision = (self.mid_value['tp_08'] /
                             self.mid_value['pred_08'])
                recall = (self.mid_value['tp_08'] /
                          self.mid_value['true_num'])
                self.metric = 2 * precision * recall / (precision + recall)


@dataclass
class Precision09(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_09'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_09'] /
                               self.mid_value['pred_09'])


@dataclass
class Recall09(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_09'] == 0:
                self.metric = 0.0
            else:
                self.metric = (self.mid_value['tp_09'] /
                               self.mid_value['true_num'])


@dataclass
class F1value09(Callback):

    mid_value: Dict = None

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if self.mid_value['tp_09'] == 0:
                self.metric = 0.0
            else:
                precision = (self.mid_value['tp_09'] /
                             self.mid_value['pred_09'])
                recall = (self.mid_value['tp_09'] /
                          self.mid_value['true_num'])
                self.metric = 2 * precision * recall / (precision + recall)


def create_metrics(is_topk: bool):
    middle_value = dict(true_num=0)

    fake_metric = FakeMetric(middle_value, is_topk)

    precision_01 = Precision01(middle_value)
    recall_01 = Recall01(middle_value)
    f1value_01 = F1value01(middle_value)

    precision_02 = Precision02(middle_value)
    recall_02 = Recall02(middle_value)
    f1value_02 = F1value02(middle_value)

    precision_03 = Precision03(middle_value)
    recall_03 = Recall03(middle_value)
    f1value_03 = F1value03(middle_value)

    precision_04 = Precision04(middle_value)
    recall_04 = Recall04(middle_value)
    f1value_04 = F1value04(middle_value)

    precision_05 = Precision05(middle_value)
    recall_05 = Recall05(middle_value)
    f1value_05 = F1value05(middle_value)

    precision_06 = Precision06(middle_value)
    recall_06 = Recall06(middle_value)
    f1value_06 = F1value06(middle_value)

    precision_07 = Precision07(middle_value)
    recall_07 = Recall07(middle_value)
    f1value_07 = F1value07(middle_value)

    precision_08 = Precision08(middle_value)
    recall_08 = Recall08(middle_value)
    f1value_08 = F1value08(middle_value)

    precision_09 = Precision09(middle_value)
    recall_09 = Recall09(middle_value)
    f1value_09 = F1value09(middle_value)

    return (fake_metric,
            precision_01, recall_01, f1value_01,
            precision_02, recall_02, f1value_02,
            precision_03, recall_03, f1value_03,
            precision_04, recall_04, f1value_04,
            precision_05, recall_05, f1value_05,
            precision_06, recall_06, f1value_06,
            precision_07, recall_07, f1value_07,
            precision_08, recall_08, f1value_08,
            precision_09, recall_09, f1value_09)
