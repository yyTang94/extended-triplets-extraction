import json

import torch
import numpy as np

from fastai.torch_core import *

from fastai import *
from fastai.vision import *

from myfastai import MyLearner, LearnerCallback


@dataclass
class TrackerCallback(LearnerCallback):
    "A `LearnerCallback` that keeps track of the best value in `monitor`."
    monitor: str = 'val_loss'
    mode: str = 'auto'

    def __post_init__(self):
        super().__post_init__()

        if self.mode not in ['auto', 'min', 'max']:
            warn(
                f'{self.__class__} mode {self.mode} is invalid, falling back to "auto" mode.')
            self.mode = 'auto'
        mode_dict = {'min': np.less, 'max': np.greater}
        mode_dict['auto'] = np.less if 'loss' in self.monitor else np.greater
        self.operator = mode_dict[self.mode]

    def on_train_begin(self, **kwargs: Any)->None:
        self.best = float('inf') if self.operator == np.less else -float('inf')

    def get_monitor_value(self):
        values = {'trn_loss': self.learn.recorder.losses[-1:][0].cpu().numpy(),
                  'val_loss': self.learn.recorder.val_losses[-1:][0]}
        for i, name in enumerate(self.learn.recorder.names[3:]):
            values[name] = self.learn.recorder.metrics[-1:][0][i]
        if values.get(self.monitor) is None:
            warn(
                f'{self.__class__} conditioned on metric `{self.monitor}` which is not available. Available metrics are: {", ".join(map(str, self.learn.recorder.names[1:]))}')
        return values.get(self.monitor)


@dataclass
class EarlyStoppingCallback(TrackerCallback):
    "A `LearnerCallback` that terminates training when monitored quantity stops improving."
    min_delta: int = 0
    patience: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.operator == np.less:
            self.min_delta *= -1

    def on_train_begin(self, **kwargs: Any)->None:
        self.wait = 0
        super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, **kwargs: Any)->None:
        current = self.get_monitor_value()
        if current is None:
            return
        if self.operator(current - self.min_delta, self.best):
            self.best, self.wait = current, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f'Epoch {epoch}: early stopping')
                return True


@dataclass
class SaveModelCallback(TrackerCallback):
    "A `LearnerCallback` that saves the model when monitored quantity is best."
    every: str = 'improvement'
    name: str = 'bestmodel'

    def __post_init__(self):
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs: Any)->None:
        if isinstance(self.every, int):
            if (epoch % self.every) == 0:
                self.learn.save(f'{self.name}_{epoch}')
        else:  # every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                self.best = current
                self.learn.save(f'{self.name}_{self.monitor}')

    def on_train_end(self, **kwargs):
        if self.every == "improvement":
            self.learn.load(f'{self.name}_{self.monitor}')


@dataclass
class SavePredictionCallback(TrackerCallback):
    "A `LearnerCallback` that saves the model when monitored quantity is best."

    record_folder: str = None
    ylookup: List = None
    every: str = 'improvement'

    def __post_init__(self):
        super().__post_init__()

    def on_epoch_begin(self, **kwargs):
        self.outputs = []

    def on_batch_end(self, last_output: Tensor, train, **kwargs):

        if train is False:
            output = torch.exp(torch.nn.functional.logsigmoid(
                last_output)).detach().cpu().numpy()

            self.outputs.append(output)

    def on_epoch_end(self, epoch, train, **kwargs: Any)->None:
        if train is False:
            if isinstance(self.every, int):
                if (epoch % self.every) == 0:
                    ######
                    total_output = np.concatenate(
                        self.outputs, axis=0).tolist()

                    with open(self.record_folder + 'predictions' + '_' + str(epoch) + '.json', 'w') as f:
                        json.dump(total_output, f)
            else:
                current = self.get_monitor_value()
                if current is not None and self.operator(current, self.best):
                    self.best = current

                    ######
                    total_output = np.concatenate(
                        self.outputs, axis=0).tolist()

                    with open(self.record_folder + 'predictions_' + self.monitor + '.json', 'w') as f:
                        json.dump(total_output, f)
