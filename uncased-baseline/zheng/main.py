import json
import logging
import argparse
from functools import partial
from typing import Dict

import torch
from torch import optim

from fastai import *
from fastai.vision import *

import inout
import convert
import feed
import network
import loss
import metric
from tracktrack import EarlyStoppingCallback
from tracktrack import SaveModelCallback
from tracktrack import SavePredictionCallback
from myfastai import MyLearner


logging.basicConfig(level=logging.DEBUG)


class Debuger(LearnerCallback):
    def __init__(self, learn: Learner):
        super().__init__(learn)

    def on_batch_begin(self, last_input, last_target, **kwargs):
        print(type(last_input))
        print(type(last_target))
        print(last_input)


def main(dataset_folder: str,
         net_hyper: Dict,
         fit_hyper: Dict,
         record_folder: str):

    # read data
    xdata, xlookup, ydata, ylookup, word_embedding = inout.read(dataset_folder)

    # convert
    x = convert.digitize_xdata(xdata, xlookup)
    y = convert.digitize_ydata(
        ydata, ylookup, fit_hyper['class_num'])

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # create data bunch
    data_bunch = feed.create_data_bunch(
        x, y, fit_hyper['batch_size'], device, fit_hyper['is_formal'])

    # create net
    net = network.Net(net_hyper, word_embedding).to(device)

    # loss_fn
    loss_fn = loss.LossFn(class_num=fit_hyper['class_num'],
                          o_index=fit_hyper['o_index'],
                          balance_weight=fit_hyper['balance_weight'],
                          device=device)

    # stop_fn
    stop_fn = partial(EarlyStoppingCallback,
                      monitor='f1calculation', mode='max',
                      patience=fit_hyper['patience'])

    # save fn
    save_model_fn = partial(
        SaveModelCallback, monitor='f1calculation', mode='max')
    # save_model_fn = partial(
    #     SaveModelCallback, every=500)

    # save prediction fn
    save_prediction_fn = partial(SavePredictionCallback,
                                 monitor='f1calculation', mode='max',
                                 ylookup=ylookup['tag_lookup'],
                                 record_folder=record_folder)
    # save_prediction_fn = partial(SavePredictionCallback,
    #                              every=500,
    #                              ylookup=ylookup['tag_lookup'],
    #                              record_folder=record_folder)

    # metric fn
    (accuracy_fn, f1_fn, precision_fn,
     recall_fn) = metric.create_metrics(fit_hyper['o_index'])

    # create learner
    learner = MyLearner(data_bunch, net,
                        opt_func=optim.Adam, loss_func=loss_fn,
                        metrics=[accuracy_fn, f1_fn, precision_fn, recall_fn],
                        true_wd=False, bn_wd=False,
                        wd=fit_hyper['weight_decay'], train_bn=False,
                        path=record_folder, model_dir='models',
                        callback_fns=[stop_fn, save_model_fn,
                                      save_prediction_fn])
    # callbacks=[m_debuger])

    # start fit
    learner.fit(fit_hyper['epoch_num'],
                lr=fit_hyper['learning_rate'])

    # write data
    train_losses = [x.cpu().numpy().tolist()
                    for x in learner.recorder.losses]
    valid_losses = [x.tolist()
                    for x in learner.recorder.val_losses]
    acc_values = [x[0]
                  for x in learner.recorder.metrics]
    prf_values = [[x[1], x[2], x[3]]
                  for x in learner.recorder.metrics]
    inout.write(train_losses, valid_losses,
                acc_values, prf_values, record_folder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # folder
    parser.add_argument('--dataset_folder', type=str,
                        default='../dataset/conll/')
    parser.add_argument('--record_folder', type=str,
                        default='./record/conll/')

    # net hyper
    parser.add_argument('--use_case', type=bool, default=True)
    parser.add_argument('--case_lookup_size', type=int, default=4)
    parser.add_argument('--case_embedding_size', type=int, default=20)

    parser.add_argument('--drop_prob', type=float, default=0.7)

    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--is_bidirection', type=bool, default=True)

    parser.add_argument('--tag_embedding_size', type=int, default=30)

    parser.add_argument('--class_num', type=int, default=40)

    # fit
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=3e-04)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--balance_weight', type=float, default=3.0)
    parser.add_argument('--o_index', type=float, default=None)

    parser.add_argument('--patience', type=int, default=900000)
    parser.add_argument('--epoch_num', type=int, default=4500)

    parser.add_argument('--is_formal', type=bool, default=False)

    # args
    args = parser.parse_args()

    # net hyper
    net_hyper = dict(use_case=args.use_case,
                     case_lookup_size=args.case_lookup_size,
                     case_embedding_size=args.case_embedding_size,
                     drop_prob=args.drop_prob,
                     hidden_size=args.hidden_size,
                     layer_num=args.layer_num,
                     is_bidirection=args.is_bidirection,
                     tag_embedding_size=args.tag_embedding_size,
                     class_num=args.class_num,
                     o_index=args.o_index)

    # fit hyper
    fit_hyper = dict(batch_size=args.batch_size,
                     learning_rate=args.learning_rate,
                     weight_decay=args.weight_decay,
                     balance_weight=args.balance_weight,
                     o_index=args.o_index,
                     class_num=args.class_num,
                     patience=args.patience,
                     epoch_num=args.epoch_num,
                     is_topk=False,
                     is_formal=args.is_formal)

    print(type(fit_hyper['is_topk']))
    print(fit_hyper['is_topk'])

    # do not run from main_py directly
    # main(args.dataset_folder, net_hyper, fit_hyper, args.record_folder)
