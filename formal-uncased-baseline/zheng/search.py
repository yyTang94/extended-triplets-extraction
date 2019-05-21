import os
import json

from sklearn.model_selection import ParameterGrid

from main import main


if __name__ == '__main__':
    dataset_folder = '../dataset/conll/'
    record_root = './record/conll/'

    params = {
        # net
        'use_case': [True, True, True, True, True],
        'case_lookup_size': [4],
        'case_embedding_size': [10],

        'drop_prob': [0.5],

        'hidden_size': [50],
        'layer_num': [4],
        'is_bidirection': [True],

        'twice_embedding_size': [10],

        'class_num': [41],

        # fit
        'batch_size': [32],
        'learning_rate': [1e-04],
        'weight_decay': [0.0],

        'balance_weight': [1.0],
        'o_index': [40],

        'patience': [100000],
        'epoch_num': [4000],

        'is_formal': [True]
    }

    hypersets = list(ParameterGrid(params))

    for i in range(len(hypersets)):
        record_folder = record_root + str(i) + '/'
        os.mkdir(record_folder)

    for i_hset, hset in enumerate(hypersets):
        # net hyper
        net_hyper = dict(use_case=hset['use_case'],
                         case_lookup_size=hset['case_lookup_size'],
                         case_embedding_size=hset['case_embedding_size'],
                         drop_prob=hset['drop_prob'],
                         hidden_size=hset['hidden_size'],
                         layer_num=hset['layer_num'],
                         is_bidirection=hset['is_bidirection'],
                         twice_embedding_size=hset['twice_embedding_size'],
                         class_num=hset['class_num'])

        # fit hyper
        fit_hyper = dict(batch_size=hset['batch_size'],
                         learning_rate=hset['learning_rate'],
                         weight_decay=hset['weight_decay'],
                         balance_weight=hset['balance_weight'],
                         o_index=hset['o_index'],
                         class_num=hset['class_num'],
                         patience=hset['patience'],
                         epoch_num=hset['epoch_num'],
                         is_topk=False,
                         is_formal=hset['is_formal'])

        # record folder
        record_folder = record_root + str(i_hset) + '/'

        main(dataset_folder, net_hyper, fit_hyper, record_folder)

    with open(record_root + 'hypersets.json', 'w') as f:
        json.dump(hypersets, f)
