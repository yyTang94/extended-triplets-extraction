import sys
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from fastai import *
from fastai.vision import *


class REDataset(Dataset):

    def __init__(self,
                 sentences: List[List[str]],
                 case_seqs: List[List[int]],
                 tag_matrix: List[List[List[int]]]):
        self.sentences = sentences
        self.case_seqs = case_seqs
        self.tag_matrix = tag_matrix

        self.loss_func = None  # no use, fix the bug of fastai

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, ix):
        return (self.sentences[ix], self.case_seqs[ix], self.tag_matrix[ix])


def collate_fn(samples: List[Tuple[List[int], List[int]]]):

    sentences = []
    case_seqs = []
    tag_matrix = []

    for s in samples:
        sentences.append(s[0])
        case_seqs.append(s[1])
        tag_matrix.append(s[2])

    sentences_ = torch.LongTensor(sentences)
    case_seqs_ = torch.LongTensor(case_seqs)
    tag_matrix_ = torch.LongTensor(tag_matrix)

    return ((sentences_, case_seqs_), tag_matrix_)


def create_data_bunch(x: Dict, y: Dict,
                      batch_size: int, device: torch.device,
                      is_formal: bool):

    # x
    train_sentences = x['train_sentences']
    valid_sentences = x['valid_sentences']
    test_sentences = x['test_sentences']

    train_case_seqs = x['train_case_seqs']
    valid_case_seqs = x['valid_case_seqs']
    test_case_seqs = x['test_case_seqs']

    # y
    train_tag_matrix = y['train_tag_matrix']
    valid_tag_matrix = y['valid_tag_matrix']
    test_tag_matrix = y['test_tag_matrix']

    # dataset
    if is_formal is False:
        train_ds = REDataset(
            train_sentences, train_case_seqs, train_tag_matrix)
        valid_ds = REDataset(
            valid_sentences, valid_case_seqs, valid_tag_matrix)
        test_ds = REDataset(test_sentences, test_case_seqs, test_tag_matrix)
    else:
        train_ds = REDataset(train_sentences + valid_sentences,
                             train_case_seqs + valid_case_seqs,
                             train_tag_matrix + valid_tag_matrix)
        valid_ds = REDataset(test_sentences, test_case_seqs, test_tag_matrix)
        test_ds = REDataset(test_sentences, test_case_seqs, test_tag_matrix)

    # dl
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True)
    valid_dl = DataLoader(valid_ds,
                          batch_size=batch_size,
                          shuffle=False)
    test_dl = DataLoader(test_ds,
                         batch_size=batch_size,
                         collate_fn=collate_fn,
                         shuffle=False)

    # ddl
    # train_ddl = DeviceDataLoader(train_dl, device, collate_fn=collate_fn)
    # valid_ddl = DeviceDataLoader(valid_dl, device, collate_fn=collate_fn)
    # test_ddl = DeviceDataLoader(test_dl, device, collate_fn=collate_fn)

    # data bunch
    # data_bunch = DataBunch(train_ddl, valid_ddl, test_ddl)
    data_bunch = DataBunch(train_dl, valid_dl, test_dl=test_dl,
                           device=device, collate_fn=collate_fn)

    return data_bunch
