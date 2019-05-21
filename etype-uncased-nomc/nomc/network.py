# pylint: disable=E1101

import json
from typing import List, Dict, Tuple

import torch
from torch import nn
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self, net_hyper: Dict, word_embedding: List[List[float]]):
        nn.Module.__init__(self)

        self.net_hyper = net_hyper

        # word embedding
        init_word_embedding = torch.FloatTensor(word_embedding)
        word_lookup_size = init_word_embedding.size()[0]
        word_embedding_size = init_word_embedding.size()[1]

        self.word_embedding_layer = nn.Embedding(word_lookup_size,
                                                 word_embedding_size)
        self.word_embedding_layer.weight = nn.Parameter(init_word_embedding)

        # case embedding
        self.use_case = net_hyper['use_case']
        if self.use_case:
            self.case_embedding_layer = nn.Embedding(
                net_hyper['case_lookup_size'],
                net_hyper['case_embedding_size'])

        # tag embedding
        # self.tag_embedding_layer = nn.Embedding(
        #     net_hyper['class_num'], net_hyper['tag_embedding_size'])

        # dropout
        self.dropout_layer = nn.Dropout(net_hyper['drop_prob'])

        # gru
        if self.use_case:
            gru_input_size = (word_embedding_size +
                              net_hyper['case_embedding_size'])
        else:
            gru_input_size = word_embedding_size
        self.gru_layer = nn.GRU(input_size=gru_input_size,
                                hidden_size=net_hyper['hidden_size'],
                                num_layers=net_hyper['layer_num'],
                                bidirectional=net_hyper['is_bidirection'],
                                batch_first=True)

        # rnn cell
        if net_hyper['is_bidirection']:
            rnn_hidden_size = net_hyper['hidden_size'] * 2
        else:
            rnn_hidden_size = net_hyper['hidden_size']

        # self.rnn_cell = nn.RNNCell(input_size=net_hyper['tag_embedding_size'],
        #                            hidden_size=rnn_hidden_size)

        # predict
        self.tag_layer = nn.Linear(rnn_hidden_size,
                                   net_hyper['class_num'])

        # self.logit_layer = nn.LogSoftmax(dim=2)
        ##################
        self.buffer_layer = nn.Linear(rnn_hidden_size,
                                      rnn_hidden_size)
        ##################

    def forward(self, sentences: torch.LongTensor, case_seqs: torch.LongTensor):

        # word embedding
        word_embeds = self.word_embedding_layer(sentences)

        # case embedding
        if self.use_case:
            case_embeds = self.case_embedding_layer(case_seqs)

            embeds = torch.cat((word_embeds, case_embeds), dim=2)
        else:
            embeds = word_embeds

        # dropout
        dropt = self.dropout_layer(embeds)

        # gru
        gru_out, _ = self.gru_layer(dropt)

        # out put
        output = self.tag_layer(gru_out)

        return output
