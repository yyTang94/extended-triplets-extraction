import json
from typing import List, Dict


def read_embedding(location: str):

    embeddings = dict()

    with open(location, 'r') as f:
        for line in f:
            items = line.rstrip().split(' ')

            w = items[0]
            e = [float(x) for x in items[1:]]

            embeddings[w] = e

    return embeddings


def read_data(folder: str):

    with open(folder + 'train_samples.json', 'r') as f:
        train_samples = json.load(f)

    with open(folder + 'valid_samples.json', 'r') as f:
        valid_samples = json.load(f)

    with open(folder + 'test_samples.json', 'r') as f:
        test_samples = json.load(f)

    return train_samples, valid_samples, test_samples


def extract_sentence_relation(samples: List[Dict]):
    sentences = []
    relations = []

    for s in samples:

        # add sentence to sentences
        sentences.append(s['sentence'])

        # add new_triples to relaitons
        new_triples = []

        for t in s['triples']:
            new_t = dict(sub_start_ix=t['sub_start_ix'],
                         sub_length=t['sub_length'],
                         obj_start_ix=t['obj_start_ix'],
                         obj_length=t['obj_length'],
                         rtype=t['rtype'])
            new_triples.append(new_t)

        relations.append(new_triples)

    return sentences, relations


def run(data_folder: str, pretrained_embedding_location: str):

    # read embedding
    embeddings = read_embedding(pretrained_embedding_location)

    # read data
    train_samples, valid_samples, test_samples = read_data(data_folder)

    # get sentence and triple
    train_sentences, train_relations = extract_sentence_relation(train_samples)
    valid_sentences, valid_relations = extract_sentence_relation(valid_samples)
    test_sentences, test_relations = extract_sentence_relation(test_samples)

    return (train_sentences, train_relations,
            valid_sentences, valid_relations,
            test_sentences, test_relations,
            embeddings)
