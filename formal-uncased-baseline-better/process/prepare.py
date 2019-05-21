import json
import argparse
import random

import snip
from tag import BILUOTagger

from typing import List, Dict, Any


def _read_from_json(location: str):
    with open(location, 'r') as f:
        return json.load(f)


def _write_to_json(data: Any, location: str):
    with open(location, 'w') as f:
        json.dump(data, f)


def read(folder: str):

    # sentence
    train_sentences = _read_from_json(folder + 'train_sentences.json')
    valid_sentences = _read_from_json(folder + 'valid_sentences.json')
    test_sentences = _read_from_json(folder + 'test_sentences.json')

    word_lookup = _read_from_json(folder + 'word_lookup.json')
    word_embedding = _read_from_json(folder + 'word_embedding.json')

    # case seq
    train_case_seqs = _read_from_json(folder + 'train_case_seqs.json')
    valid_case_seqs = _read_from_json(folder + 'valid_case_seqs.json')
    test_case_seqs = _read_from_json(folder + 'test_case_seqs.json')

    # relation
    train_relations = _read_from_json(folder + 'train_relations.json')
    valid_relations = _read_from_json(folder + 'valid_relations.json')
    test_relations = _read_from_json(folder + 'test_relations.json')

    rtype_lookup = _read_from_json(folder + 'rtype_lookup.json')

    return (train_sentences, valid_sentences, test_sentences,
            word_lookup, word_embedding,
            train_case_seqs, valid_case_seqs, test_case_seqs,
            train_relations, valid_relations, test_relations,
            rtype_lookup)


def write(train_sentences: List[List[str]],
          valid_sentences: List[List[str]],
          test_sentences: List[List[str]],
          word_lookup: List[str],
          word_embedding: List[List[float]],

          train_case_seqs: List[List[int]],
          valid_case_seqs: List[List[int]],
          test_case_seqs: List[List[int]],

          train_tag_seqs: List[List[str]],
          valid_tag_seqs: List[List[str]],
          test_tag_seqs: List[List[str]],
          tag_lookup: List[str],

          folder: str):

    # sentence
    _write_to_json(train_sentences, folder + 'train_sentences.json')
    _write_to_json(valid_sentences, folder + 'valid_sentences.json')
    _write_to_json(test_sentences, folder + 'test_sentences.json')
    _write_to_json(word_lookup, folder + 'word_lookup.json')
    _write_to_json(word_embedding, folder + 'word_embedding.json')

    # write case_seqs
    _write_to_json(train_case_seqs, folder + 'train_case_seqs.json')
    _write_to_json(valid_case_seqs, folder + 'valid_case_seqs.json')
    _write_to_json(test_case_seqs, folder + 'test_case_seqs.json')

    # tag seq
    _write_to_json(train_tag_seqs, folder + 'train_tag_seqs.json')
    _write_to_json(valid_tag_seqs, folder + 'valid_tag_seqs.json')
    _write_to_json(test_tag_seqs, folder + 'test_tag_seqs.json')
    _write_to_json(tag_lookup, folder + 'tag_lookup.json')

def exclude_t_add_o(tag_seqs):
    added_tag_seqs = []

    for tag_seq in tag_seqs:
        added_tag_seq = []
        for tag_bag in tag_seq:
            if len(tag_bag) == 0:
                added_tag_seq.append(["O"])
            else:
                added_tag_seq.append([random.choice(tag_bag)])
        added_tag_seqs.append(added_tag_seq)

    return added_tag_seqs

def main(max_length: int, ready_folder: str,  dataset_folder: str):

    # read
    (train_sentences, valid_sentences, test_sentences,
     word_lookup, word_embedding,
     train_case_seqs, valid_case_seqs, test_case_seqs,
     train_relations, valid_relations, test_relations,
     rtype_lookup) = read(ready_folder)

    # snip
    train_snip_seqs = snip.relation_to_snip_seq(train_relations)
    valid_snip_seqs = snip.relation_to_snip_seq(valid_relations)
    test_snip_seqs = snip.relation_to_snip_seq(test_relations)

    # tag
    tagger = BILUOTagger(max_length)
    tag_lookup = tagger.get_tag_lookup(rtype_lookup)

    train_tag_seqs = tagger.snip_seq_to_tag_seq(train_snip_seqs, tag_lookup)
    valid_tag_seqs = tagger.snip_seq_to_tag_seq(valid_snip_seqs, tag_lookup)
    test_tag_seqs = tagger.snip_seq_to_tag_seq(test_snip_seqs, tag_lookup)

    # add "O" to train
    added_train_tag_seqs = exclude_t_add_o(train_tag_seqs)
    added_valid_tag_seqs = exclude_t_add_o(valid_tag_seqs)

    tag_lookup.append("O")


    # write sentence and tag
    write(train_sentences, valid_sentences, test_sentences,
          word_lookup, word_embedding,
          train_case_seqs, valid_case_seqs, test_case_seqs,
          added_train_tag_seqs, added_valid_tag_seqs, test_tag_seqs,
          tag_lookup,
          dataset_folder)


if __name__ == '__main__':

    # accept arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--ready_folder', type=str,
                        default='../ready/conll/')
    parser.add_argument('--dataset_folder', type=str,
                        default='../dataset/conll/')

    parser.add_argument('--max_length', type=int, default=120)

    args = parser.parse_args()

    # main
    main(args.max_length, args.ready_folder,  args.dataset_folder)
