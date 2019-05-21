import re
from random import shuffle
from itertools import combinations
from typing import List, Dict


def _cut_triple(triples: List[Dict], max_length: int):

    cutted_triples = []

    for t in triples:
        if (t['sub_start_ix'] + t['sub_length'] <= max_length and
                t['obj_start_ix'] + t['obj_length'] <= max_length):
            cutted_triples.append(t)

    return cutted_triples


def adjust_relation(relations: List[List[Dict]], max_length: int):

    new_relations = []

    for triples in relations:

        cutted_triples = _cut_triple(triples, max_length)
        # new_triples = _remove_duplicate(cutted_triples)

        new_relations.append(cutted_triples)

    return new_relations
