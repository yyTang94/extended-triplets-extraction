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

def _remove_duplicate(triples: List[Dict]):

    shuffle(triples)

    used = set()

    new_triples = []

    for tp in triples:
        sub_id = str(tp['sub_start_ix']) + '-' + str(tp['sub_length'])
        obj_id = str(tp['obj_start_ix']) + '-' + str(tp['obj_length'])

        if sub_id not in used and obj_id not in used:
            new_triples.append(tp)

            used.add(sub_id)
            used.add(obj_id)

    return new_triples

def adjust_relation(relations: List[List[Dict]], max_length: int, is_train):

    new_relations = []

    for triples in relations:

        cutted_triples = _cut_triple(triples, max_length)
        if is_train:
            new_triples = _remove_duplicate(cutted_triples)
        else:
            new_triples = cutted_triples

        new_relations.append(new_triples)

    return new_relations
