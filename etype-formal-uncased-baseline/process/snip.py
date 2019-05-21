import re
import sys
from random import shuffle
from itertools import combinations
from typing import List, Dict
from typing import List, Dict, Any, Callable


from util import find_nearest_element, find_nearest_pair


def relation_to_snip_seq_once(triples: List[Dict]):

    snip_seq = []

    for tp in triples:
        sub_sp = dict(start_ix=tp['sub_start_ix'],
                      length=tp['sub_length'],
                      etype=tp['sub_etype'],
                      rtype=tp['rtype'],
                      role='sub')
        obj_sp = dict(start_ix=tp['obj_start_ix'],
                      length=tp['obj_length'],
                      etype=tp['obj_etype'],
                      rtype=tp['rtype'],
                      role='obj')

        snip_seq.append(sub_sp)
        snip_seq.append(obj_sp)

    return snip_seq


def relation_to_snip_seq(relations: List[List[Dict]]):

    snip_seqs = []

    for rl in relations:

        snip_seq = relation_to_snip_seq_once(rl)  # rl is triples
        snip_seqs.append(snip_seq)

    return snip_seqs


def make_candicate(snip_seq: List[Dict]):

    candicate_by_rtype = dict()

    for sp in snip_seq:
        rtype = sp['rtype']
        role = sp['role']
        start_ix = sp['start_ix']
        length = sp['length']

        if rtype in candicate_by_rtype:
            candicate_by_rtype[rtype][role].append((start_ix, length))
        else:
            candicate_by_rtype[rtype] = dict(sub=[], obj=[])

            candicate_by_rtype[rtype][role].append((start_ix, length))

    return candicate_by_rtype


def _distance_first_fn(candicate_by_rtype: Dict[str, Dict[str, List]]):

    triples = []

    for rtype in candicate_by_rtype:
        sub_candicates = candicate_by_rtype[rtype]['sub']
        obj_candicates = candicate_by_rtype[rtype]['obj']

        sub_masks = [True] * len(sub_candicates)
        obj_masks = [True] * len(obj_candicates)

        while any(sub_masks) and any(obj_masks):
            which_sub, which_obj, _ = find_nearest_pair(sub_candicates,
                                                        sub_masks,
                                                        obj_candicates,
                                                        obj_masks)
            if which_sub == -1 or which_obj == -1:
                break

            sub = sub_candicates[which_sub]
            obj = obj_candicates[which_obj]

            tp = dict(sub_start_ix=sub[0], sub_length=sub[1],
                      obj_start_ix=obj[0], obj_length=obj[1],
                      rtype=rtype)
            triples.append(tp)

            # update masks
            sub_masks[which_sub] = False
            obj_masks[which_obj] = False

    return triples


def snip_seq_to_relation(snip_seqs: List[List[Dict]], algorithm: str):

    methods = {'random': '_random_fn',
               'sub-first': '_sub_first_fn',
               'obj-first': '_obj_first_fn',
               'order-first': '_order_first_fn',
               'distance-first': _distance_first_fn}

    generate_relation_fn = methods[algorithm]
    relations = []

    for snip_seq in snip_seqs:

        # make candidates by type
        candidate_by_rtype = make_candicate(snip_seq)

        # generate relations
        relation = generate_relation_fn(candidate_by_rtype)

        # append to relations
        relations.append(relation)

    return relations
