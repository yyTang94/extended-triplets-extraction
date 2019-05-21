import sys
from typing import List, Dict, Any


def _distance_function(a0, a1, b0, b1):
    return abs(max(a0, b0) - min(a0+a1, b0+b1))


def find_nearest_pair(al: List[Any], am: List[bool],
                      bl: List[Any], bm: List[bool]):

    shortest_dist = sys.maxsize
    hit_a_ix = -1
    hit_b_ix = -1

    for a_ix, (a, a_mask) in enumerate(zip(al, am)):
        if a_mask is True:
            b_ix, cur_dist = find_nearest_element(a, bl, bm)

            if b_ix != -1 and cur_dist < shortest_dist:
                hit_a_ix = a_ix
                hit_b_ix = b_ix

                # update shortest distance
                shortest_dist = cur_dist

    return hit_a_ix, hit_b_ix, shortest_dist

def find_nearest_element(a: Any, bl: List[Any], bm: List[bool]):

    shortest_dist = sys.maxsize
    hit_ix = -1

    for ix, (b, mask) in enumerate(zip(bl, bm)):

        if (mask is True) and (max(a[0], b[0]) >= min(a[0] + a[1], b[0] + b[1])):
            cur_dist = _distance_function(a[0], a[1], b[0], b[1])

            if cur_dist < shortest_dist:
                hit_ix = ix

                # update shortest distance
                shortest_dist = cur_dist

    return hit_ix, shortest_dist


def find_nearest_pair2(al: List[Any], am: List[bool],
                      bl: List[Any], bm: List[bool], the_gamma):

    shortest_dist = sys.maxsize
    hit_a_ix = -1
    hit_b_ix = -1

    for a_ix, (a, a_mask) in enumerate(zip(al, am)):
        b_ix, cur_dist = find_nearest_element2(a, a_mask, bl, bm, the_gamma)

        if b_ix != -1 and cur_dist < shortest_dist:
            hit_a_ix = a_ix
            hit_b_ix = b_ix

            # update shortest distance
            shortest_dist = cur_dist

    return hit_a_ix, hit_b_ix, shortest_dist

def find_nearest_element2(a: Any, a_mask, bl: List[Any], bm: List[bool], the_gamma):

    shortest_dist = sys.maxsize
    hit_ix = -1

    for ix, (b, mask) in enumerate(zip(bl, bm)):
        if (mask is True or a_mask is True) and (max(a[0], b[0]) >= min(a[0] + a[1], b[0] + b[1])) and (_distance_function(a[0], a[1], b[0], b[1]) < the_gamma):
            cur_dist = _distance_function(a[0], a[1], b[0], b[1])

            if cur_dist < shortest_dist:
                hit_ix = ix

                # update shortest distance
                shortest_dist = cur_dist

    return hit_ix, shortest_dist




def _distance_first_fn1(candicate_by_rtype: Dict[str, Dict[str, List]], the_gamma):

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

def _distance_first_fn2(candicate_by_rtype: Dict[str, Dict[str, List]], the_gamma):

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
            
        while any(sub_masks) or any(obj_masks):
            which_sub, which_obj, _ = find_nearest_pair2(sub_candicates,
                                                        sub_masks,
                                                        obj_candicates,
                                                        obj_masks, the_gamma)
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

def _all_possible(candicate_by_rtype: Dict[str, Dict[str, List]], the_gamma):

    triples = []

    for rtype in candicate_by_rtype:
        sub_candicates = candicate_by_rtype[rtype]['sub']
        obj_candicates = candicate_by_rtype[rtype]['obj']
        
        for sub in sub_candicates:
            for obj in obj_candicates:
                if (max(sub[0], obj[0]) >= min(sub[0] + sub[1], obj[0] + obj[1])):
                    tp = dict(sub_start_ix=sub[0], sub_length=sub[1],
                              obj_start_ix=obj[0], obj_length=obj[1],
                              rtype=rtype)
                    triples.append(tp)

    return triples

def _order_first_fn(candicate_by_rtype: Dict[str, Dict[str, List]], the_gamma):
    triples = []

    for rtype in candicate_by_rtype:
        sub_candicates = candicate_by_rtype[rtype]['sub']
        obj_candicates = candicate_by_rtype[rtype]['obj']
        
        sorted_sub_candidates = sorted(sub_candicates, key=lambda xxx: xxx[0])
        sorted_obj_candidates = sorted(obj_candicates, key=lambda yyy: yyy[0])
        
        for i in range(min(len(sorted_sub_candidates), len(sorted_obj_candidates))):
            sub = sorted_sub_candidates[i]
            obj = sorted_obj_candidates[i]
            
            if (max(sub[0], obj[0]) >= min(sub[0] + sub[1], obj[0] + obj[1])): # and (abs(sub[0] - obj[0]) <= 32):
                tp = dict(sub_start_ix=sub[0], sub_length=sub[1],
                          obj_start_ix=obj[0], obj_length=obj[1],
                          rtype=rtype)
                triples.append(tp)

    return triples



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

def obtain_triple_lists(snip_seqs: List[List[Dict]], have_etype, algorithm: str, the_gamma):

    methods = {'random': '_random_fn',
               'sub-first': '_sub_first_fn',
               'obj-first': '_obj_first_fn',
               'order-first': _order_first_fn,
               'distance-first1': _distance_first_fn1,
               'distance-first2': _distance_first_fn2,
               'all-possible': _all_possible}

    generate_relation_fn = methods[algorithm]
    relations = []

    for snip_seq in snip_seqs:

        # make candidates by type
        candidate_by_rtype = make_candicate(snip_seq)

        # generate relations
        relation = generate_relation_fn(candidate_by_rtype, the_gamma)
        
        # add etype information
        if have_etype:
            boundary_to_etype_dict = dict()
            for sp in snip_seq:
                boundary_to_etype_dict[str(sp['start_ix']) + '-' + str(sp['length'])] = sp['etype']
            for tri in relation:
                tri['sub_etype'] = boundary_to_etype_dict[str(tri['sub_start_ix']) + '-' + str(tri['sub_length'])]
                tri['obj_etype'] = boundary_to_etype_dict[str(tri['obj_start_ix']) + '-' + str(tri['obj_length'])]

        # append to relations
        relations.append(relation)

    return relations








