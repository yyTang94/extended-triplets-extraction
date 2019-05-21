import re
import sys
import json

import numpy as np

from typing import List, Dict, Any

def _get_the_xxx(a_tag, have_etype):
    items = a_tag.split('#')
    if have_etype:
        the_xxx = items[0] + '-' + items[1]
    else:
        the_xxx = items[0]
        
    return the_xxx

def _remove_inconsistent_bag(tag_seq: List[List[str]], have_etype):
    new_tag_seq = []

    for bg in tag_seq:
        if len(bg) != 0:
            flag = True
            for t in bg:
                if _get_the_xxx(t, have_etype) != _get_the_xxx(bg[0], have_etype):
                    flag = False
                    break
            if flag == False:
                new_bg = []
            else:
                new_bg = bg
        else:
            new_bg = []

        new_tag_seq.append(new_bg)

    return new_tag_seq


def tag_seq_to_snip_seq_once(tag_seq: List[List[str]], have_etype):

    # remove inconsistent
    new_tag_seq = _remove_inconsistent_bag(tag_seq, have_etype)

    # pos labels
    pos_labels = ''
    for bag in new_tag_seq:
        if len(bag) == 0:
            pos_labels = pos_labels + 'O'
        else:
            pos_labels = pos_labels + bag[0][0]

    pat = re.compile('BI*L|U')

    # fragments
    fragments = []

    for m in pat.finditer(pos_labels):
        fragments.append(m.span())

    # simple tag seq
    simple_tag_seq = []
    for bag in new_tag_seq:
        simple_bag = [t[2:] for t in bag]
        simple_tag_seq.append(simple_bag)

    # snips
    snip_seq = []

    for fg in fragments:
        start_ix = fg[0]
        length = fg[1] - fg[0]

        all_sufs = []
        for i in range(start_ix, start_ix + length):
            all_sufs.extend(simple_tag_seq[i])

        cand_sufs = list(set(all_sufs))
        real_sufs = []

        for suf in cand_sufs:
            if all_sufs.count(suf) == length:
                real_sufs.append(suf)

        for suf in real_sufs:
            if have_etype:
                etype, rtype, role = suf.split('#')
                sp = dict(start_ix=start_ix, length=length, etype=etype, rtype=rtype, role=role)
            else:
                rtype, role = suf.split('#')
                sp = dict(start_ix=start_ix, length=length, rtype=rtype, role=role)
            snip_seq.append(sp)

    return snip_seq

def obtain_candidate_lists(tag_seqs: List[List[List[str]]], have_etype):
    snip_seqs = []

    for tag_seq in tag_seqs:
        snip_seq = tag_seq_to_snip_seq_once(tag_seq, have_etype)

        snip_seqs.append(snip_seq)

    return snip_seqs