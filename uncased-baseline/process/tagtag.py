import re
from random import shuffle
from itertools import product
from typing import List, Dict


class BILUOTagger(object):
    o_tag = 'O'

    def __init__(self, max_length: int):
        self.max_length = max_length

    @classmethod
    def get_tag_lookup(cls, rtype_lookup: List[str]):

        prod_ = product(rtype_lookup,
                        ['sub', 'obj'],
                        ['B', 'I', 'L', 'U'])
        tag_lookup = []

        for rtype, role, pos in prod_:
            tag_lookup.append(pos + '#' + rtype + '#' + role)

        return tag_lookup

    def _tag_fn(self, rtype: str, role: str, length: int):

        if length == 1:
            tags = ['U' + '#' + rtype + '#' + role]
        elif length == 2:
            tags = ['B' + '#' + rtype + '#' + role,
                    'L' + '#' + rtype + '#' + role]
        elif length >= 3:
            tags = (['B' + '#' + rtype + '#' + role] +
                    ['I' + '#' + rtype + '#' + role
                     for _ in range(length - 2)] +
                    ['L' + '#' + rtype + '#' + role])

        return tags

    def snip_seq_to_tag_seq_once(self, snip_seq: List[Dict],
                                 tag_lookup: List[str]):

        tag_seq = [set() for _ in range(self.max_length)]

        for sp in snip_seq:
            cur_tags = self._tag_fn(sp['rtype'], sp['role'], sp['length'])
            cur_ixes = range(sp['start_ix'], sp['start_ix'] + sp['length'])

            for ix, tg in zip(cur_ixes, cur_tags):
                tag_seq[ix].add(tg)

        # sort
        new_tag_seq = []

        for tags in tag_seq:
            new_tags = sorted(list(tags), key=lambda x: tag_lookup.index(x))

            new_tag_seq.append(new_tags)

        return new_tag_seq

    def snip_seq_to_tag_seq(self, snip_seqs: List[List[Dict]],
                            tag_lookup: List[str]):

        tag_seqs = []

        for snip_seq in snip_seqs:

            tag_seq = self.snip_seq_to_tag_seq_once(snip_seq, tag_lookup)
            tag_seqs.append(tag_seq)

        return tag_seqs

    def _remove_inconsistent_bag(self, tag_seq: List[List[str]]):
        new_tag_seq = []

        for bg in tag_seq:
            if len(bg) != 0:
                pos = bg[0][0]
                flag = True
                for t in bg:
                    if t[0] != pos:
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

    def tag_seq_to_snip_seq_once(self, tag_seq: List[List[str]]):

        # remove inconsistent
        new_tag_seq = self._remove_inconsistent_bag(tag_seq)

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
                rtype, role = suf.split('#')
                sp = dict(start_ix=start_ix, length=length,
                          rtype=rtype, role=role)
                snip_seq.append(sp)

        return snip_seq

    def tag_seq_to_snip_seq(self, tag_seqs: List[List[List[str]]]):
        snip_seqs = []

        for tag_seq in tag_seqs:
            snip_seq = self.tag_seq_to_snip_seq_once(tag_seq)

            snip_seqs.append(snip_seq)

        return snip_seqs
