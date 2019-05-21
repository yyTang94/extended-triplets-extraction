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
                        ['B', 'I', 'L', 'U'],
                        ['Peop', 'Loc', 'Org'])
        tag_lookup = []

        for rtype, role, pos, etype in prod_:
            tag_lookup.append(pos + '#' + etype + '#' + rtype + '#' + role)

        return tag_lookup

    def _tag_fn(self, rtype: str, role: str, length: int, etype):

        if length == 1:
            tags = ['U' + '#' + etype + '#' + rtype + '#' + role]
        elif length == 2:
            tags = ['B' + '#' + etype + '#' + rtype + '#' + role,
                    'L' + '#' + etype + '#' + rtype + '#' + role]
        elif length >= 3:
            tags = (['B' + '#' + etype + '#' + rtype + '#' + role] +
                    ['I' + '#' + etype + '#' + rtype + '#' + role
                     for _ in range(length - 2)] +
                    ['L' + '#' + etype + '#' + rtype + '#' + role])

        return tags

    def snip_seq_to_tag_seq_once(self, snip_seq: List[Dict],
                                 tag_lookup: List[str]):

        tag_seq = [set() for _ in range(self.max_length)]

        for sp in snip_seq:
            cur_tags = self._tag_fn(sp['rtype'], sp['role'], sp['length'], sp['etype'])
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

    def _is_consistent(self, tags: List[str]):
        first_tag = tags[0]

        first_rtype = first_tag.split('#')[1]
        first_role = first_tag.split('#')[2]

        flag = True

        for tg in tags[1:]:
            rtype = tg.split('#')[1]
            role = tg.split('#')[2]

            if rtype != first_rtype or role != first_role:
                flag = False

                break

        return flag

    def tag_seq_to_snip_seq_once(self, tag_seq: List[str]):

        pos_labels = ''.join([tg[0] for tg in tag_seq])
        pat = re.compile('BI*L|U')

        # fragments
        fragments = []

        for m in pat.finditer(pos_labels):
            fragments.append(m.span())

        # snip_seq
        snip_seq = []

        for fg in fragments:
            start_ix = fg[0]
            length = fg[1] - fg[0]

            if self._is_consistent(tag_seq[start_ix: start_ix + length]):
                _, rtype, role = tag_seq[start_ix].split('#')
                sp = dict(start_ix=start_ix, length=length,
                          rtype=rtype, role=role)
                snip_seq.append(sp)

        return snip_seq

    def tag_seq_to_snip_seq(self, tag_seqs: List[List[List[str]]]):

        snip_seqs = []

        for tag_seq in tag_seqs:
            for i in range(len(tag_seq)):
                tag_seq[i].append("O")
                tag_seq[i].append("O")
            tag_seq1 = [x[0] for x in tag_seq]
            tag_seq2 = [x[1] for x in tag_seq]

            snip_seq1 = self.tag_seq_to_snip_seq_once(tag_seq1)
            snip_seq2 = self.tag_seq_to_snip_seq_once(tag_seq2)

            snip_seq = snip_seq1 + snip_seq2

            snip_seqs.append(snip_seq)

        return snip_seqs
