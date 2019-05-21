import random
from typing import List, Dict, Any

from preprocessor import Preprocessor


class ConllPreprocessor(Preprocessor):

    def __init__(self):
        Preprocessor.__init__(self)

    def _read_chunks(self, loc: str):

        with open(loc, 'r') as f:
            chunks = []

            # initialize num_of_blank and cur_chunk
            num_of_blank = 0
            cur_chunk = dict(detail_lines=[], addition_lines=[])

            for line in f:
                items = line.strip().split()

                if num_of_blank == 0:

                    if len(items) == 0:  # blank
                        num_of_blank += 1
                    else:  # not blank
                        cur_chunk['detail_lines'].append(items)

                elif num_of_blank == 1:

                    if len(items) == 0:  # blank
                        chunks.append(cur_chunk)

                        # initialize num_of_blank and cur_chunk
                        num_of_blank = 0
                        cur_chunk = dict(detail_lines=[], addition_lines=[])
                    else:  # not blank
                        cur_chunk['addition_lines'].append(items)

                else:
                    raise ValueError

            # add the last chunk
            chunks.append(cur_chunk)

        return chunks

    def _read_sids(self, loc: str):

        sids = []

        with open(loc, 'r') as f:
            for line in f:
                items = line.strip().split()
                cur_sid = items[0][0: -1]  # delete the ":" in "123:"

                sids.append(cur_sid)

        return sids

    def read(self, raw_folder: str):

        # read chunks
        chunks = self._read_chunks(raw_folder + 'corp.tsv')

        # read train sids
        train_sids = self._read_sids(raw_folder + 'training_indx_sentence.txt')

        # read test sids
        test_sids = self._read_sids(raw_folder + 'testing_indx_sentence.txt')

        # split 100 from train for valid
        random.shuffle(train_sids)
        divide_indexes = dict(train=train_sids[100:],
                              valid=train_sids[0: 100],
                              test=test_sids)

        # build raw_data
        raw_data = (chunks, divide_indexes)

        return raw_data

    def _extract_chain(self, detail_lines: List[List[str]]):

        # build chain from each molecule
        chain = []

        for line in detail_lines:
            tokens = line[5].split('/')

            length = len(tokens)
            span = ' '.join(tokens)
            etype = line[1]

            molecule = dict(length=length,
                            span=span,
                            etype=etype)
            chain.append(molecule)

        # add start_ix to each molecule
        next_start_ix = 0

        for molecule in chain:
            molecule['start_ix'] = next_start_ix

            next_start_ix += molecule['length']

        return chain

    def _extract_relations(self, addition_lines: List[List[str]]):

        relations = []

        for line in addition_lines:
            sub_molecule_ix = int(line[0])
            obj_molecule_ix = int(line[1])
            rtype = line[2]

            r = dict(sub_molecule_ix=sub_molecule_ix,
                     obj_molecule_ix=obj_molecule_ix,
                     rtype=rtype)
            relations.append(r)

        return relations

    def _obtain_sid(self, detail_lines: List[List[str]]):
        return detail_lines[0][0]

    def _obtain_sentence(self, chain: List[Dict]):

        # build sentence

        sentence = []

        for molecule in chain:
            sentence.extend(molecule['span'].split())

        # in raw data ',' is replaced by COMMA

        for i, token in enumerate(sentence):
            if token == 'COMMA':
                sentence[i] = ','

        return sentence

    def _obtain_entities(self, chain: List[Dict]):

        entities = []

        for molecule in chain:

            if molecule['etype'] != 'O' and molecule['etype'] != 'Other':
                e = dict(start_ix=molecule['start_ix'],
                         length=molecule['length'],
                         span=molecule['span'],
                         etype=molecule['etype'])
                entities.append(e)

        return entities

    def _obtrain_triples(self, relations: List[Dict], chain: List[Dict]):

        triples = []

        for r in relations:
            sub_molecule_ix = r['sub_molecule_ix']
            obj_molecule_ix = r['obj_molecule_ix']
            rtype = r['rtype']

            t = dict(sub_start_ix=chain[sub_molecule_ix]['start_ix'],
                     sub_length=chain[sub_molecule_ix]['length'],
                     sub_span=chain[sub_molecule_ix]['span'],
                     sub_etype=chain[sub_molecule_ix]['etype'],
                     obj_start_ix=chain[obj_molecule_ix]['start_ix'],
                     obj_length=chain[obj_molecule_ix]['length'],
                     obj_span=chain[obj_molecule_ix]['span'],
                     obj_etype=chain[obj_molecule_ix]['etype'],
                     rtype=rtype)
            triples.append(t)

        return triples

    def preprocess(self, raw_data: Any):

        (chunks, divide_indexes) = raw_data
        samples = []

        for c in chunks:
            detail_lines = c['detail_lines']
            addition_lines = c['addition_lines']

            # prepare chain and relations
            chain = self._extract_chain(detail_lines)
            relations = self._extract_relations(addition_lines)

            # sid
            sid = self._obtain_sid(detail_lines)

            # sentence
            sentence = self._obtain_sentence(chain)

            # entities
            entities = self._obtain_entities(chain)

            # triples
            triples = self._obtrain_triples(relations, chain)

            # construct current sample
            s = dict(sid=sid,
                     sentence=sentence,
                     entities=entities,
                     triples=triples)
            samples.append(s)

        return samples, divide_indexes
