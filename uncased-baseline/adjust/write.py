import json
from typing import List, Dict, Any


def _write_to_json(data: Any, location: str):
    with open(location, 'w') as f:
        json.dump(data, f)


def run(train_sentences: List[List[str]],
        valid_sentences: List[List[str]],
        test_sentences: List[List[str]],
        word_lookup: List[str],
        word_embedding: List[List[float]],

        train_case_seqs: List[List[int]],
        valid_case_seqs: List[List[int]],
        test_case_seqs: List[List[int]],

        train_relations: List[List[str]],
        valid_relations: List[List[str]],
        test_relations: List[List[str]],

        folder: str):

    # write sentences
    _write_to_json(train_sentences, folder + 'train_sentences.json')
    _write_to_json(valid_sentences, folder + 'valid_sentences.json')
    _write_to_json(test_sentences, folder + 'test_sentences.json')
    _write_to_json(word_lookup, folder + 'word_lookup.json')
    _write_to_json(word_embedding, folder + 'word_embedding.json')

    # write case_seqs
    _write_to_json(train_case_seqs, folder + 'train_case_seqs.json')
    _write_to_json(valid_case_seqs, folder + 'valid_case_seqs.json')
    _write_to_json(test_case_seqs, folder + 'test_case_seqs.json')

    # write relations
    _write_to_json(train_relations, folder + 'train_relations.json')
    _write_to_json(valid_relations, folder + 'valid_relations.json')
    _write_to_json(test_relations, folder + 'test_relations.json')

    # write rtype_lookup
    _write_to_json(["OrgBased_In", "Located_In", "Live_In", "Kill", "Work_For"], folder + 'rtype_lookup.json')

