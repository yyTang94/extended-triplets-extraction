import json
from typing import Any


def _read_from_json(location: str):
    with open(location, 'r') as f:
        return json.load(f)


def _write_to_json(data: Any, location: str):
    with open(location, 'w') as f:
        json.dump(data, f)


def read(folder: str):

    # xd
    train_sentences = _read_from_json(folder + 'train_sentences.json')
    valid_sentences = _read_from_json(folder + 'valid_sentences.json')
    test_sentences = _read_from_json(folder + 'test_sentences.json')

    word_lookup = _read_from_json(folder + 'word_lookup.json')
    word_embedding = _read_from_json(folder + 'word_embedding.json')

    train_case_seqs = _read_from_json(folder + 'train_case_seqs.json')
    valid_case_seqs = _read_from_json(folder + 'valid_case_seqs.json')
    test_case_seqs = _read_from_json(folder + 'test_case_seqs.json')

    # yd
    train_tag_seqs = _read_from_json(folder + 'train_tag_seqs.json')
    valid_tag_seqs = _read_from_json(folder + 'valid_tag_seqs.json')
    test_tag_seqs = _read_from_json(folder + 'test_tag_seqs.json')

    tag_lookup = _read_from_json(folder + 'tag_lookup.json')

    # build
    xdata = dict(train_sentences=train_sentences,
                 valid_sentences=valid_sentences,
                 test_sentences=test_sentences,

                 train_case_seqs=train_case_seqs,
                 valid_case_seqs=valid_case_seqs,
                 test_case_seqs=test_case_seqs)
    xlookup = dict(word_lookup=word_lookup)

    ydata = dict(train_tag_seqs=train_tag_seqs,
                 valid_tag_seqs=valid_tag_seqs,
                 test_tag_seqs=test_tag_seqs)
    ylookup = dict(tag_lookup=tag_lookup)

    return xdata, xlookup, ydata, ylookup, word_embedding


def write(train_losses, valid_losses, prf_values, folder):
    _write_to_json(train_losses, folder + 'train_losses.json')
    _write_to_json(valid_losses, folder + 'valid_losses.json')
    _write_to_json(prf_values, folder + 'prf_values.json')
