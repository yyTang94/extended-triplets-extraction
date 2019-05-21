import copy
import random
from typing import List, Dict


def add_pad_unk(embeddings: Dict, pad_token: str, unk_token: str):

    embedding_size = len(next(iter(embeddings.values())))

    # pad token
    if pad_token not in embeddings:
        embeddings[pad_token] = (
            [random.gauss(0, 1) for _ in range(embedding_size)])

    # unk token
    if unk_token not in embeddings:
        embeddings[unk_token] = (
            [random.gauss(0, 1) for _ in range(embedding_size)])


def filter_unused_word(embeddings: Dict, sentences: List[List[str]]):

    # used words
    used_words = set()
    for s in sentences:
        used_words.update(s)

    # build word_lookup and word_embedding
    word_lookup = []
    word_embedding = []

    for w in used_words:
        word_lookup.append(w)
        word_embedding.append(embeddings[w])

    return word_lookup, word_embedding


def _cut_and_pad(s: List[str], max_length: int, pad_token: str):

    s_len = len(s)

    if s_len < max_length:
        new_s = s + [pad_token] * (max_length - s_len)
    elif s_len > max_length:
        new_s = s[0: max_length]
    else:
        new_s = s

    return new_s


def _low_case(s: List[str]):

    cases = []
    lowed = []

    for token in s:
        if token.isalpha():
            if token.isupper():
                cases.append(0)  # upper: 0
            elif token.istitle():
                cases.append(1)  # title: 1
            elif token.islower():
                cases.append(2)  # lower: 2
            else:
                cases.append(3)  # hybird: 3
        else:
            cases.append(3)  # hybird

        lowed.append(token.lower())

    return lowed, cases


def _replace_oov(s: List[str], embeddings: Dict, unk_token: str):

    new_s = []

    for token in s:
        if token in embeddings:
            new_s.append(token)
        else:
            new_s.append(unk_token)

    return new_s


def adjust_sentence(sentences: List[List[str]],
                    max_length: int, pad_token: str,
                    embeddings: Dict, unk_token: str):

    new_sentences = []
    case_seqs = []

    for s in sentences:

        # cut and pad
        cutted = _cut_and_pad(s, max_length, pad_token)

        # low case
        lowed, case_seq = _low_case(cutted)

        # replace_oov
        new_s = _replace_oov(lowed, embeddings, unk_token)

        # append to new_sentences
        new_sentences.append(new_s)
        case_seqs.append(case_seq)

    return new_sentences, case_seqs
