from typing import List, Dict


def _digitize_sentence(data: List, lookup: Dict):

    ix_data = []

    for i_data in data:
        i_ix_data = [lookup[g] for g in i_data]
        ix_data.append(i_ix_data)

    return ix_data


def _digitize_tag(data: List, lookup: Dict):

    ix_tag_seqs = []

    for tag_seq in data:
        ix_tag_seq = []
        for tags in tag_seq:
            ix_tag_seq.append([lookup[x] for x in tags])

        ix_tag_seqs.append(ix_tag_seq)

    return ix_tag_seqs


def _lookup_to_dict(lookup_l: List):

    lookup_d = dict()

    for i, g in enumerate(lookup_l):
        lookup_d[g] = i

    return lookup_d


def digitize_xdata(xdata: Dict, xlookup: Dict):

    train_sentences = xdata['train_sentences']
    valid_sentences = xdata['valid_sentences']
    test_sentences = xdata['test_sentences']

    # change from list to dict
    word_lookup = xlookup['word_lookup']
    word_lookup_ = _lookup_to_dict(word_lookup)

    # digitize
    train_sentences_ = _digitize_sentence(train_sentences, word_lookup_)
    valid_sentences_ = _digitize_sentence(valid_sentences, word_lookup_)
    test_sentences_ = _digitize_sentence(test_sentences, word_lookup_)

    # build
    x = dict(train_sentences=train_sentences_,
             valid_sentences=valid_sentences_,
             test_sentences=test_sentences_,

             train_case_seqs=xdata['train_case_seqs'],
             valid_case_seqs=xdata['valid_case_seqs'],
             test_case_seqs=xdata['test_case_seqs'])

    return x


def _make_matrix(tag_seqs: List[List[List[str]]], class_num: int):
    mat_l0 = []
    for tag_seq in tag_seqs:
        mat_l1 = []
        for tgs in tag_seq:
            mat_l2 = [0 for _ in range(class_num)]
            for tg in tgs:
                mat_l2[tg] = 1

            mat_l1.append(mat_l2)

        mat_l0.append(mat_l1)

    return mat_l0


def digitize_ydata(ydata: Dict, ylookup: List, class_num: int):

    train_tag_seqs = ydata['train_tag_seqs']
    valid_tag_seqs = ydata['valid_tag_seqs']
    test_tag_seqs = ydata['test_tag_seqs']

    # change from list to dict
    tag_lookup = ylookup['tag_lookup']
    tag_lookup_ = _lookup_to_dict(tag_lookup)

    # digitize
    train_tag_seqs_ = _digitize_tag(train_tag_seqs, tag_lookup_)
    valid_tag_seqs_ = _digitize_tag(valid_tag_seqs, tag_lookup_)
    test_tag_seqs_ = _digitize_tag(test_tag_seqs, tag_lookup_)

    # matrix
    train_tag_matrix = _make_matrix(train_tag_seqs_, class_num)
    valid_tag_matrix = _make_matrix(valid_tag_seqs_, class_num)
    test_tag_matrix = _make_matrix(test_tag_seqs_, class_num)

    # build
    y = dict(train_tag_matrix=train_tag_matrix,
             valid_tag_matrix=valid_tag_matrix,
             test_tag_matrix=test_tag_matrix)

    return y


# def _realize(ix_data: List, lookup: List):

#     data = []

#     for i_ix_data in ix_data:
#         i_data = [lookup[g] for g in i_ix_data]
#         data.append(i_data)

#     return data


# def realize_y(gen_y: Dict, ylookup: List):

#     gen_tag_seqs_ = gen_y['gen_tag_seqs']
#     gen_tag_seqs = _realize(gen_tag_seqs_, ylookup)
#     gen_ydata = dict(gen_tag_seqs=gen_tag_seqs)

#     return gen_ydata
