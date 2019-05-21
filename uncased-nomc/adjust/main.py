import argparse

import read
from sentence import add_pad_unk, adjust_sentence, filter_unused_word
from relation import adjust_relation
import write


def main(clean_folder: str, pretrained_embedding_location: str,
         max_length: int, pad_token: str, unk_token: str, ready_folder: str):

    # read all files
    (train_sentences, train_relations,
     valid_sentences, valid_relations,
     test_sentences, test_relations,
     embeddings) = read.run(clean_folder,
                            pretrained_embedding_location)

    # xd: sentence
    add_pad_unk(embeddings, pad_token, unk_token)

    train_sentences_, train_case_seqs = adjust_sentence(train_sentences,
                                                        max_length, pad_token,
                                                        embeddings, unk_token)
    valid_sentences_, valid_case_seqs = adjust_sentence(valid_sentences,
                                                        max_length, pad_token,
                                                        embeddings, unk_token)
    test_sentences_, test_case_seqs = adjust_sentence(test_sentences,
                                                      max_length, pad_token,
                                                      embeddings, unk_token)

    # filter unused words
    sentences = train_sentences_ + valid_sentences_ + test_sentences_
    word_lookup, word_embedding = filter_unused_word(embeddings, sentences)

    # yd: relation
    train_relations_ = adjust_relation(train_relations, max_length)
    valid_relations_ = adjust_relation(valid_relations, max_length)
    test_relations_ = adjust_relation(test_relations, max_length)

    # write sentence and relation
    write.run(train_sentences_, valid_sentences_, test_sentences_, word_lookup, word_embedding,
              train_case_seqs, valid_case_seqs, test_case_seqs,
              train_relations_, valid_relations_, test_relations_, ready_folder)


if __name__ == '__main__':

    # accept arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_embedding_location', type=str,
                        default=('/home/yytang/Projects/Word-Embedding/' +
                                 'glove.6B.50d.txt'))

    parser.add_argument('--clean_folder', type=str, default='../clean/conll/')
    parser.add_argument('--ready_folder', type=str, default='../ready/conll/')

    parser.add_argument('--max_length', type=int, default=120)
    parser.add_argument('--pad_token', type=str, default='__pad__')
    parser.add_argument('--unk_token', type=str, default='__unk__')

    args = parser.parse_args()

    main(args.clean_folder, args.pretrained_embedding_location,
         args.max_length, args.pad_token, args.unk_token, args.ready_folder)
