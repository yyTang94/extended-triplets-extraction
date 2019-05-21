import logging
import argparse

from conll import ConllPreprocessor
from ace import AcePreprocessor

logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':

    # accept arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='ace')
    parser.add_argument('--raw_folder', type=str, default='../raw/ace/')
    parser.add_argument('--clean_folder', type=str, default='../clean/ace/')

    args = parser.parse_args()

    # create preprocessor and run it

    if args.dataset == 'conll':
        preprocessor = ConllPreprocessor()
    elif args.dataset == 'ace':
        preprocessor = AcePreprocessor()

    preprocessor.run(args.raw_folder, args.clean_folder)
