import json
from typing import List, Dict, Any


class Preprocessor(object):

    def __init__(self):
        pass

    def read(self, raw_folder: str):
        raise NotImplementedError

    def preprocess(self, raw_data: Any):
        raise NotImplementedError

    def divide_by_indexes(self, samples: List[Dict], divide_indexes: Dict):

        train_sids = set(divide_indexes['train'])
        valid_sids = set(divide_indexes['valid'])
        test_sids = set(divide_indexes['test'])

        train_samples = []
        valid_samples = []
        test_samples = []

        for s in samples:
            if s['sid'] in train_sids:
                train_samples.append(s)

            if s['sid'] in valid_sids:
                valid_samples.append(s)

            if s['sid'] in test_sids:
                test_samples.append(s)

        return train_samples, valid_samples, test_samples

    def write(self,
              train_samples: List[Dict],
              valid_samples: List[Dict],
              test_samples: List[Dict],
              folder: str):

        with open(folder + 'train_samples.json', 'w') as f:
            json.dump(train_samples, f)

        with open(folder + 'valid_samples.json', 'w') as f:
            json.dump(valid_samples, f)

        with open(folder + 'test_samples.json', 'w') as f:
            json.dump(test_samples, f)

    def run(self, raw_folder: str, clean_folder: str):

        # read raw data from raw folder
        raw_data = self.read(raw_folder)

        # preprocess raw data and achieve clean samples + divide_indexes
        samples, divide_indexes = self.preprocess(raw_data)

        # divide by indexes
        train_samples, valid_samples, test_samples = self.divide_by_indexes(
            samples, divide_indexes)

        # write samples and divide_indexes to clean folder
        self.write(train_samples, valid_samples, test_samples, clean_folder)
