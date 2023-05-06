import numpy as np


class AverageMeter:
    def __init__(self):
        self._records = []

    def update(self, result):
        self._records.append(result)

    def reset(self):
        self._records.clear()

    def get_records(self, last_n=None):
        if last_n is not None:
            return self._records[-last_n:]
        return self._records

    def sum(self, last_n=None):
        records = self.get_records(last_n=last_n)
        return np.sum(records)

    def mean(self, last_n=None):
        records = self.get_records(last_n=last_n)
        return np.mean(records)

    def std(self, last_n=None):
        records = self.get_records(last_n=last_n)
        return np.std(records)

    def median(self, last_n=None):
        records = self.get_records(last_n=last_n)
        return np.median(records)
