import numpy as np

class ArrayUtils:
    @staticmethod
    def quantile(a, q):
        return np.quantile(a, q)

    @staticmethod
    def arange(start, stop, step=None):
        return np.arange(start, stop, step)

    @staticmethod
    def linspace(start, end, steps):
        return np.linspace(start, end, steps)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def interp(x, xp, fp):
        return np.interp(x, xp, fp)

    @staticmethod
    def get_sizes(datas):
        return np.array(datas).shape

    @staticmethod
    def get_length(datas):
        length = 1
        for dim in datas:
            length *= dim
        return length
