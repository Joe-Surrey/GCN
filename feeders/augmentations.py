import random
import numpy as np
from utils import import_class


class Augmentor:
    def __init__(self, augmentations):
        self.augs = [import_class(aug) for aug in augmentations]

    def __call__(self, data_numpy):
        for aug in self.augs:
            data_numpy = aug(data_numpy)
        return data_numpy


def random_mirror(data_numpy):
    #  TODO random!!!
    if random.random() < 0.5:
        data_numpy[0] = - data_numpy[0]
    return data_numpy


def random_jitter(data_numpy):
    jitter = np.random.normal(loc=0., scale=0.05, size=data_numpy[:2].shape)  # loc=mean, scale=std
    data_numpy[:2] += jitter
    return data_numpy


def random_dropout(data_numpy):
    droput = np.random.random(size=data_numpy[2:].shape) < data_numpy[2:]
    data_numpy[:2] *= droput
    return data_numpy


def augment(data_numpy):
    data_numpy = random_mirror(data_numpy)
    return data_numpy