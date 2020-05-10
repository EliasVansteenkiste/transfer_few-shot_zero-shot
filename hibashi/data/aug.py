"""
This module contains the abstract augment class
Note: this mother class needs its own file to avoid circular dependencies
"""
from abc import ABC, abstractmethod
from hibashi.framework.utils import get_subclass


class Augment(ABC):
    @abstractmethod
    def __call__(self, sample):
        """
        The call function should be implemented in child classes
        :param sample: (dict) dictionary with data for one sample
        """
        pass


class AugmentSample(object):

    def __init__(self, aug_name_lst):
        self.aug_name_lst = aug_name_lst

    def __call__(self, sample):
        for aug_name in self.aug_name_lst:
            aug = get_subclass(aug_name, Augment)()
            sample = aug(sample)
        return sample
