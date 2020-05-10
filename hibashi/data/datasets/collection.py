import torch
from functools import reduce
import numpy as np

from hibashi.data.datasets.datasets import Dataset
from hibashi.framework.utils import get_subclass



class Collection(Dataset):
    """
    Collection dataset is designed to instantiate multiple datasets and iterate over them simultaneously
    """

    def __init__(self, ds_names: tuple, sample_keys: tuple, aug_names=None, n_sub_samples=None, ds_kw_args=None,
                 **kwargs):
        """
        In the initialization of the Collection all the datasets in the collection are constructed

        :param ds_names: (tuple of strings) names of the datasets
        :param sample_keys: (tuple of strings) names of the keys used to return the batch,
        order is in corresponding to the names tuple
        :param oversampling: (tuple of ints) number of samples to return, for taking a subset during debugging,
        order is in corresponding to the names tuple
        :param n_samples: (tuple of integers) number of samples per item get fetched for the respective dataset,
        order is in corresponding to the names tuple
        :param aug_names: (tuple of tuples of strings) sequence of augmentation class names for the respective dataset,
        order is in corresponding to the names tuple
        :param n_sub_samples: (tuple of ints or int)
        # TODO should we add kwargs here and how should we pass them to the child classes?
        :param kwargs:
        """
        super(Collection, self).__init__()

        self.ds_names = ds_names
        self.sample_keys = sample_keys

        self.ds = []

        n_ds = len(ds_names)  # number of datasets part of the collection

        if aug_names is None:
            aug_names = n_ds * (None,)

        n_sub_samples = self._check_and_extend(n_sub_samples, n_ds)
        self.n_sub_samples = n_sub_samples

        ds_kw_args = self._check_and_extend(ds_kw_args, n_ds)

        assert n_ds == len(sample_keys) == len(aug_names) == len(n_sub_samples)

        for ds_name, n_subsample, aug_sequence, keyword_args in \
                zip(ds_names, n_sub_samples, aug_names, ds_kw_args):
            # TODO discuss the following options, if the argument is not given and is therefor by default None,
            #  we can either overwrite these arguments for the instantiation each class inside the collection
            #  or we can use the default argument from the class and thus not pass it along. The last option would
            #  a little bit dirtier code wise but could make sense
            ds_class = get_subclass(ds_name, Dataset)
            if isinstance(keyword_args, dict):
                ds_inst = ds_class(subsample=n_subsample, aug_names=aug_sequence, **keyword_args)
            else:
                ds_inst = ds_class(subsample=n_subsample, aug_names=aug_sequence)
            self.ds.append(ds_inst)

    @staticmethod
    def _check_and_extend(x, n_ds: int):
        """
        Utility function to extend the argument to have the length as the number of datasets in the collection
        :param x: argument to extend
        :param n_ds: number of datasets in the collection
        :return: extended x
        """
        if x is None:
            return n_ds * (None,)
        elif isinstance(x, int):
            return n_ds * (x,)
        else:
            return x

    @staticmethod
    def merge_batch(batch: dict):
        """
        Helper function to merge a batch after it comes from the multi index sampler.
        We cannot merge samples coming from different datasets in the getitem because then it is not indexable anymore.
        This function takes out the tensors and lists out of sub dictionaries and merges them and
        returns a new dictionary without sub levels
        :param batch: batch with at least two level hierarchy
        :return:
        """
        new_batch = {}

        l0_keys = list(batch.keys())

        if isinstance(batch[l0_keys[0]], dict): # 2 levels
            for l1_key in batch[l0_keys[0]].keys():
                values = []
                for l0_key in l0_keys:
                    values.append(batch[l0_key][l1_key])
                for val in values[1:]:
                    if type(values[0]) != type(val):
                        raise ValueError('Types of values are not the same')
                if isinstance(values[0], list):
                    new_batch[l1_key] = reduce(list.__add__, values)
                elif isinstance(values[0], torch.Tensor):
                    new_batch[l1_key] = torch.cat(values, 0)
                else:
                    raise NotImplementedError('key {}: merging {} is not yet supported'.format(key, type(p1_value)))
            return new_batch
        elif isinstance(batch[l0_keys[0]], list): # 3 levels
            for l2_key in batch[l0_keys[0]][0].keys():
                values = []
                for l0_key in l0_keys:
                    for sample in batch[l0_key]:
                        values.append(sample[l2_key])
                for val in values[1:]:
                    if type(values[0]) != type(val):
                        raise ValueError('Types of values are not the same')
                if isinstance(values[0], list):
                    new_batch[l2_key] = reduce(list.__add__, values)
                elif isinstance(values[0], torch.Tensor):
                    new_batch[l2_key] = torch.cat(values, 0)
                else:
                    raise NotImplementedError('key {}: merging {} is not yet supported'.format(key, type(p1_value)))
            return new_batch
        else:
            raise NotImplementedError

    def __getitem__(self, idcs):
        """
        Compose a sample by aggregating all the samples for the different datasets
        :param idcs: (int) index
        :return:
        """
        if not isinstance(idcs, tuple):
            raise ValueError('A Collection object is only indexable with a tuple')
        elif len(idcs) != self.n_datasets:
            raise ValueError('The number of indices should be equal to the number of datasets in the collection')


        sample = {}
        for key, dataset, idx in zip(self.sample_keys, self.ds, idcs):
            if type(idx) in [int, np.int32, np.int64]:
                sample[key] = dataset[idx]
            elif type(idx) == list:
                sample[key] = [dataset[single_idx] for single_idx in idx]
            else:
                raise NotImplementedError

        return sample

    def __len__(self):
        """
        Length of the dataset
        :return: the maximum of all the datasets in the collection
        """
        lengths = []
        for dataset in self.ds:
            if dataset.finite:
                lengths.append(len(dataset))
        return max(lengths)

    @property
    def n_datasets(self):
        return len(self.ds)

    @property
    def finite(self) -> bool:
        """
        :return: Is this dataset finite?
        """
        raise NotImplementedError
