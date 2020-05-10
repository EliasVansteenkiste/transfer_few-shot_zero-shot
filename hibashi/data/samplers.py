import torch
import numpy as np
from itertools import chain
from torch.utils.data import Sampler

from hibashi.data.datasets.collection import Collection


class MultiIndexSampler(Sampler):
    """Samples elements from a collection. """

    def __init__(self, data_source):
        """
        Simple init function
        :param data_source: instance of Collection
        """
        self.collection = data_source

        if not isinstance(data_source, Collection):
            raise ValueError("The Multi Index Sampler is only designed to work with a Collection Dataset instance")

    @property
    def n_samples(self):
        return len(self.collection)

    def __iter__(self):
        indices = []
        for dataset in self.collection.ds:
            if dataset.finite:
                idxs = (np.arange(self.n_samples) % len(dataset))
            else:
                idxs = np.arange(self.n_samples)
            indices.append(idxs.tolist())
        return iter(zip(*indices))

    def __len__(self):
        return self.n_samples


class MultiIndexRandomSampler(Sampler):
    """ Samples elements randomly from the collection.
    If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.collection = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if self._num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.n_samples, int) or self.n_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.n_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if not isinstance(data_source, Collection):
            raise ValueError("The Multi Index Random Sampler is only designed"
                             " to work with a Collection Dataset instance")

        self.n_iter_calls = 0

    @property
    def n_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.collection)
        return self._num_samples

    def __iter__(self):

        indices = []
        for dataset in self.collection.ds:
            if self.replacement:
                low = 0 if dataset.finite else self.n_iter_calls * self.n_samples
                high = len(dataset) if dataset.finite else (self.n_iter_calls + 1) * self.n_samples
                indices.append(torch.randint(low=low, high=high, size=(self.n_samples,), dtype=torch.int64).tolist())
            else:
                random_perm = torch.randperm(self.n_samples)
                if dataset.finite:
                    random_perm = random_perm % len(dataset)
                else:
                    random_perm = random_perm + self.n_iter_calls * self.n_samples

                indices.append(random_perm.tolist())

        self.n_iter_calls += 1
        return iter(zip(*indices))

    def __len__(self):
        return self.n_samples


class MultiIndexOverSampler(Sampler):
    """
    The multi index over sampler is used to oversample the random generator while keeping the other indexes constant
    Current use case is to generate multiple variations of the same source image and different noise inputs
    """
    def __init__(self, data_source, dataset='RandomVector', n_os_samples=10, num_samples=None):
        """

        :param data_source: collection with datasets
        :param dataset: the name of the dataset to be oversampled
        :param n_os_samples: the number of over sampling draws. The dataset that is being oversampled
        will be n_os_samples times change before the other indexes change
        :param num_samples:
        """
        self.collection = data_source
        self._num_samples = num_samples
        self.n_os_samples = n_os_samples

        if not isinstance(self.n_samples, int) or self.n_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.n_samples))

        if not isinstance(data_source, Collection):
            raise ValueError("The Multi Index Over Sampler is only designed to work with a Collection Dataset instance")

        if isinstance(dataset, int):
            self.os_ds_idx = dataset
        elif isinstance(dataset, str):
            if dataset not in self.collection.ds_names:
                raise ValueError('The dataset that is supposed to be oversampled is not part of the collection')
            self.os_ds_idx = self.collection.ds_names.index(dataset)
        else:
            raise ValueError(f"The {self.__name__} is only designed to work with a Collection Dataset instance")

    @property
    def n_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.collection) * self.n_os_samples
        return self._num_samples

    def __iter__(self):

        indices = []

        for ds_idx, dataset in enumerate(self.collection.ds):

            if ds_idx == self.os_ds_idx:
                if dataset.finite:
                    idxs = (np.arange(self.n_samples) % len(dataset)).tolist()
                else:
                    idxs = np.arange(self.n_samples).tolist()

            else:
                if dataset.finite:
                    idxs = (np.arange(len(self.collection)) % len(dataset))
                else:
                    idxs = np.arange(len(self.collection))
                idxs = self.n_os_samples * (idxs,)
                idxs = list(chain.from_iterable(zip(*idxs)))

            indices.append(idxs)

        return iter(zip(*indices))

    def __len__(self):
        return self.n_samples


class MultiIndexBalancingSampler(Sampler):
    """ Samples elements randomly from the collection.
    If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, n_per_ds, replacement=False):
        self.collection = data_source
        self.replacement = replacement
        self.n_per_ds = n_per_ds
        self.n_iter_calls = 0

        if not isinstance(self.n_samples, int) or self.n_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.n_samples))

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self.replacement:
            raise NotImplementedError

        if not isinstance(data_source, Collection):
            print("Warning: The Multi Index Balancing Sampler is only designed"
                             " to work with a Collection Dataset instance")

        if not isinstance(n_per_ds, tuple) and not isinstance(n_per_ds, list):
            raise ValueError("number of samples to draw per dataset should be tuple")

        if len(n_per_ds) != len(data_source.ds):
            raise ValueError("The sampler_n_per_ds should contain the same amount of integers as the collection does.")

    @property
    def n_samples(self):
        n_samples = 0
        for idx, ds in enumerate(self.collection.ds):
            if ds.finite:
                n_samples_ds = len(ds) // self.n_per_ds[idx]
                n_samples = max(n_samples, n_samples_ds)
        return n_samples

    def __iter__(self):
        indices = []
        n_total_samples = self.n_samples
        for n_draw, dataset in zip(self.n_per_ds, self.collection.ds):
            if dataset.finite:
                n_its_per_iter = 1 + n_total_samples // (len(dataset)//n_draw)
                random_perms = [torch.randperm(len(dataset)) for _ in range(n_its_per_iter)]
                random_perms = torch.cat(random_perms, dim=0)
                # subselect an whole number of draws
                random_perms = random_perms[:n_draw * (len(random_perms)//n_draw)]
                random_perms = random_perms.view(-1, n_draw)
                indices.append(random_perms.tolist())
            else:
                random_perm = torch.randperm(n_total_samples * n_draw) + self.n_iter_calls * n_total_samples * n_draw
                random_perm = random_perm[:n_draw * (len(random_perm) // n_draw)]
                random_perm = random_perm.view(-1, n_draw)
                indices.append(random_perm.tolist())

        self.n_iter_calls += 1
        return iter(zip(*indices))

    def __len__(self):
        return self.n_samples