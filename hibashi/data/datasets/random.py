import sys

import numpy as np
import torch
from torchvision import transforms

from hibashi.data.datasets.datasets import Dataset
from hibashi.data.transformer import ToTensor, ToFloatTensor, ToLongTensor
from hibashi.utils.random import temp_seed


class RandomNormalVector(Dataset):
    """
    Random generator wrapped in a random vector dataset
    """

    def __init__(self, sizes: tuple = ((128,),),
                 seed_offset: int = 0,
                 implementation: str = 'torch', **kwargs):
        """
        Dataset class that contains a random vector generator, whose elements are drawn from a normal distribution,
        commonly used for training GANs. We commonly need to have multiple vectors filled with random numbers.
        This is the reason why sizes is by default considered a tuple of tuples.n
        You can choose between two random generators, with different characteristics and execution speeds

        :param sizes: a tuple of tuples indicating the size of the vectors that need to be generated.
        The default is only one vector with 64 elements
        :param seed_offset: an offset added to the idx seed.
        This is mainly used to have a different sequences for train, validation and test
        :param implementation: Either 'torch' or 'numpy'
        :param kwargs:
        """
        super(RandomNormalVector, self).__init__()

        self.sizes = sizes
        self.seed_offset = seed_offset

        self.implementation = implementation

        if implementation == 'numpy':
            self.rs = np.random.RandomState()
            transform_steps = [ToTensor({'random_normal': 'from_numpy_per_item'}),
                               ToFloatTensor(['random_normal'], per_item=True)]

            self.transform = transforms.Compose(transform_steps)

    def __getitem__(self, idx):
        # I think we have to do this trick here to make sure that the random generator is both thread-safe and
        # deterministic. I implemented it in two ways, the numpy implementation should be thread-safe,
        # I am not exactly sure about the torch implementation
        if self.implementation == 'torch':
            # Pytorch does not allow to instantiate random generators so we set the seed temporarily
            # this may break when using multiple threads, but it is almost twice as fast as numpy and
            # that matters if you have to draw large tensors
            with temp_seed(self.seed_offset + idx, library='torch'):
                random_vectors = []
                for size in self.sizes:
                    random_vectors.append(torch.randn(*size, dtype=torch.float, requires_grad=False))
                sample = {'random_normal': random_vectors}
                return sample
        elif self.implementation == 'numpy':
            # For numpy we can instantiate an random generator in the init and so i think this is safer
            # in a multi threading program
            self.rs.seed(self.seed_offset + idx)
            random_vectors = []
            for size in self.sizes:
                random_vectors.append(self.rs.randn(*size))
            sample = {'random_normal': random_vectors}
            sample = self.transform(sample)
            return sample
        else:
            raise NotImplementedError

    def __len__(self):
        # TODO discuss here what we should do here, the length is the maximum range of the integer value,
        #  but if we use this dataset in a collection, the collection takes the maximum of the lengths of datasets
        #  inside the collection. I solved it now with the finite flag
        return sys.maxsize

    @property
    def finite(self) -> bool:
        """
        :return: Is this dataset finite?
        """
        return False


class RandomUniformVector(Dataset):
    """
    Random generator wrapped in a random vector dataset
    """

    def __init__(self, sizes: tuple = ((128,),),
                 seed_offset: int = 0,
                 implementation: str = 'torch', **kwargs):
        """
        Dataset class that contains a random vector generator, whose elements are drawn from a uniform distribution,
        commonly used for mixing. We commonly need to have multiple vectors filled with random numbers.
        This is the reason why sizes is by default considered a tuple of tuples.n
        You can choose between two random generators, with different characteristics and execution speeds

        :param sizes: a tuple of tuples indicating the size of the vectors that need to be generated.
        The default is only one vector with 64 elements
        :param seed_offset: an offset added to the idx seed.
        This is mainly used to have a different sequences for train, validation and test
        :param implementation: Either 'torch' or 'numpy'
        :param kwargs:
        """
        super(RandomUniformVector, self).__init__()

        self.sizes = sizes
        self.seed_offset = seed_offset

        self.implementation = implementation

        if implementation == 'numpy':
            self.rs = np.random.RandomState()
            transform_steps = [ToTensor({'random_normal': 'from_numpy_per_item'}),
                               ToFloatTensor(['random_normal'], per_item=True)]

            self.transform = transforms.Compose(transform_steps)

    def __getitem__(self, idx):
        # I think we have to do this trick here to make sure that the random generator is both thread-safe and
        # deterministic. I implemented it in two ways, the numpy implementation should be thread-safe,
        # I am not exactly sure about the torch implementation
        if self.implementation == 'torch':
            # Pytorch does not allow to instantiate random generators so we set the seed temporarily
            # this may break when using multiple threads, but it is almost twice as fast as numpy and
            # that matters if you have to draw large tensors
            with temp_seed(self.seed_offset + idx, library='torch'):
                random_vectors = []
                for size in self.sizes:
                    random_vectors.append(torch.rand(*size, dtype=torch.float, requires_grad=False))
                sample = {'random_uniform': random_vectors}
                return sample
        elif self.implementation == 'numpy':
            # For numpy we can instantiate an random generator in the init and so i think this is safer
            # in a multi threading program
            self.rs.seed(self.seed_offset + idx)
            random_vectors = []
            for size in self.sizes:
                random_vectors.append(self.rs.rand(*size))
            sample = {'random_uniform': random_vectors}
            sample = self.transform(sample)
            return sample
        else:
            raise NotImplementedError

    def __len__(self):
        return sys.maxsize

    @property
    def finite(self) -> bool:
        return False


class RandomIntVector(Dataset):
    """
    Random generator wrapped in a random vector dataset
    """

    def __init__(self, sizes: tuple = ((1,),),
                 low: int = 0,
                 high: int = 10,
                 seed_offset: int = 0,
                 implementation: str = 'torch', **kwargs):
        """
        Dataset class that contains a random vector generator, whose elements are drawn from a normal distribution,
        commonly used for training GANs. We commonly need to have multiple vectors filled with random numbers.
        This is the reason why sizes is by default considered a tuple of tuples.n
        You can choose between two random generators, with different characteristics and execution speeds

        :param sizes: a tuple of tuples indicating the size of the vectors that need to be generated.
        The default is only one vector with 64 elements
        :param seed_offset: an offset added to the idx seed.
        This is mainly used to have a different sequences for train, validation and test
        :param implementation: Either 'torch' or 'numpy'
        :param kwargs:
        """
        super(RandomIntVector, self).__init__()

        self.low = low
        self.high = high

        self.sizes = sizes
        self.seed_offset = seed_offset

        self.implementation = implementation

        if implementation == 'numpy':
            self.rs = np.random.RandomState()
            transform_steps = [ToTensor({'randint_uniform': 'from_numpy_per_item'}),
                               ToLongTensor(['randint_uniform'], per_item=True)]

            self.transform = transforms.Compose(transform_steps)

    def __getitem__(self, idx):
        # I think we have to do this trick here to make sure that the random generator is both thread-safe and
        # deterministic. I implemented it in two ways, the numpy implementation should be thread-safe,
        # I am not exactly sure about the torch implementation
        if self.implementation == 'torch':
            # Pytorch does not allow to instantiate random generators so we set the seed temporarily
            # this may break when using multiple threads, but it is almost twice as fast as numpy and
            # that matters if you have to draw large tensors
            with temp_seed(self.seed_offset + idx, library='torch'):
                random_vectors = []
                for size in self.sizes:
                    if size == (1,):
                        random_vectors.append(torch.randint(self.low, self.high, size, dtype=torch.long, requires_grad=False))
                    else:
                        random_vectors.append(torch.randint(self.low, self.high, tuple(size), dtype=torch.long, requires_grad=False))
                sample = {'randint_uniform': random_vectors}
                return sample
        elif self.implementation == 'numpy':
            # For numpy we can instantiate an random generator in the init and so i think this is safer
            # in a multi threading program
            self.rs.seed(self.seed_offset + idx)
            random_vectors = []
            for size in self.sizes:
                random_vectors.append(self.rs.randint(self.low, self.high, *size))
            sample = {'randint_uniform': random_vectors}
            sample = self.transform(sample)
            return sample
        else:
            raise NotImplementedError

    def __len__(self):
        # TODO discuss here what we should do here, the length is the maximum range of the integer value,
        #  but if we use this dataset in a collection, the collection takes the maximum of the lengths of datasets
        #  inside the collection. I solved it now with the finite flag
        return sys.maxsize

    @property
    def finite(self) -> bool:
        """
        :return: Is this dataset finite?
        """
        return False

class RandomShuffleVector(Dataset):
    """
    Random shuffling of elements in a vector
    """

    def __init__(self, base_elements,
                 seed_offset: int = 0,
                 implementation: str = 'torch', **kwargs):
        """
        Dataset class that contains a random vector generator, whose elements are drawn from a normal distribution,
        commonly used for training GANs. We commonly need to have multiple vectors filled with random numbers.
        This is the reason why sizes is by default considered a tuple of tuples.n
        You can choose between two random generators, with different characteristics and execution speeds

        :param sizes: a tuple of tuples indicating the size of the vectors that need to be generated.
        The default is only one vector with 64 elements
        :param seed_offset: an offset added to the idx seed.
        This is mainly used to have a different sequences for train, validation and test
        :param implementation: Either 'torch' or 'numpy'
        :param kwargs:
        """
        super(RandomShuffleVector, self).__init__()

        self.seed_offset = seed_offset

        self.implementation = implementation

        if implementation == 'numpy':
            self.base_elements = np.array(base_elements)
            self.rs = np.random.RandomState()
            transform_steps = [ToTensor({'randint_uniform': 'from_numpy'}),
                               ToLongTensor(['randint_uniform'])]

            self.transform = transforms.Compose(transform_steps)

        elif implementation == 'torch':
            self.base_elements = torch.tensor(base_elements, dtype=torch.long, requires_grad=False)

    def __getitem__(self, idx):
        # I think we have to do this trick here to make sure that the random generator is both thread-safe and
        # deterministic. I implemented it in two ways, the numpy implementation should be thread-safe,
        # I am not exactly sure about the torch implementation
        if self.implementation == 'torch':
            # Pytorch does not allow to instantiate random generators so we set the seed temporarily
            # this may break when using multiple threads, but it is almost twice as fast as numpy and
            # that matters if you have to draw large tensors
            with temp_seed(self.seed_offset + idx, library='torch'):
                idcs = torch.randperm(self.base_elements.nelement())
                sample = {'rand_shuffled': self.base_elements[idcs].contiguous()}
                return sample
        elif self.implementation == 'numpy':
            # For numpy we can instantiate an random generator in the init and so i think this is safer
            # in a multi threading program
            self.rs.seed(self.seed_offset + idx)
            sample = {'rand_shuffled': np.random.shuffle(self.base_elements)}
            sample = self.transform(sample)
            return sample
        else:
            raise NotImplementedError

    def __len__(self):
        # TODO discuss here what we should do here, the length is the maximum range of the integer value,
        #  but if we use this dataset in a collection, the collection takes the maximum of the lengths of datasets
        #  inside the collection. I solved it now with the finite flag
        return sys.maxsize

    @property
    def finite(self) -> bool:
        """
        :return: Is this dataset finite?
        """
        return False
