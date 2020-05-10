from abc import ABCMeta
from torch.utils.data import Dataset as TorchDataset

from hibashi.framework.utils import get_subclass, get_parameter_of_cls

AGE_STDEV = 2.


class Dataset(TorchDataset):
    """
    Dataset parent class. All datasets should inherit from this class.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @staticmethod
    def all_datasets() -> dict:
        """
        Returns a dictionary with all the subclasses of this class.
        :return: key: class name, value: class type
        """
        datasets = {cls.__name__.lower(): cls for cls in Dataset.__subclasses__()}
        return datasets

    @staticmethod
    def get_cls(name: str) -> str:
        """
        Gets the class type of a class with a given name which is a subclass of dataset.
        :param name:
        :return:
        """
        datasets = Dataset.all_datasets()
        dataset = datasets.get(name.lower())
        if dataset is None:
            raise ValueError('Unknown dataset: "%s". Valid options are: %s' % (name.lower(), list(datasets.keys())))

        return dataset

    @property
    def name(self) -> str:
        """
        Returns the name of the class
        :return:
        """
        return type(self).__name__

    @classmethod
    def get_dataset_params(cls, name) -> dict:
        """
        Returns configurable parameters of a subclass
        :param name: sub class name
        :return: parameters in a dictionary
        """
        subclass = get_subclass(name, cls)
        params = get_parameter_of_cls(subclass)

        return params

    @property
    def finite(self):
        raise NotImplementedError

    def set_augmentations_and_transformations(self, aug_names: tuple):
        """
        Set the augmentations and compose the transformation steps
        :param aug_names: the names of the augmentation classes you want to apply
        :return: None
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Returns one batch
        :param idx: batch index
        :return:
        """
        raise NotImplementedError
