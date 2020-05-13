"""
This module contains the basic image augmentation base class definition
"""
from abc import abstractmethod

from hibashi.data.aug import Augment


class AugmentImage(Augment):

    def __init__(self, key_source='image', key_target=None):
        super(AugmentImage, self).__init__()
        self.key_source = key_source
        if key_target is None:
            self.key_target = key_source
        else:
            self.key_target = key_target

    @abstractmethod
    def augment_img(self, image):
        """
        This class needs to be implemented in all child classes
        :param image: The image that needs to be augmented
        :return: the augmented image
        """
        pass

    def __call__(self, sample):
        """
        The call function just takes out the image and passes it along
        :param sample: (dict) with at least the image (key) inside
        :return: the sample dictionary with the augmented dictionary
        """
        sample[self.key_target] = self.augment_img(sample[self.key_source])
        return sample


class AugmentImageMask(Augment):

    @abstractmethod
    def augment_img_mask(self, image, mask):
        """
        This class needs to be implemented in all child classes
        :param image: The image that needs to be augmented
        :param mask: the corresponding mask that needs to be augmented
        :return: the augmented image
        """
        pass

    def __call__(self, sample):
        """
        The call function just takes out the tuple image and maks and passes it along
        :param sample: (dict) with at least the image (key) inside
        :return: the sample dictionary with the augmented dictionary
        """
        sample['image'], sample['target_mask'] = self.augment_img_mask(sample['image'], sample['target_mask'])
        return sample
