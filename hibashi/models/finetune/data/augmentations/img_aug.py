"""
This module contains the image augmentation classes for the age model
"""
import imgaug
from imgaug import augmenters as iaa

from hibashi.data.img_aug import AugmentImage


class PadToSquareResize(AugmentImage):
    """
    Image augment class that only takes care of the correct padding and size
    """

    def __init__(self, key_source='image', key_target=None):
        super(PadToSquareResize, self).__init__(key_source=key_source, key_target=key_target)

        self.sequence = iaa.Sequential([
            # TODO: write your own padding function that uses the left top pixel value
            #  since we have some border replicate artifacts
            iaa.size.PadToSquare(pad_mode='edge'),
            iaa.Resize({"height": 128, "width": 128})
        ])

    def augment_img(self, image):
        """
        Apply augmentations
        :param image: The image that needs to be augmented
        :return: the augmented image
        """
        return self.sequence.augment_image(image)


class FlipLR(AugmentImage):
    def __init__(self, p=.5, key_source='image', key_target=None):
        super(FlipLR, self).__init__(key_source=key_source, key_target=key_target)

        self.sequence = iaa.Fliplr(p=p)

    def augment_img(self, image):
        """
        Apply augmentations
        :param image: The image that needs to be augmented
        :return: the augmented image
        """
        return self.sequence.augment_image(image)


class RandomResizedCropFlip(AugmentImage):
    """
    Image augment class that only takes care of the correct padding and size
    """

    def __init__(self, key_source='image', key_target=None):
        super(RandomResizedCropFlip, self).__init__(key_source=key_source, key_target=key_target)

        self.sequence = iaa.Sequential([
            iaa.size.Crop(percent=((0, .3), (0, .3), (0, .3), (0, .3))),
            iaa.size.PadToSquare(pad_mode=imgaug.ALL),
            iaa.Resize({"height": 128, "width": 128}),
            iaa.Fliplr(p=.5)
        ])

    def augment_img(self, image):
        """
        Apply augmentations
        :param image: The image that needs to be augmented
        :return: the augmented image
        """
        return self.sequence.augment_image(image)


class RandomColorJitter(AugmentImage):
    """
    Image augment class that only takes care of the correct padding and size
    """

    def __init__(self, key_source='image', key_target=None):
        super(RandomColorJitter, self).__init__(key_source=key_source, key_target=key_target)

        self.sequence = iaa.Sequential([
            iaa.Sometimes(0.8, iaa.Sequential([
                iaa.MultiplyBrightness((0.8, 1.25)),
                iaa.MultiplyHueAndSaturation(mul_hue=(0.8, 1.25), mul_saturation=(0.8, 1.25))
            ])),
            iaa.Sometimes(0.2, iaa.Grayscale())
        ])

    def augment_img(self, image):
        """
        Apply augmentations
        :param image: The image that needs to be augmented
        :return: the augmented image
        """
        return self.sequence.augment_image(image)