"""
This module contains the image augmentation classes for the age model
"""
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
