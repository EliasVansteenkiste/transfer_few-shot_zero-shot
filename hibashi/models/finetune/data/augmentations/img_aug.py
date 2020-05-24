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


class FlipLR(AugmentImage):
    def __init__(self, p=.5, key_source='image', key_target=None):
        super(FlipLR, self).__init__(key_source=key_source, key_target=key_target)

        self.sequence = iaa.Fliplr(p=p)


class Affine(AugmentImage):
    """
    Affine image augmentation, includes scaling, translation, rotation, shear
    """
    def __init__(self, key_source='image', key_target=None):
        super(Affine, self).__init__(key_source=key_source, key_target=key_target)

        def sometimes(aug):
            return iaa.Sometimes(0.5, aug)

        self.sequence = sometimes(iaa.Affine(
            scale={"x": (0.8, 1.4), "y": (0.8, 1.4)},  # scale images to 80-140% of their size, individually per axis
            translate_percent={"x": (-0.38, 0.38), "y": (-0.38, 0.38)},  # translate by -38 to +38 percent (per axis)
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-5, 5),  # shear
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=imgaug.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ))


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
