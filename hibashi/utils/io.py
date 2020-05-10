import os

import numpy as np
from skimage import io
import skimage
from skimage.transform import estimate_transform, warp


def ensure_dir(directory):
    """
    Creates a directory if it does not exist yet.
    :param directory: path to directory to be created
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def ensure_dir_from_filepath(filepath):
    """
    Creates a directory if it does not exist yet.
    :param filepath: file path for which the parent direct needs to be created
    :return:
    """
    parts = os.path.split(filepath)
    parent_dir_path = os.path.join(*parts[:-1])
    ensure_dir(parent_dir_path)


def read_image(path_absolute, fail_on_wrong_path=True, image_to_3_channels=True, verbose=False):
    """
    Read image from given path using skimage.io.imread function.
    :param path_absolute: absolute path to read the image from
    :param fail_on_wrong_path: Should the function raise an error if it cannot read the image? default: True
    :param image_to_3_channels: Should we check and set image channels to 3? default: True
    :return: image as numpy array or None if wrong path and fail_on_wrong_path is False
    """
    if verbose:
        print('Attempting to read the image at %s' % path_absolute)
    try:
        img = io.imread(path_absolute)

        # Fix for images with multiple modes, this happens for iphone live photos.
        # They contain different modes:
        #  - a base jpg image
        #  - a short Quicktime MOV
        #  - an HDR mode
        
        if img.shape == (3,):
            img = img[0]

        if image_to_3_channels:
            if len(img.shape) == 2:
                if verbose:
                    print('Image with shape {} is grayscale, extending to 3 channels.'.format(img.shape))
                img = np.stack((img,) * 3, -1)
            elif img.shape[2] == 2:
                if verbose:
                    print('Image with shape {} is grayscale with transparency'.format(img.shape) +
                             ', extending the grayscale to 3 channels and dropping the transparency channel')
                img = np.squeeze(img[:, :, :-1])
                img = np.stack((img,) * 3, -1)
            elif img.shape[2] == 3:
                pass  # This is a normal image, good.
            elif img.shape[2] == 4:
                if verbose:
                    print('Image with shape {} has 4 channels, dropping the transparency channel.'.format(img.shape))
                img = img[:, :, :-1]
            else:
                raise ValueError('Image has unexpected number of channels in shape {}. '.format(img.shape) +
                                 'Supported is grayscale, grayscaleA, RGB, RGBA')
        return img
    except Exception as e:
        print('Failed to read image from {} with exception: {}'.format(path_absolute, e))
        if fail_on_wrong_path:
            raise Exception(e)
        else:
            return None


def save_image(path_absolute, image, fail_on_wrong_path=True, verbose=False):
    """
    Save image into given path using skimage.io.imsave function.
    :param path_absolute: absolute path to save the image to
    :param image: image as numpy array
    :param fail_on_wrong_path: Should the function raise an error if it cannot save the image? default: True
    :return:
    """
    if verbose:
        print('Attempting to save the image at %s' % path_absolute)
    try:
        io.imsave(path_absolute, image, quality=100)
    except Exception as e:
        print('Failed to save image into {} with exception: {}'.format(path_absolute, e))
        if fail_on_wrong_path:
            raise Exception(e)


def crop_image(image, resolution=256):
    """
    Crops the image to a desired size (resolution) to a square with black edges (in case the image is not square)
    :param image: patch image
    :param resolution: desired size for the image
    :return: cropped image
    """
    bounding_box = BoundingBox((0, 0, image.shape[1], image.shape[0]))

    frame_image = skimage.img_as_float32(image)

    left = bounding_box.left
    right = bounding_box.right
    top = bounding_box.top
    bottom = bounding_box.bottom
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * 1.)

    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])

    dst_pts = np.array([[0, 0], [0, resolution - 1], [resolution - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)

    cropped_image = warp(frame_image, tform.inverse, output_shape=(resolution, resolution))
    return cropped_image, tform
