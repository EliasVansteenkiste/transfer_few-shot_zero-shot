"""
Trim "boring" borders

based on the following StackOverflow answer
https://stackoverflow.com/a/44719803/2047558

"""
import pyvips
import os
from os import path as osp
from datetime import datetime

from hibashi.utils.io import ensure_dir


def trim_borders(path_img_source: str, path_img_target: str, threshold: float = 30) -> None:
    """
    An equivalent of ImageMagick's -trim in libvips ... automatically remove "boring" image edges.

    We use .project to sum the rows and columns of a 0/255 mask image, the first
    non-zero row or column is the object edge. We make the mask image with an
    amount-different-from-background image plus a threshold.

    :param path_img_source: path to the source image
    :param path_img_target: path where the image result will be saved
    :param threshold: threshold value to indicate interesting content
    :return:
    """
    # We use .project to sum the rows and columns of a 0/255 mask image, the first
    # non-zero row or column is the object edge. We make the mask image with an
    # amount-different-from-background image plus a threshold.

    im = pyvips.Image.new_from_file(path_img_source)

    # find the value of the pixel at (0, 0) ... we will search for all pixels
    # significantly different from this
    background = im(0, 0)

    # we need to smooth the image, subtract the background from every pixel, take
    # the absolute value of the difference, then threshold
    mask = (im.median(3) - background).abs() > threshold

    # sum mask rows and columns, then search for the first non-zero sum in each
    # direction
    columns, rows = mask.project()

    # .profile() returns a pair (v-profile, h-profile)
    left = columns.profile()[1].min()
    right = columns.width - columns.fliphor().profile()[1].min()
    top = rows.profile()[0].min()
    bottom = rows.height - rows.flipver().profile()[0].min()

    # and now crop the original image
    im = im.crop(left, top, right - left, bottom - top)

    im.write_to_file(path_img_target)


def process_images(path_source: str, path_target: str) -> None:
    """
    loop through all the images found in the source path

    :param path_source: path to directory containing original source images
    :param path_target: path where the processed images will be saved
    :return:
    """

    ensure_dir(path_target)

    bad_images = set(['44998.jpg'])  # empty or bad images

    for idx, filename in enumerate(os.listdir(path_source)):

        if filename not in bad_images:

            if not idx % 1000:
                print(f'{datetime.now().strftime("%H:%M:%S")}: Processed {idx} images.')

            if idx > 40000:
                print(filename)
                trim_borders(osp.join(path_source, filename), osp.join(path_target, filename))


if __name__ == "__main__":


    """
    After first initial tests, the trimming works well except for images with stronger background gradients or 
    images containing products that have exacltly the same color as the background.
    
    I fine tuned the threshold. A too high threshold cuts off too much, so there is a trade-off to be made here.
    """
    s_path = '/Users/elias/Downloads/fashion-dataset/images'
    test_path = '/Users/elias/Downloads/fashion-dataset/test_trimming'
    ensure_dir(test_path)
    img_ids_difficult_examples = [4983, 4969, 4668, 4953, 4941, 4940, 5257, 1637, 2507, 2510, 2511, 3183, 7124, 7737,
                                  9126, 9254, 12447, 13002, 14977, 15706, 28024, 30469, 42829, 44012, 44662, 48990,
                                  54612, 56049, 59583, 59563, 49743, 44057, 55835]

    for img_id in img_ids_difficult_examples:
        trim_borders(osp.join(s_path, str(img_id)+'.jpg'), osp.join(test_path, str(img_id)+'.jpg'), threshold=35)

    process_images('/Users/elias/Downloads/fashion-dataset/images',
                   '/Users/elias/Downloads/fashion-dataset/trimmed_images')

