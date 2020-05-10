import numpy as np


def make_grid(arr, nrow=8, ncol=8, padding=2,
              normalize=False, norm_range=None, scale_each=False, pad_value=0):
    """
    Make a grid of images.
    Source: See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`
    :param arr: (np.ndarray, list): 4D mini-batch Tensor of shape (B x H x W x C)
            or a list of images all of the same n_feat_base
    :param nrow: (int, optional): Number of images displayed in each row of the grid.
            The Final grid n_feat_base is (B / nrow, nrow). Default is 8
    :param ncol: (int, optional): Number of images displayed in each column of the grid.
            The Final grid n_feat_base is (B / nrow, nrow). Default is 8
    :param padding: (int, optional): amount of padding. Default is 2
    :param normalize: (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value
    :param norm_range: (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor
    :param scale_each: (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images
    :param pad_value: (float, optional): Value for the padded pixels
    :return: (np.ndarray): generated grid of input images
    """
    if isinstance(arr, list):
        arr = np.array(arr)

    if len(arr.shape) == 2:  # single image H x W
        arr = arr.reshape((1, *arr.shape, 1))
    if len(arr.shape) == 4 and arr.shape[-1] == 1:  # single-channel images
        arr = np.concatenate((arr, arr, arr), -1)
    if len(arr.shape) == 4 and arr.shape[-1] == 4:  # alpha
        arr = arr[:, :, :, :-1]

    if normalize is True:
        arr = arr.copy()  # avoid modifying tensor in-place
        if norm_range is not None:
            assert isinstance(norm_range, tuple), \
                "Range has to be a tuple (min, max) if specified. Min and Max are numbers"

        def norm_ip(img, min, max):
            # img = np.clip(img, a_min=min, a_max=max)
            if (max - min) < 1e-8:
                eps = 1e-8
            else:
                eps = 0

            img = (img - min) / (max - min + eps)
            return img

        def norm_to_range(t, norm_range):
            if norm_range is not None:
                return norm_ip(t, norm_range[0], norm_range[1])
            else:
                return norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for i, t in enumerate(arr):  # loop over mini-batch dimension
                arr[i] = norm_to_range(t, norm_range)
        else:
            arr = norm_to_range(arr, norm_range)

    # make the mini-batch of images into a grid
    nmaps = arr.shape[0]
    xmaps = min(nrow, nmaps)
    if ncol is not None:
        ymaps = ncol
    else:
        ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(arr.shape[1] + padding), int(arr.shape[2] + padding)
    grid = np.full((height * ymaps + padding, width * xmaps + padding, 3), pad_value, dtype=arr.dtype)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[y * height + padding: y * height + padding + height - padding,
            x * width + padding:x * width + padding + width - padding] = arr[k]
            k += 1
    return grid