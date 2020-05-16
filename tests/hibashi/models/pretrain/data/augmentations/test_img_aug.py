import os
from os import path as osp

from hibashi.models.pretrain.data.datasets.fashion import FashionPretrainTrain, FashionPretrainVal, \
    FashionPretrainTest
from hibashi.models.pretrain.data.augmentations.img_aug import PadToSquareResize
from hibashi.utils.io import ensure_dir, save_image


class TestPadToSquareResize:

    path_temp_test_res = '/Users/elias/Downloads/fashion-dataset/temporary'

    def _test_data_iterator(self, data_gen_cls):
        temp_path = osp.join(self.path_temp_test_res, 'PadToSquareResize', data_gen_cls.__name__)
        ensure_dir(temp_path)

        data_iterator = data_gen_cls(aug_names=('PadToSquareResize',))
        for idx, sample in enumerate(data_iterator):
            if idx < 20:
                sample_image = sample['image'].data.numpy().transpose(1, 2, 0)
                save_image(os.path.join(temp_path, f'{idx}.jpg'), sample_image)
            else:
                break
            assert sample['image'].size() == (3, 128, 128)
            assert sample['cls_idx'].size() == (1,)

    def test_pretrain_train(self):
        self._test_data_iterator(FashionPretrainTrain)

    def test_pretrain_val(self):
        self._test_data_iterator(FashionPretrainVal)

    def test_pretrain_test(self):
        self._test_data_iterator(FashionPretrainTest)


class TestRandomResizedCropFlip:

    path_temp_test_res = '/Users/elias/Downloads/fashion-dataset/temporary'

    def _test_data_iterator(self, data_gen_cls):
        temp_path = osp.join(self.path_temp_test_res, 'RandomResizedCropFlip', data_gen_cls.__name__)
        ensure_dir(temp_path)

        data_iterator = data_gen_cls(base_data_path='/Users/elias/Google Drive/datasets/fashion-dataset',
                                     aug_names=('RandomResizedCropFlip',))
        for idx, sample in enumerate(data_iterator):
            if idx < 20:
                sample_image = sample['image'].data.numpy().transpose(1, 2, 0)
                save_image(os.path.join(temp_path, f'{idx}.jpg'), sample_image)
            else:
                break
            assert sample['image'].size() == (3, 128, 128)
            assert sample['cls_idx'].size() == (1,)

    def test_pretrain_train(self):
        self._test_data_iterator(FashionPretrainTrain)

    def test_pretrain_val(self):
        self._test_data_iterator(FashionPretrainVal)

    def test_pretrain_test(self):
        self._test_data_iterator(FashionPretrainTest)
