import os
from os import path as osp
from collections import Counter
from datetime import datetime

from hibashi.models.pretrain.data.datasets.fashion import FashionPretrainTrain, FashionPretrainVal, \
    FashionPretrainTest
from hibashi.utils.io import ensure_dir, save_image


class TestFashionDataIterators:
    """

    """
    path_temp_test_res = '/Users/elias/Downloads/fashion-dataset/temporary'

    article_type_2_cls_idx = {"Jeans": 19,
                              "Perfume and Body Mist": 18,
                              "Formal Shoes": 17,
                              "Socks": 16,
                              "Backpacks": 15,
                              "Belts": 14,
                              "Briefs": 13,
                              "Sandals": 12,
                              "Flip Flops": 11,
                              "Wallets": 10,
                              "Sunglasses": 9,
                              "Heels": 8,
                              "Handbags": 7,
                              "Tops": 6,
                              "Kurtas": 5,
                              "Sports Shoes": 4,
                              "Watches": 3,
                              "Casual Shoes": 2,
                              "Shirts": 1,
                              "Tshirts": 0}

    def _test_data_iterator(self, data_gen_cls):
        temp_path = osp.join(self.path_temp_test_res, data_gen_cls.__name__)
        ensure_dir(temp_path)

        data_iterator = data_gen_cls()

        counter = Counter()

        print(data_gen_cls.__name__, 'has', len(data_iterator), 'samples')

        for idx, sample in enumerate(data_iterator):

            if not idx % 1000:
                print(f'{datetime.now().strftime("%H:%M:%S")}: Processed {idx} images.')

            assert -1 < sample['cls_idx'] < 20
            assert sample['article_type'] in self.article_type_2_cls_idx
            counter[sample['article_type']] += 1

            if idx < 10:
                print(sample['image'].size(), sample['article_type'], sample['cls_idx'])
                sample_image = sample['image'].data.numpy().transpose(1, 2, 0)
                save_image(os.path.join(temp_path, f'{idx}.jpg'), sample_image)

        print(counter)

    def test_pretrain_train(self):
        self._test_data_iterator(FashionPretrainTrain)

    def test_pretrain_val(self):
        self._test_data_iterator(FashionPretrainVal)

    def test_pretrain_test(self):
        self._test_data_iterator(FashionPretrainTest)

