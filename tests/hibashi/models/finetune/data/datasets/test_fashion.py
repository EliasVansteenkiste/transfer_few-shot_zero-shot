import os
from os import path as osp
from collections import Counter
from datetime import datetime

from hibashi.models.finetune.data.datasets.fashion import FashionFinetuneTrain, FashionFinetuneVal, FashionFinetuneTest
from hibashi.utils.io import ensure_dir, save_image


class TestFashionFinetuneDataIterators:
    """

    """
    path_temp_test_res = '/Users/elias/Downloads/fashion-dataset/temporary'

    def _test_data_iterator(self, data_gen_cls):
        temp_path = osp.join(self.path_temp_test_res, data_gen_cls.__name__)
        ensure_dir(temp_path)

        data_iterator = data_gen_cls(base_data_path='/Users/elias/Google Drive/datasets/fashion-dataset')

        counter = Counter()

        print(data_gen_cls.__name__, 'has', len(data_iterator), 'samples')

        for idx, sample in enumerate(data_iterator):

            if not idx % 1000:
                print(f'{datetime.now().strftime("%H:%M:%S")}: Processed {idx} images.')

            assert -1 < sample['cls_idx'] < 57
            assert sample['article_type'] in data_iterator.article_type_2_cls_idx
            counter[sample['article_type']] += 1

            if idx < 10:
                print(sample['image'].size(), sample['article_type'], sample['cls_idx'])
                sample_image = sample['image'].data.numpy().transpose(1, 2, 0)
                save_image(os.path.join(temp_path, f'{idx}.jpg'), sample_image)

        print('Frequency of categories (to calculate the weights):')
        print(counter)

    def test_finetune_train(self):
        self._test_data_iterator(FashionFinetuneTrain)

    def test_finetune_val(self):
        self._test_data_iterator(FashionFinetuneVal)

    def test_finetune_test(self):
        self._test_data_iterator(FashionFinetuneTest)

