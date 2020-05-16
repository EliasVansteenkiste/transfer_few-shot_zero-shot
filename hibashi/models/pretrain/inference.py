"""
A script for testing trained models on the testset and calculate aggregated metrics.
Multiple trained weights can be tested at once
Metrics are reported for every trained weights seperately
"""

from os import path as osp

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.sampler import SequentialSampler

from hibashi.metrics.metrics import LossFromDict
from hibashi.models.pretrain.data.augmentations.img_aug import PadToSquareResize
from hibashi.models.pretrain.data.datasets.fashion import FashionPretrainTest
from hibashi.models.pretrain.pretrain import PreTrain

testset = FashionPretrainTest(base_data_path='/Users/elias/Google Drive/datasets/fashion-dataset',
                              aug_names=('PadToSquareResize',))

model = PreTrain(gpu_ids=-1, is_train=False)


metrics = {'non balanced accuracy': LossFromDict(loss_name='loss_accuracy')}

base_log_path = '/Users/elias/Google Drive/tensorboard_logs/pretrain/'
for path_rel in ['1-first_run_fixed_lr/checkpoints/last_net_classifier_120.pth',
                 '5-RandomResizedCropFlip-constant_lr_2e-4/checkpoints/last_net_classifier_96.pth']:

    model.load_weights(osp.join(base_log_path, path_rel))
    model.eval()

    data_loader = DataLoader(testset,
                             batch_size=32,
                             sampler=SequentialSampler(testset),
                             num_workers=10,
                             drop_last=False,
                             collate_fn=default_collate,
                             pin_memory=True)

    for idx, sample in enumerate(data_loader):
        model.set_input(sample)
        model.test()
        output = model.state

        for metric in metrics.values():
            metric.update(output)

    for name, metric in metrics.items():
        print(f'{name}: {metric.compute().item()}')
        metric.reset()





