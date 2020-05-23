"""
A script for testing trained models on the testset and calculate aggregated metrics.
Multiple trained weights can be tested at once
Metrics are reported for every trained weights seperately
"""

from os import path as osp

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.sampler import SequentialSampler

from hibashi.metrics.metrics import LossFromDict, TopKAccuracy
from hibashi.models.finetune.data.augmentations.img_aug import PadToSquareResize
from hibashi.models.finetune.data.datasets.fashion import FashionPretrainTest
from hibashi.models.finetune.finetune import Finetune


testset = FashionPretrainTest(base_data_path='/Users/elias/Google Drive/datasets/fashion-dataset',
                              aug_names=('PadToSquareResize',))

model = Finetune(gpu_ids=-1, is_train=False)


metrics = {'non balanced accuracy': LossFromDict(loss_name='loss_accuracy'),
           'Average top 1 accuracy': TopKAccuracy(num_classes=20, top_k=1),
           'Average top 5 accuracy': TopKAccuracy(num_classes=20, top_k=5)}

base_log_path = '/Users/elias/sideprojects/tensorboard_logs/pretrain/'
for path_rel in [
    '11-lr_schedule-CE-batch_size_128-no_aug/checkpoints/best_net_classifier_59_AverageTop1ErrorRatePretrain=0.03866385.pth',
]:

    print(path_rel)

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

    for i in range(20):
        print(f'Average top 1 accuracy for {model.cls_idx_2_label[i]}:',
              metrics['Average top 1 accuracy'].compute_per_idx(i))
        print(f'Average top 5 accuracy for {model.cls_idx_2_label[i]}:',
              metrics['Average top 5 accuracy'].compute_per_idx(i))

    for name, metric in metrics.items():
        print(f'{name}: {metric.compute().item()}')
        metric.reset()


