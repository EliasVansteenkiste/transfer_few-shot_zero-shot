"""
A script for testing trained models on the testset and calculate aggregated metrics.
Multiple trained weights can be tested at once
Metrics are reported for every trained weights seperately
"""
import pickle
from os import path as osp

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.sampler import SequentialSampler

from hibashi.metrics.metrics import LossFromDict, TopKAccuracy
from hibashi.models.finetune.data.augmentations.img_aug import PadToSquareResize
from hibashi.models.finetune.data.datasets.fashion import FashionFinetuneTest
from hibashi.models.finetune.finetune import Finetune
from hibashi.models.finetune.utils import plot_confusion_matrix_validation
from hibashi.utils.io import ensure_dir

# paths to set
# base_log_path = '/Users/elias/sideprojects/tensorboard_logs/finetune/'
# save_path = '/Users/elias/Google Drive/imc/test_results'
base_log_path = '/home/ubuntu/tensorboard_logs/finetune/'
save_path = '/home/ubuntu/imc/test_results'
ensure_dir(save_path)

# base_data_path = '/Users/elias/Google Drive/datasets/fashion-dataset'
base_data_path = '/home/ubuntu/fashion-dataset'

test_set = FashionFinetuneTest(base_data_path=base_data_path,
                               aug_names=('PadToSquareResize',))

data_loader = DataLoader(test_set,
                         batch_size=32,
                         sampler=SequentialSampler(test_set),
                         num_workers=10,
                         drop_last=False,
                         collate_fn=default_collate,
                         pin_memory=True)

model = Finetune(gpu_ids=[0], is_train=False)

metrics = {'non balanced accuracy': LossFromDict(loss_name='loss_accuracy'),
           'Average top 1 accuracy': TopKAccuracy(num_classes=len(model.article_type_2_cls_idx), top_k=1),
           'Average top 5 accuracy': TopKAccuracy(num_classes=len(model.article_type_2_cls_idx), top_k=5),
           }

non_scalar_metrics = {'Confusion Matrix': LossFromDict(loss_name='non_scalar_metric_cm', reduce=False)}


for path_rel in [
    # '01-resnet50_feats/checkpoints/best_net_classifier_31_AverageTop1ErrorRateFinetune=0.2194643.pth',
    # '02-first_run_pretrained/checkpoints/best_net_classifier_30_AverageTop1ErrorRateFinetune=0.2202502.pth',
    # '04-FlipLR/checkpoints/best_net_classifier_20_AverageTop1ErrorRateFinetune=0.2226842.pth',
    # '05-FlipLR-Affine/checkpoints/best_net_classifier_13_AverageTop1ErrorRateFinetune=0.2396111.pth',
    # '06-new_lr_schedule_500/checkpoints/best_net_classifier_27_AverageTop1ErrorRateFinetune=0.2301688.pth',
    # '07-loss_cls_weighted/checkpoints/best_net_classifier_27_AverageTop1ErrorRateFinetune=0.2301688.pth',
    # '09-CE-lr_schedule_500-only_train_fc_and_enc_layer3_layer4/checkpoints/best_net_classifier_25_AverageTop1ErrorRateFinetune=0.2265884.pth',
    # '10-CE-differential_lr_2_3_4_5_6/checkpoints/best_net_classifier_18_AverageTop1ErrorRateFinetune=0.2438305.pth',
    # '11-CE-differential_lr_2_4_4_6_6/checkpoints/best_net_classifier_18_AverageTop1ErrorRateFinetune=0.2438305.pth',
    # '12-ce_weighted-lr_schedule-FlipLR_Affine/checkpoints/best_net_classifier_12_AverageTop1ErrorRateFinetune=0.2450421.pth',
    # '13-ce_weighted-lr_schedule-FlipLR_Affine-2layerfc/checkpoints/best_net_classifier_18_AverageTop1ErrorRateFinetune=0.2244431.pth',
    # "14-ce_weighted-lr_schedule-RandomColorJitter_Affine_FlipLR/checkpoints/best_net_classifier_23_AverageTop1ErrorRateFinetune=0.209832.pth",
    # "15-ce_weighted-lr_schedule-RandomColorJitter_Affine_FlipLR-imagenet_pretrained/checkpoints/best_net_classifier_12_AverageTop1ErrorRateFinetune=0.1985388.pth",
    '16-ce_weighted-lr_schedule_Affine_FlipLR-imagenet_pretrained/checkpoints/best_net_classifier_21_AverageTop1ErrorRateFinetune=0.206297.pth',
]:

    print(path_rel)
    experiment_model_id = path_rel.replace('/checkpoints/', '')

    model.load_weights(osp.join(base_log_path, path_rel))
    model.eval()

    for idx, sample in enumerate(data_loader):
        model.set_input(sample)
        model.test()
        output = model.state

        for metric in metrics.values():
            metric.update(output)
        for metric in non_scalar_metrics.values():
            metric.update(output)

    for name, metric in metrics.items():
        print(f'{name}: {metric.compute().item()}')

    article_type_2_top_1 = {}
    article_type_2_top_5 = {}
    for i in range(len(model.article_type_2_cls_idx)):
        article_type = model.cls_idx_2_article_type[i]
        top_1 = metrics["Average top 1 accuracy"].compute_per_idx(i).item()
        top_5 = metrics["Average top 5 accuracy"].compute_per_idx(i).item()
        article_type_2_top_1[article_type] = top_1
        article_type_2_top_5[article_type] = top_5
        print(f'Average top 1 accuracy for {article_type}: {top_1:.4f}')
        print(f'Average top 5 accuracy for {article_type}: {top_5:.4f}')

    exp_path = osp.join(save_path, experiment_model_id)
    ensure_dir(exp_path)

    pickle.dump(article_type_2_top_1, open(osp.join(exp_path, "article_type_2_top_1.pickle"), "wb"))
    pickle.dump(article_type_2_top_5, open(osp.join(exp_path, "article_type_2_top_5.pickle"), "wb"))

    fig = plot_confusion_matrix_validation(
        cm=non_scalar_metrics['Confusion Matrix'].compute(),
        labels=model.criterion_cm.labels,
        figsize=(8, 8),
    )
    fig.savefig(osp.join(exp_path, 'confusion_matrix.png'))

    for metric in metrics.values():
        metric.reset()
    for metric in non_scalar_metrics.values():
        metric.reset()
