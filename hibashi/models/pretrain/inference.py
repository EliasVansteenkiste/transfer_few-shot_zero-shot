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
from hibashi.models.pretrain.data.augmentations.img_aug import PadToSquareResize
from hibashi.models.pretrain.data.datasets.fashion import FashionPretrainTest
from hibashi.models.pretrain.pretrain import PreTrain
from hibashi.models.pretrain.utils import plot_confusion_matrix_validation
from hibashi.utils.io import ensure_dir

# paths to set
base_log_path = '/content/gdrive/My Drive/imc/tensorboard_logs/pretrain'
save_path = '/content/gdrive/My Drive/imc/test_results/pretrain'
ensure_dir(save_path)

testset = FashionPretrainTest(base_data_path='/home/fashion-dataset',
                              aug_names=('PadToSquareResize',))

model = PreTrain(gpu_ids=[0], is_train=False)

metrics = {'non balanced accuracy': LossFromDict(loss_name='loss_accuracy'),
           'Average top 1 accuracy': TopKAccuracy(num_classes=20, top_k=1),
           'Average top 5 accuracy': TopKAccuracy(num_classes=20, top_k=5),
           }

non_scalar_metrics = {'Confusion Matrix': LossFromDict(loss_name='non_scalar_metric_cm', reduce=False)}


for path_rel in [
    '01-fixed_lr-CE/checkpoints/best_net_classifier_26_loss_error_rate=0.06415929.pth',
    '02-fixed_lr-CE-batch_size_64/checkpoints/best_net_classifier_16_loss_error_rate=0.06743421.pth',
    '03-fixed_lr-CE-batch_size_128/checkpoints/best_net_classifier_43_loss_error_rate=0.06498894.pth',
    '04-lr_schedule-CE-batch_size_64/checkpoints/best_net_classifier_53_loss_error_rate=0.05088495.pth',
    '05-fixed_lr_schedule-CE-batch_size_256/checkpoints/best_net_classifier_10_loss_error_rate=0.06554203.pth',
    '07-lr_schedule-CE-batch_size_128/checkpoints/best_net_classifier_1_AverageTop1ErrorRatePretrain=0.05263158.pth',
    '08-5k_lr_schedule-CE-batch_size_256/checkpoints/best_net_classifier_1_AverageTop1ErrorRatePretrain=0.05263158.pth',
    '09-lr_schedule-weighted_CE-batch_size_128/checkpoints/best_net_classifier_1_AverageTop1ErrorRatePretrain=0.05263158.pth',
    '10-lr_schedule-CE-batch_size_128-fliplr_only/checkpoints/best_net_classifier_1_AverageTop1ErrorRatePretrain=0.05263158.pth',
    '11-lr_schedule-CE-batch_size_128-no_aug/checkpoints/best_net_classifier_38_AverageTop1ErrorRatePretrain=0.0423285.pth',
    '11-lr_schedule-CE-batch_size_128-no_aug/checkpoints/best_net_classifier_59_AverageTop1ErrorRatePretrain=0.03866385.pth',
    '12-lr_schedule-CE-batch_size_128-RandomColorJitter/checkpoints/best_net_classifier_26_AverageTop1ErrorRatePretrain=0.04507069.pth',
    '12-lr_schedule-CE-batch_size_128-RandomColorJitter/checkpoints/best_net_classifier_67_AverageTop1ErrorRatePretrain=0.04180649.pth',
    '13-lr_schedule-CE-batch_size_256-no_aug/checkpoints/best_net_classifier_62_AverageTop1ErrorRatePretrain=0.04636856.pth',
    '14-lr_schedule-CE-batch_size_128-no_aug-init_lin_layer/checkpoints/best_net_classifier_94_AverageTop1ErrorRatePretrain=0.04102252.pth',
]:

    print(path_rel)
    experiment_model_id = path_rel.replace('/checkpoints/', '')

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
        for metric in non_scalar_metrics.values():
            metric.update(output)

    for name, metric in metrics.items():
        print(f'{name}: {metric.compute().item()}')

    article_type_2_top_1 = {}
    article_type_2_top_5 = {}
    for i in range(20):
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

    fig = plot_confusion_matrix_validation(non_scalar_metrics['Confusion Matrix'].compute(), model.criterion_cm.labels)
    fig.savefig(osp.join(exp_path, 'confusion_matrix.jpg'))

    for metric in metrics.values():
        metric.reset()
    for metric in non_scalar_metrics.values():
        metric.reset()


