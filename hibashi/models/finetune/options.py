from sacred import Ingredient

from hibashi.data.datasets.datasets import Dataset
from hibashi.models.finetune.data import datasets, augmentations

train = Ingredient('train')
train_data = Ingredient('train_data')
val_data = Ingredient('val_data')


@train.config
def train_cfg():
    connect_mongo_db = False
    connect_slack = False
    n_epochs = 160
    metadata_path = '/home/ubuntu/tensorboard_logs'  # '/Users/elias/Downloads' #'/home/ubuntu/tensorboard_logs'
    log_interval = 100
    img_log_interval = 1000
    eval_interval = 125  # Run evaluator every n iterations
    save_interval = 1
    save_n_last = 5
    overwrite_id_with = '16-ce_weighted-lr_schedule_Affine_FlipLR-imagenet_pretrained'


@train_data.config
def train_data_cfg():
    name = 'FashionFinetuneTrain'

    ds_params = Dataset.get_dataset_params(name)
    ds_params['base_data_path'] = '/mnt/ramdisk/fashion-dataset' # '/Users/elias/Google Drive/datasets/fashion-dataset' # '/mnt/ramdisk/fashion-dataset'
    ds_params['aug_names'] = ('PadToSquareResize', 'Affine', 'FlipLR')
    ds_params['sampler'] = 'RandomSampler'
    train_data.add_config(ds_params)
    batch_size = 128
    n_workers = 64
    shuffle = True
    drop_last = True


@val_data.config
def val_data_cfg():
    names = ['FashionFinetuneVal', ]  # the names of the datasets
    external_metrics = ['AverageAccuracyFinetune', 'AverageTop1ErrorRateFinetune']
    main_dataset = 'FashionFinetuneVal'  # the main validation dataset, will be used to track the best loss
    name = None
    for name in names:
        val_data.add_config({name: {
            'base_data_path': '/mnt/ramdisk/fashion-dataset', #'/Users/elias/Google Drive/datasets/fashion-dataset', #'/mnt/ramdisk/fashion-dataset',
            'aug_names': ('PadToSquareResize',),
            'sampler': 'SequentialSampler',
            'batch_size': 128,
            'n_workers': 32,
            'shuffle': False,
            'drop_last': False,
        }})

    del name
