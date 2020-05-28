from sacred import Ingredient

from hibashi.data.datasets.datasets import Dataset
from hibashi.models.pretrain.data import datasets, augmentations

train = Ingredient('train')
train_data = Ingredient('train_data')
val_data = Ingredient('val_data')


@train.config
def train_cfg():
    connect_mongo_db = False
    connect_slack = False
    n_epochs = 160
    metadata_path = '/home/ubuntu/tensorboard_logs'  # '/Users/elias/Downloads'
    log_interval = 100
    img_log_interval = 1000
    eval_interval = 125  # Run evaluator every n iterations
    save_interval = 1
    save_n_last = 5
    overwrite_id_with = '15-lr_schedule-CE-batch_size_128-no_aug-no_pretraining'


@train_data.config
def train_data_cfg():
    name = 'FashionPretrainTrain'

    ds_params = Dataset.get_dataset_params(name)
    ds_params['base_data_path'] = '/mnt/ramdisk/fashion-dataset'  # '/Users/elias/Google Drive/datasets/fashion-dataset'
    ds_params['aug_names'] = ('PadToSquareResize',)
    ds_params['sampler'] = 'RandomSampler'
    train_data.add_config(ds_params)
    batch_size = 128
    n_workers = 64
    shuffle = True
    drop_last = True


@val_data.config
def val_data_cfg():
    names = ['FashionPretrainVal', ]  # the names of the datasets
    external_metrics = ['AverageTop1ErrorRatePretrain', 'AverageTop1ErrorRatePretrain',
                        'F1Jeans', 'F1PerfumeAndBodyMist', 'F1FormalShoes', 'F1Socks', 'F1Backpacks', 'F1Belts',
                        'F1Briefs', 'F1Sandals', 'F1FlipFlops', 'F1Wallets', 'F1Sunglasses', 'F1Heels', 'F1Handbags',
                        'F1Tops', 'F1Kurtas', 'F1SportShoes', 'F1Watches', 'F1CasualShoes', 'F1Shirts', 'F1Tshirts']
    main_dataset = 'FashionPretrainVal'  # the main validation dataset, will be used to track the best loss
    name = None
    for name in names:
        val_data.add_config({name: {
            'base_data_path': '/mnt/ramdisk/fashion-dataset',  # '/Users/elias/Google Drive/datasets/fashion-dataset',
            'aug_names': ('PadToSquareResize',),
            'sampler': 'SequentialSampler',
            'batch_size': 128,
            'n_workers': 32,
            'shuffle': False,
            'drop_last': False,
        }})

    del name
