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
    n_epochs = 120
    metadata_path = '/content/gdrive/My Drive/tensorboard_logs'
    log_interval = 100
    img_log_interval = 1000
    eval_interval = 1000  # Run evaluator every n iterations
    save_interval = 1
    save_n_last = 5
    overwrite_id_with = 'test'


@train_data.config
def train_data_cfg():
    name = 'FashionPretrainTrain'

    ds_params = Dataset.get_dataset_params(name)
    ds_params['aug_names'] = ('PadToSquareResize',)
    ds_params['sampler'] = 'RandomSampler'
    train_data.add_config(ds_params)
    batch_size = 32
    n_workers = 3
    shuffle = True
    drop_last = True


@val_data.config
def val_data_cfg():
    names = ['FashionPretrainVal', ]  # the names of the datasets

    main_dataset = 'MNISTVal'  # the main validation dataset, will be used to track the best loss
    name = None
    for name in names:
        val_data.add_config({name: {
            'sampler': 'SequentialSampler',
            'batch_size': 32,
            'n_workers': 4,
            'shuffle': False,
            'drop_last': False,
        }})

    del name
