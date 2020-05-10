"""
This module contains the class Trainer for the training of the Network.
"""
import importlib
import os
import pprint

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from hibashi.data.datasets.datasets import Dataset
from hibashi.data.samplers import *
from hibashi.framework.utils import get_subclass, get_meta_dir
from hibashi.metrics.metrics import *
from hibashi.models import Model
from hibashi.data.datasets import collate


class Trainer:
    """
    Class which setups the training logic which mainly involves defining callback handlers and attaching them to
    the training loop.
    """

    def __init__(self, model, config, evaluator, data_loader, tb_writer, run_info, logger, checkpoint_dir):
        """
        Creates a new trainer object for training a model.
        :param model: model to train. Needs to inherit from the BaseModel class.
        :param config: dictionary containing the whole configuration of the experiment
        :param evaluator: Instance of the evaluator class, used to run evaluation on a specified schedule
        :param data_loader: pytorch data loader providing the training data
        :param tb_writer: tensorboardX summary writer
        :param run_info: sacred run info for loging training progress
        :param logger: python logger object
        :param checkpoint_dir: directory path for storing checkpoints
        """
        self.run_info = run_info
        self.logger = logger
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.engine = Engine(self._step)
        self.model = model
        self.config = config
        self.train_cfg = config['train']
        self.tb_writer = tb_writer

        self.pbar = ProgressBar(ascii=True, desc='* Epoch')
        self.timer = Timer(average=True)
        self.save_last_checkpoint_handler = ModelCheckpoint(checkpoint_dir, 'last',
                                                            save_interval=self.train_cfg['save_interval'],
                                                            n_saved=self.train_cfg['save_n_last'],
                                                            require_empty=False)

        self.add_handler()

    def run(self):
        """
        Start the training loop which will run until all epochs are complete
        :return:
        """
        self.engine.run(self.data_loader, max_epochs=self.train_cfg['n_epochs'])

    def add_handler(self):
        """
        Adds all the callback handlers to the trainer engine. Should be called in the end of the init.
        :return:
        """
        # Learning rate decay
        for lr_s in self.model.schedulers:
            self.engine.add_event_handler(Events.ITERATION_STARTED, lr_s)

        # Checkpoint saving
        self.engine.add_event_handler(Events.EPOCH_STARTED,
                                      self.save_last_checkpoint_handler,
                                      self.model.networks)

        # Progbar
        monitoring_metrics = self.model.metric_names
        for mm in monitoring_metrics:
            RunningAverage(output_transform=self._extract_loss(mm)).attach(self.engine, mm)
        self.pbar.attach(self.engine, metric_names=monitoring_metrics)

        # Timer
        self.timer.attach(self.engine, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                          pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

        # Logging
        self.engine.add_event_handler(Events.ITERATION_COMPLETED, self._handle_log_train_results)
        self.engine.add_event_handler(Events.ITERATION_COMPLETED, self._handle_log_train_images)
        self.engine.add_event_handler(Events.ITERATION_COMPLETED, self._handle_run_evaluation)
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, self._handle_print_times)

        # Exception handling
        self.engine.add_event_handler(Events.EXCEPTION_RAISED, self._handle_exception)

    def _step(self, engine, batch):
        """
        Definition of a single training step. This function gets automatically called by the engine every iteration.
        :param engine: trainer engine
        :param batch: one batch provided by the dataloader
        :return:
        """
        self.model.train()
        self.model.set_input(batch)
        self.model.optimize_parameters()
        return self.model.state

    def _handle_log_train_results(self, engine):
        """
        Handler for writing the losses to tensorboard and sacred.
        :param engine: train engine
        :return:
        """
        if (engine.state.iteration - 1) % self.train_cfg['log_interval'] == 0:
            metrics = engine.state.metrics  # does not include non scalar metrics, since loggers can not handle this

            for m_name, m_val in metrics.items():
                if m_val is None:
                    raise ValueError(f'Value for {m_name} is None')
                self.run_info.log_scalar("train.%s" % m_name, m_val, engine.state.iteration)
                self.tb_writer.add_scalar("train/%s" % m_name, m_val, engine.state.iteration)

            for lr_name, lr_val in self.model.learning_rates.items():
                if lr_val is None:
                    raise ValueError(f'Value for {lr_name} is None')
                self.run_info.log_scalar("train.%s" % lr_name, lr_val, engine.state.iteration)
                self.tb_writer.add_scalar("train/%s" % lr_name, lr_val, engine.state.iteration)

    def _handle_log_train_images(self, engine):
        """
        Handler for writing visual samples to tensorboard.
        :param engine: train engine
        :return:
        """
        if (engine.state.iteration - 1) % self.train_cfg['img_log_interval'] == 0:
            for name, visual in self.model.visuals.items():
                # TODO remove the visual.transpose here and put it in the visualization function of the models
                self.tb_writer.add_image('train/%s' % name,
                                         visual.transpose(2, 0, 1),
                                         engine.state.iteration)
            for name, figure in self.model.figures.items():
                self.tb_writer.add_figure('train_metrics/%s' % name,
                                          figure,
                                          engine.state.iteration)

    def _handle_run_evaluation(self, engine):
        """
        Handler which will execute evaluation by running the evaluator object.
        :param engine: train engine
        :return:
        """
        if (engine.state.iteration - 1) % self.train_cfg['eval_interval'] == 0:
            self.evaluator.run()

    def _handle_exception(self, engine, e):
        """
        Exception handler which ensures that the model gets saved when stopped through a keyboard interruption.
        :param engine: train engine
        :param e: the exception which caused the training to stop
        :return:
        """
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            self.logger.warning('KeyboardInterrupt caught. Exiting gracefully.')
            self.save_last_checkpoint_handler(engine, self.model.networks)
        else:
            raise e

    def _handle_print_times(self, engine):
        """
        Handler for logging timer information for different training and evaluation steps.
        :param engine: train engine
        :return:
        """
        self.logger.info('Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, self.timer.value()))
        self.timer.reset()

    @staticmethod
    def _extract_loss(key):
        """
        Helper method to return losses for the RunningAverage
        :param key: (str) loss name
        :return: (fn) for the corresponding key
        """
        def _func(losses):
            return losses[key]
        return _func


class Evaluator:
    """
    Class which setups the evaluation logic which mainly involves defining callback handlers and attaching them to
    the evaluation loop.
    """

    def __init__(self, model, config, data_loaders, tb_writer, run_info, logger, checkpoint_dir):
        """
        Creates a new evaluator object for evaluating a model.
        :param model: model to train. Needs to inherit from the BaseModel class.
        :param config: dictionary containing the whole configuration of the experiment
        :param data_loaders: (dictionary) the keys represent the name and each value contains
         a pytorch data loader providing the validation data
        :param tb_writer: tensorboardX summary writer
        :param run_info: sacred run info for loging training progress
        :param logger: python logger object
        :param checkpoint_dir: directory path for storing checkpoints
        """
        self.run_info = run_info
        self.logger = logger
        self.data_loaders = data_loaders
        self.config = config
        self.engine = Engine(self._step)
        self.model = model
        self.tb_writer = tb_writer
        self.trainer = None

        # Using custom metric wrapper which retrieves metrics from dictionary instead of separately calculating them.
        self.metrics = {k: LossFromDict(k) for k in self.model.metric_names}
        self.non_scalar_metrics = {k: LossFromDict(k, reduce=False) for k in self.model.non_scalar_metrics_names}

        if 'external_metrics' in config['val_data']:
            for idx, name in enumerate(config['val_data']['external_metrics']):
                if 'external_metrics_kw_args' in config['val_data']:
                    self.metrics[name] = get_subclass(name, Metric)(config['devices'][0],
                                                                    **config['val_data']['external_metrics_kw_args'][idx])
                else:
                    self.metrics[name] = get_subclass(name, Metric)()

        self._handle_save_best_checkpoint_handler = \
            ModelCheckpoint(checkpoint_dir, 'best',
                            score_function=lambda engine: -self.model.main_metric(engine.state.metrics),
                            score_name=self.model.name_main_metric,
                            n_saved=1,
                            require_empty=False)

        self.add_handler()
        self.best_loss = None
        self.current_data_loader = None
        self.main_data_loader = config['val_data']['main_dataset']

    def run(self):
        """
        Start the evaluation run which will run through one epoch for each validation dataset
        :return:
        """
        for name, data_loader in self.data_loaders.items():
            self.current_data_loader = name
            self.engine.run(data_loader)

    def set_trainer(self, trainer):
        """
        Setter method for setting the trainer object which is mainly needed for getting information on the current
        training iteration.
        :param trainer: Trainer object
        :return:
        """
        self.trainer = trainer

    def add_handler(self):
        """
        Adds all the callback handlers to the trainer engine. Should be called in the end of the init.
        :return:
        """
        for name, metric in self.metrics.items():
            metric.attach(self.engine, name)

        for name, non_scalar_metric in self.non_scalar_metrics.items():
            non_scalar_metric.attach(self.engine, name)

        # on epoch complete
        self.engine.add_event_handler(Events.EPOCH_COMPLETED,
                                      self._handle_save_best_checkpoint_handler, self.model.networks)
        self.engine.add_event_handler(Events.EPOCH_COMPLETED,
                                      self._handle_log_validation_results)

        # on iteration complete
        self.engine.add_event_handler(Events.ITERATION_COMPLETED,
                                      self._handle_log_val_images)

    def _step(self, engine, batch):
        """
        Definition of a single evaluation step. This function gets automatically called by the engine every iteration.
        :param engine: evaluator engine
        :param batch: one batch provided by the data loader
        :return:
        """
        self.model.eval()
        self.model.set_input(batch)
        self.model.test()
        return self.model.state

    def _handle_log_validation_results(self, engine):
        """
        Handler for writing the losses to tensorboard and sacred.
        :param engine: evaluation engine
        :return:
        """
        metrics = self.engine.state.metrics

        loss = self.model.main_metric(metrics)
        metrics[self.model.name_main_metric] = loss

        for name, m in metrics.items():
            if 'non_scalar_metric_' not in name:  # Only add scalars
                # log to sacred
                self.run_info.log_scalar(f"val_{self.current_data_loader}.{name}.", m, self.trainer.engine.state.iteration)
                self.tb_writer.add_scalar(f"val_{self.current_data_loader}/{name}.", m, self.trainer.engine.state.iteration)

        self.logger.info(
            "Validation Results for {} - Epoch: {}  Avg loss: {:.6f}".format(self.current_data_loader,
                                                                             self.trainer.engine.state.epoch, loss))
        if self.current_data_loader == self.main_data_loader and \
                (self.best_loss is None or loss < self.best_loss):
            self.best_loss = loss
        self.run_info.result = self.best_loss

        self._handle_complete_val_dataset_figure(engine)

    def _handle_log_val_images(self, engine):
        """
        Handler for writing visual samples to tensorboard.
        :param engine: evaluation engine
        :return:
        """
        if engine.state.iteration == 1:
            for name, visual in self.model.visuals.items():
                self.tb_writer.add_image(f"val_{self.current_data_loader}/{name}.",
                                         visual.transpose(2, 0, 1),
                                         self.trainer.engine.state.iteration)

    def _score_function(self, engine):
        """
        Helper method use in ModelCheckpoint to save the best model. Need to change the sign because it saves the
        ModelCheckpoint saves the best scores.
        :param engine: evaluation engine
        :return:
        """
        val_loss = engine.state.metrics[self.model.name_main_metric]
        return -val_loss

    def _handle_complete_val_dataset_figure(self, engine):
        """
        Adds complete validation dataset metric figure to tensorboard.
        :param engine: evaluation engine
        :return:
        """
        figures = self.model.get_validation_figures(engine.state)
        for name, figure in figures.items():
            self.tb_writer.add_figure(f"val_{self.current_data_loader}_metrics/{name}",
                                      figure,
                                      self.trainer.engine.state.iteration)


def train(model_name: str, run, logger):
    """
    Main function for starting the training. Sets up the dataloader, trainer and evaluater objects
    and starts the training.
    :param model_name: name of the model definition that is defined in the hibashi/models directory
    :param run: sacred run object containing for example the configuration of the experiment
    :param logger: python logger object
    :return: final validation loss
    """
    config = run.config
    print(f'Printing out configuration:')
    pprint.pprint(config)

    run_meta_dir = get_meta_dir(run)
    checkpoint_dir = os.path.join(run_meta_dir, 'checkpoints')
    tb_logger_dir = run_meta_dir

    run.info.update({"tensorflow": {"logdirs": [tb_logger_dir]}})
    writer = SummaryWriter(tb_logger_dir, filename_suffix='')

    importlib.import_module(f'hibashi.models.{model_name}.data.datasets', __package__)
    importlib.import_module(f'hibashi.models.{model_name}.data.augmentations', __package__)
    importlib.import_module(f'hibashi.models.{model_name}.losses', __package__)

    train_dataset = get_subclass(config['train_data']['name'], Dataset)(**config['train_data'])

    if 'sampler_n_per_ds' in config['train_data']:
        train_sampler = get_subclass(config['train_data']['sampler'], Sampler)(
            data_source=train_dataset, n_per_ds=config['train_data']['sampler_n_per_ds'])
    else:
        train_sampler = get_subclass(config['train_data']['sampler'], Sampler)(
            data_source=train_dataset)

    if 'collate_fn' in config['train_data']:
        collate_fn = getattr(collate, config['train_data']['collate_fn'])
    else:
        collate_fn = default_collate

    train_loader = DataLoader(train_dataset,
                              batch_size=config['train_data']['batch_size'],
                              num_workers=config['train_data']['n_workers'],
                              drop_last=config['train_data']['drop_last'],
                              sampler=train_sampler,
                              collate_fn=collate_fn,
                              pin_memory=True)

    val_loaders = {}
    for name in config['val_data']['names']:
        # TODO I would like to make the following a bit nicer
        if name == 'Collection':
            keyword_args = config['val_data'][name]['ds_kwargs']
            val_dataset = get_subclass(name, Dataset)(**config['val_data'][name], ds_kw_args=keyword_args)
        else:
            val_dataset = get_subclass(name, Dataset)(**config['val_data'][name])

        logger.info('Validation dataset {}: {}'.format(name, len(val_dataset)))

        if 'n_val_samples' in config['val_data'][name]:
            val_sampler = get_subclass(config['val_data'][name]['sampler'], Sampler)(
                data_source=val_dataset,
                num_samples=config['val_data'][name]['n_val_samples'], replacement=True)
        else:
            val_sampler = get_subclass(config['val_data'][name]['sampler'], Sampler)(data_source=val_dataset)

        if 'collate_fn' in config['val_data'][name]:
            collate_fn = getattr(collate, config['val_data'][name]['collate_fn'])
        else:
            collate_fn = default_collate

        val_loaders[name] = DataLoader(val_dataset,
                                       batch_size=config['val_data'][name]['batch_size'],
                                       sampler=val_sampler,
                                       num_workers=config['val_data'][name]['n_workers'],
                                       drop_last=config['val_data'][name]['drop_last'],
                                       collate_fn=collate_fn,
                                       pin_memory=True)

    logger.info('Train data: {}'.format(len(train_dataset)))

    # Build the model
    model = get_subclass(model_name, Model)(config['devices'], is_train=True, **config['model'])
    model.print_networks(verbose=True)

    # Trainer / Evaluator
    evaluator = Evaluator(model, config, val_loaders, writer, run, logger, checkpoint_dir)

    trainer = Trainer(model, config, evaluator, train_loader, writer, run, logger, checkpoint_dir)

    evaluator.set_trainer(trainer)

    trainer.run()
    writer.close()

    return evaluator.best_loss
