import importlib
import os
import random
import shutil
import zipfile
import imgaug
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import MongoObserver, SlackObserver
from sacred.settings import SETTINGS

import hibashi.framework.factory as factory
from hibashi.commands import add_commands
from hibashi.framework.utils import PrefixLineFilter, get_username, get_meta_dir, OverwritingFileStorageObserver

SETTINGS.DISCOVER_SOURCES = 'dir'

ex = Experiment('Embers')
ex.captured_out_filter = PrefixLineFilter('*')
add_commands(ex)


# In the following function the model configuration is setup. Model init parameters and factory objects
# are added to the model config parameter by joining them into one dictionary. There are dynamically received based
# on the selected model_name.
@ex.config
def cfg(_log):
    _log.info('cfg triggered')
    devices = [0]  # Device ids to use. -1 for cpu. 0-n for GPU ids. Multiple like [0, 1]
    user = get_username()  # add user for omniboard

    # # Model parameters. Should be set first.
    model = {**factory.get_model_factory_class_names(ex.path),
             **factory.get_model_factory_params(ex.path)}
    print(model)

    # adds a parameter group for every factory object of the model
    params = factory.get_model_factory_params(ex.path)

    if len(params) > 0:
        ex.add_config(**factory.get_model_factory_params(ex.path))


@ex.pre_run_hook
def factory_hook(_run, _log):
    _log.info('triggered')
    factory.update_params(ex.path, _run.config)


@ex.pre_run_hook
def device_hook(devices, _log, _run):
    devices_string = ""
    if devices != -1:
        if not isinstance(devices, list):
            devices = [devices]
        devices_string = ",".join(str(d) for d in devices)

        # os.environ['CUDA_VISIBLE_DEVICES'] = devices_string
        # _log.info('Setting CUDA_VISIBLE_DEVICES to: %s' % devices_string)
    _run.config['devices'] = devices


@ex.pre_run_hook
def seed_hook(seed, _log):
    _log.info('Setting seed to: %s' % seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    imgaug.seed(seed)


@ex.pre_run_hook
def set_username_hook(_log, _run):
    user = os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))
    _run.info.update({"user": user})
    _log.info('Setting user to: %s' % user)


@ex.pre_run_hook
def dir_hook(_log, _run):
    _run.info['run_meta_dir'] = get_meta_dir(_run)

    if _run.meta_info['options']['--unobserved'] and os.path.isdir(_run.info['run_meta_dir']):
        shutil.rmtree(_run.info['run_meta_dir'], ignore_errors=True)

    if not os.path.isdir(_run.info['run_meta_dir']):
        os.makedirs(_run.info['run_meta_dir'])


@ex.pre_run_hook
def zip_module_hook(_log, _run):
    from modulefinder import ModuleFinder

    mod = importlib.import_module('hibashi.models.{}.{}'.format(ex.path, ex.path))

    finder = ModuleFinder(path=[_run.experiment_info['base_dir']])
    finder.run_script(mod.__file__)
    files = [m.__file__ for k, m in finder.modules.items() if
             isinstance(m.__file__, str) and '/hibashi' in m.__file__]

    exp_path = os.path.join(_run.info['run_meta_dir'], 'model.zip')
    zip_file = zipfile.ZipFile(exp_path, "w")

    for file in files:
        file_rel = os.path.relpath(file, _run.experiment_info['base_dir'])
        zip_file.write(file, file_rel, compress_type=zipfile.ZIP_DEFLATED)
    zip_file.close()

    _log.info('Model successfully exported as model.zip')


@ex.config_hook
def meta_hook(config, command_name, logger):
    experiment_meta_dir = os.path.join(config['train']['metadata_path'], ex.path)

    if command_name == 'run':

        if 'connect_mongo_db' not in config['train'] or config['train']['connect_mongo_db']:
            ex.observers.append(MongoObserver.create(url='mongodb://192.168.77.15:27017', db_name='sacred'))

        ex.observers.append(OverwritingFileStorageObserver.create(experiment_meta_dir))
        logger.info('Storing metadata at: %s' % experiment_meta_dir)

        if 'connect_slack' not in config['train'] or config['train']['connect_slack']:
            complete_template = ":white_check_mark: *{experiment[name]}: {_id}* from *{config[user]}* on *{host_info[" \
                                "hostname]}* completed after _{elapsed_time}_ with result=`{result}` "
            interrupted_template = ":warning: *{experiment[name]}: {_id}* from *{config[user]}* on *{host_info[" \
                                   "hostname]}* interrupted after _{elapsed_time}_ "
            error_template = ":x: *{experiment[name]}: {_id}* from *{config[user]}* on *{host_info[hostname]}* failed " \
                             "after _{elapsed_time}_ with `{error}` "
            slack_observer = SlackObserver('https://hooks.slack.com/services/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',
                                           icon=':imp:')
            slack_observer.completed_text = complete_template
            slack_observer.interrupted_text = interrupted_template
            slack_observer.failed_text = error_template
            ex.observers.append(slack_observer)



