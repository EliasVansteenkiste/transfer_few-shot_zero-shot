"""
This module contains functional tests for experiments.py
"""
import importlib

from unittest import TestCase


class TestExperiment(TestCase):
    """
    This class contains all functional tests for the experiment building functionality in experiment.py
    """

    def test_option_consistency_age(self):
        self._test_option_consistency('age')

    def test_option_consistency_attributes(self):
        self._test_option_consistency('attributes')

    def test_option_consistency_lrgan(self):
        self._test_option_consistency('lrgan')

    def test_option_consistency_prnet(self):
        self._test_option_consistency('prnet')

    def test_option_consistency_prnetattributes(self):
        self._test_option_consistency('prnetattributes')

    @staticmethod
    def _test_option_consistency(experiment_name):
        """
        We test if the options that are actually assigned to the run object are the same as in the options file
        :return: None
        """
        print(f'Testing the options for experiment{experiment_name}')

        # mock options
        options = {'--print_config': False,
                   '--queue': False,
                   '--enforce_clean': False,
                   '--tiny_db': None,
                   '--sql': None,
                   '--model': experiment_name,
                   '--pdb': False,
                   '--cudnn_benchmark': False,
                   '--beat_interval': None,
                   '--comment': 'test',
                   '--unobserved': True,
                   '--file_storage': None,
                   '--force': False,
                   '--loglevel': None,
                   '--mongo_db': None,
                   '--name': None,
                   '--capture': None,
                   '--debug': False,
                   '--priority': None,
                   '--help': False,
                   'with': True,
                   'UPDATE': ['devices=[0]', 'seed=37145'],
                   'help': False,
                   'COMMAND': None}

        import experiment
        importlib.reload(experiment)
        run = experiment.ex.run(command_name='print_config', options=options)
        opts_loc = 'hibashi.models.' + experiment_name + '.options'
        opts = importlib.import_module(opts_loc, __package__)
        for key in run.config.keys():
            method_name = key + '_cfg'
            if hasattr(opts, method_name):
                print(f'{method_name} found in {opts_loc}')
                cfg = getattr(opts, method_name)
                for k, v in cfg().items():
                    print(key, k)
                    print(v)
                    print(run.config[key][k])
                    assert v == run.config[key][k]
            else:
                print(f'{method_name} not found in {opts_loc}, skipping it.')

    @staticmethod
    def test_with_consistency():
        """
        We test if the options that are actually assigned to the run object are the same as passed via the with argument
        :return: None
        """
        experiment_name = 'prnet'

        print(f'Testing the options for experiment{experiment_name}')

        # mock options and config updates
        # Important here is that the UPDATE string in the options should correspond
        # to the config_updates
        config_updates = {'devices': [2],
                          'train_data': {'batch_size': 32, 'n_sub_samples': 1000},
                          'seed': 37145}

        options = {'--print_config': False,
                   '--queue': False,
                   '--enforce_clean': False,
                   '--tiny_db': None,
                   '--sql': None,
                   '--model': experiment_name,
                   '--pdb': False,
                   '--cudnn_benchmark': False,
                   '--beat_interval': None,
                   '--comment': 'test',
                   '--unobserved': True,
                   '--file_storage': None,
                   '--force': False,
                   '--loglevel': None,
                   '--mongo_db': None,
                   '--name': None,
                   '--capture': None,
                   '--debug': False,
                   '--priority': None,
                   '--help': False,
                   'with': True,
                   'UPDATE': ['devices=[2]',
                              'train_data.batch_size=32',
                              'train_data.n_sub_samples=1000',
                              'seed=37145'],
                   'help': False,
                   'COMMAND': None}

        import experiment
        importlib.reload(experiment)
        run = experiment.ex.run(command_name='print_config',
                                config_updates=config_updates,
                                options=options)

        ignore = ['devices']
        for opt_str in options['UPDATE']:
            loc, value = opt_str.split('=')
            modules = loc.split('.')
            assert modules[0] in run.config
            child = run.config[modules[0]]
            for i in range(1, len(modules)):
                assert modules[i] in child
                child = child[modules[i]]

            if modules[0] not in ignore:
                assert str(child) == value





