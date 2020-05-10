from experiment import ex
from hibashi.framework.training import train


@ex.automain
def run(_log, _run):
    _log.info('Starting the training process.')
    return train(ex.path, _run, _log)
