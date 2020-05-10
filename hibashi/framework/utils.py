import inspect
import os
import shutil

from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds


def str_match(str1: str, str2: str):
    """
    Convenience function for comparing strings ignoring case.
    :param str1:
    :param str2:
    :return: if strings are identical
    """
    return str1.lower() == str2.lower()


def all_subclasses(cls: type) -> list:
    """
    Returns a list of all the subclasses of a givven class.
    :param cls: type object
    :return: list of types
    """
    return list(set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]))


def get_parameter_of_cls(cls: type, ancestors=True) -> dict:
    """
    Returns a dictionary containing all the keyword arguments of the init of a given class and optionally also
    of all it's ancestors.
    :param cls: type object
    :param ancestors:
    :return:
    """
    if ancestors:
        cls_check = inspect.getmro(cls)
    else:
        cls_check = [cls]

    params = {}
    for c in cls_check:
        argspec = inspect.getfullargspec(c.__init__)
        if argspec.defaults:
            keyword_names = argspec.args[-len(argspec.defaults):]
            for arg, default in zip(keyword_names, argspec.defaults):
                params[arg] = default

    return params


def get_args(cls: type) -> list:
    """
    Gets all the init arguments of a class which don't have a default.
    :param cls: type object
    :return: list of args
    """
    argspec = inspect.getfullargspec(cls.__init__)
    args = argspec.args
    if argspec.defaults:
        args = args[:-len(argspec.defaults)]
    if args[0] == 'self':
        _ = args.pop(0)

    return args


def list_of_classes_to_dict(loc: list) -> dict:
    """
    Turns a list of class types into a dictionary with the class name as key and the corresponding class type as value.
    :param loc: list of classes
    :return: dictionary of classes
    """

    return {cls.__name__: cls for cls in loc}


def get_subclass(cls_descriptor: str, parent_class: type) -> type:
    """
    Get a subclasses type object of a given class with a specific name.
    :param cls_descriptor: sub class name as string
    :param parent_class: type object
    :return: type object of sub class
    """
    class_name = cls_descriptor if isinstance(cls_descriptor, str) else cls_descriptor.__name__

    subclasses = all_subclasses(parent_class)
    subclasses = {cls.__name__.lower(): cls for cls in subclasses}
    cls = subclasses.get(class_name.lower())

    if cls is None:
        raise ValueError('%s is not a valid subclass of %s. Valid options are %s.' % (
            class_name, parent_class.__name__, str(subclasses.keys())))

    return cls


class OverwritingFileStorageObserver(FileStorageObserver):
    """
    Custom sacred observer which inherits the behaviour of sacreds FileStorageObserver but overwrites existing
    files if necessary.
    """

    def started_event(self, ex_info, command, host_info, start_time, config,
                      meta_info, _id):
        if _id is not None:
            exp_dir = os.path.join(self.basedir, str(_id))
            if os.path.isdir(exp_dir) and len(os.listdir(exp_dir)) > 0:
                shutil.rmtree(exp_dir, ignore_errors=True)

        super().started_event(ex_info, command, host_info, start_time, config,
                              meta_info, _id)


def get_username() -> str:
    """
    Convenience function to get the username of the executing user which should work on different operating systems.
    :return: username
    """
    return os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))


def get_meta_dir(run) -> str:
    """
    Gets the experiment metadata directory based on a sacred run object. If the run is flagged as --unobserved, metadata
    is stored in a folder <username>_debug. Otherwise the experiment id is used.
    :param run: sacred run object
    :return: path
    """
    meta_path = run.config['train']['metadata_path']
    experiment_meta_dir = os.path.join(meta_path, run.experiment_info['name'])

    if run.meta_info['options']['--unobserved']:
        run._id = run.config['user'] + '_debug'
        run_meta_dir = os.path.join(experiment_meta_dir, run._id)
    elif run.config['train']['overwrite_id_with']:
        run_meta_dir = os.path.join(experiment_meta_dir, run.config['train']['overwrite_id_with'])
        if os.path.exists(run_meta_dir):
            print(f"Warning: {run_meta_dir} already exists.")
        run._id = run.config['train']['overwrite_id_with']

    return run_meta_dir


class PrefixLineFilter:
    """
    Filters all line out with a given prefix.
    """

    def __init__(self, prefix):
        self.prefix = prefix

    def __call__(self, text):
        text = apply_backspaces_and_linefeeds(text)
        lines = text.split('\n')
        lines_filtered = [l for l in lines if not l.startswith(self.prefix)]
        return '\n'.join(lines_filtered)
