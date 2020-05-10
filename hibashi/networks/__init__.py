import importlib
import os
import pkgutil

path = os.path.split(__file__)[0]
for (module_loader, name, ispkg) in pkgutil.iter_modules([path]):
    importlib.import_module('hibashi.networks.' + name, __package__)

path = os.path.split(path)[0]
path = os.path.join(path, 'models')

for (module_loader, name, ispkg) in pkgutil .iter_modules([path]):
    importlib.import_module('hibashi.models.{}.networks'.format(name), __package__)