import importlib
import os
import pkgutil

path = os.path.split(__file__)[0]
for (module_loader, name, ispkg) in pkgutil.iter_modules([path]):
    importlib.import_module('hibashi.data.datasets.' + name, __package__)



