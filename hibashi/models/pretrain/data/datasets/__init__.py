import importlib
import os
import pkgutil


path = os.path.split(__file__)[0]
for (module_loader, name, is_pkg) in pkgutil.iter_modules([path]):
    importlib.import_module(__name__ + '.' + name, __package__)