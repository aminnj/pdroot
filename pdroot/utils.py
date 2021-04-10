import sys
import importlib.util
import warnings

class LazyImport:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None
    def __getattr__(self, name):
        if self.module is None:
            warnings.simplefilter("ignore", category=FutureWarning)
            self.module = __import__(self.module_name)
            warnings.resetwarnings()
        return getattr(self.module, name)

awkward0 = LazyImport("awkward0")
awkward1 = LazyImport("awkward1")
uproot3 = LazyImport("uproot3")
uproot4 = LazyImport("uproot4")
uproot3_methods = LazyImport("uproot3_methods")
