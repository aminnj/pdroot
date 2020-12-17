import pandas
from pandas.core.base import PandasObject

from .draw import tree_draw, iter_draw
PandasObject.draw = tree_draw

from .jitdraw import jitdraw
PandasObject.jitdraw = jitdraw

from .rw import read_root, to_root
setattr(pandas, "read_root", read_root)
PandasObject.to_root = to_root

