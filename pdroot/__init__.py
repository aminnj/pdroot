import pandas
from pandas.core.base import PandasObject


from .draw import tree_draw, tree_adraw, iter_draw

PandasObject.draw = tree_draw
PandasObject.adraw = tree_adraw

from .readwrite import read_root, to_root, iter_chunks, ChunkDataFrame, to_pandas

setattr(pandas, "read_root", read_root)
PandasObject.to_root = to_root

from .accessors import AwkwardArrayAccessor, LorentzVectorAccessor
