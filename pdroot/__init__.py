import pandas
from pandas.core.base import PandasObject


from .draw import tree_draw, iter_draw

PandasObject.draw = tree_draw

from .readwrite import read_root, to_root, iter_chunks, ChunkDataFrame

setattr(pandas, "read_root", read_root)
PandasObject.to_root = to_root

from .accessors import AwkwardArrayAccessor, LorentzVectorAccessor
