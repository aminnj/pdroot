import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import uproot4
import awkward1

warnings.simplefilter("ignore", category=FutureWarning)
import uproot3
warnings.resetwarnings()

warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import fletcher

def array_to_fletcher_or_numpy(array):
    if array.ndim >= 2:
        return fletcher.FletcherContinuousArray(awkward1.to_arrow(array))
    else:
        a = array.layout
        if hasattr(a, "content"):
            a = a.content
        return np.array(a, copy=False)

def awkward1_arrays_to_dataframe(arrays):
    df = pd.DataFrame({name:array_to_fletcher_or_numpy(arrays[name]) for name in awkward1.fields(arrays)}, copy=False)
    return df

def read_root(
    filename, treename=None, columns=None, entry_start=None,  entry_stop=None,
):
    """
    Read ROOT file containing one TTree into pandas DataFrame.
    See documentation for `uproot4.open` and `uproot4.arrays`.

    filename: filename/file pattern
    treename: name of input TTree. If `None`, defaults to the only tree in a file, otherwise prefers `Events`.
    columns: list of columns ("branches") to read (default of `None` reads all)
    entry_start: start entry index (default of `None` means start of file)
    entry_stop: stop entry index (default of `None` means end of file)
    """
    f = uproot4.open(filename)
    if treename is None:
        treenames = [n.rsplit(";",1)[0] for n in f.keys()]
        if len(treenames) == 1:
            treename = treenames[0]
        elif "Events" in treenames:
            treename = "Events"
        else:
            raise RuntimeError("`treename` must be specified. File contains keys: {treenames}")

    t = f[treename]
    if columns is None: columns = lambda x: not x.endswith("_varn")
    arrays = t.arrays(
            filter_name=columns,
            entry_start=entry_start,
            entry_stop=entry_stop,
        )
    df = awkward1_arrays_to_dataframe(arrays)
    return df

def to_root(
    df, filename, treename="t", chunksize=1e6, compression=uproot3.LZ4(1), progress=False
):
    """
    Writes ROOT file containing one TTree with the input pandas DataFrame.

    filename: name of output file
    treename: name of output TTree
    chunksize: number of rows per basket
    compression: uproot compression object (LZ4, ZLIB, LZMA, or None)
    progress: show tqdm progress bar?
    """
    tree_dtypes = dict()
    jagged_branches = []
    for bname, dtype in df.dtypes.items():
        if "fletcher" in str(dtype):
            dtype = np.dtype(dtype.arrow_dtype.value_type.to_pandas_dtype())
            tree_dtypes[bname] = uproot3.newbranch(dtype, size=bname + "_varn", compression=None)
            jagged_branches.append(bname)
        elif "object" in str(dtype):
            raise RuntimeError(f"Don't know how to serialize column {bname} with object dtype.")
        else:
            dtype = str(dtype).lstrip("u")
            tree_dtypes[bname] = dtype
    with uproot3.recreate(filename, compression=compression) as f:
        t = uproot3.newtree(tree_dtypes)
        f[treename] = t
        chunksize = int(chunksize)
        iterable = range(0, len(df), chunksize)
        if progress:
            iterable = tqdm(iterable)
        for i in iterable:
            chunk = df.iloc[i : i + chunksize]
            basket = dict()
            for column in chunk.columns:
                if column in jagged_branches:
                    arr = chunk[column].ak(version=0)
                    basket[column] = arr
                    basket[column + "_varn"] = arr.counts.astype("int32")
                else:
                    basket[column] = chunk[column].values
            f[treename].extend(basket)

class ChunkDataFrame(pd.DataFrame):
    tree = None
    treename = "Events"
    filename = None
    entry_start = None
    entry_stop = None

    def __init__(self, *args, **kwargs):
        self.filename = kwargs.pop("filename", None)
        self.treename = kwargs.pop("treename", "Events")
        self.entry_start = kwargs.pop("entry_start", None)
        self.entry_stop = kwargs.pop("entry_stop", None)
        super(ChunkDataFrame, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return ChunkDataFrame

    def _load_tree(self):
        if self.tree is None:
            self.tree = uproot4.open(self.filename)[self.treename]

    def _add_column(self, column):
        self._load_tree()
        array = self.tree[column].array(entry_start=self.entry_start, entry_stop=self.entry_stop)
        array = array_to_fletcher_or_numpy(array)
        self[column] = array

    def _possibly_cache(self, key):
        is_str = isinstance(key, (str))
        if is_str:
            key = [key]
        else:
            is_list = isinstance(key, (tuple, list))
            if not is_list: return

        for column in key:
            if column in self.columns.values:
                continue
            self._add_column(column)

    def __getitem__(self, key):
        self._possibly_cache(key)
        return super().__getitem__(key)
