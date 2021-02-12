import time
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import uproot4
import awkward1

warnings.simplefilter("ignore", category=FutureWarning)
import uproot3
import awkward0

warnings.resetwarnings()

warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import fletcher


def array_to_fletcher_or_numpy(array):
    arrow_array = awkward1.to_arrow(array)
    fletcher_array = fletcher.FletcherContinuousArray(awkward1.to_arrow(array))
    if (array.ndim >= 2) or (fletcher_array.data.null_count > 0):
        return fletcher_array
    if "list<" not in str(fletcher_array.dtype):
        a = array.layout
        if hasattr(a, "content"):
            a = a.content
        return np.array(a, copy=False)
    return fletcher_array


def awkward1_arrays_to_dataframe(arrays):
    fields = awkward1.fields(arrays)
    fields = filter(lambda x: not x.endswith("_varn"), fields)
    df = pd.DataFrame(
        {name: array_to_fletcher_or_numpy(arrays[name]) for name in fields}, copy=False,
    )
    return df


def maybe_unmask_jagged_array(array):
    """
    Going through Fletcher/parquet can make JaggedArrays
    end up as BitMaskedArray (even though there are no NaN),
    so if the mask is dummy, return a regular JaggedArray
    """
    if not "BitMaskedArray" in str(type(array)):
        return array

    mask = array.mask
    keep_all = np.all(np.unpackbits(mask, count=len(array), bitorder="little"))
    if not keep_all:
        return array

    content = array.content.content.content
    offsets = array.content.offsets
    return awkward0.JaggedArray.fromoffsets(offsets, content)


def find_tree_name(f):
    treename = None
    treenames = [n.rsplit(";", 1)[0] for n in f.keys()]
    if len(treenames) == 1:
        treename = treenames[0]
    elif "Events" in treenames:
        treename = "Events"
    return treename


def read_root(
    filename,
    treename=None,
    columns=None,
    entry_start=None,
    entry_stop=None,
    nthreads=4,
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
        treename = find_tree_name(f)
    if treename is None:
        raise RuntimeError(
            f"`treename` must be specified. File contains keys: {f.keys()}"
        )

    executor = None
    if nthreads > 1:
        import concurrent.futures

        executor = concurrent.futures.ThreadPoolExecutor(nthreads)

    t = f[treename]
    arrays = t.arrays(
        filter_name=columns,
        entry_start=entry_start,
        entry_stop=entry_stop,
        decompression_executor=executor,
    )
    df = awkward1_arrays_to_dataframe(arrays)
    df.columns
    return df


def to_root(
    df,
    filename,
    treename="t",
    chunksize=20e3,
    compression=uproot3.ZLIB(1),
    compression_jagged=uproot3.ZLIB(1),
    progress=False,
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
            tree_dtypes[bname] = uproot3.newbranch(
                dtype, size=bname + "_varn", compression=compression_jagged
            )
            jagged_branches.append(bname)
        elif "object" in str(dtype):
            raise RuntimeError(
                f"Don't know how to serialize column {bname} with object dtype."
            )
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
                    arr = maybe_unmask_jagged_array(arr)
                    # profiling says 30% of the time is spent checking if jagged __getitem__ is given a string
                    # this is not needed for writing out TTree branches, so free speedup.
                    arr._util_isstringslice = lambda x: False
                    basket[column] = arr
                    basket[column + "_varn"] = arr.counts.astype("int32")
                else:
                    basket[column] = chunk[column].values
            f[treename].extend(basket)

def iter_chunks(
    path,
    treename="t",
    progress=True,
    step_size="50MB",
    columns=None,
):
    """
    Loop over specified ROOT files in `path` in chunks, returning dataframes.
    Tree name is specified via `treename`.
    Iterates over the files in chunks of `step_size` (as per `uproot4.iterate`), reading

    columns: list of columns ("branches") to read (default of `None` reads all)
    """
    if ":" not in path:
        path = f"{path}:{treename}"

    iterable = uproot4.iterate(path, filter_name=columns, step_size=step_size)

    if progress:
        iterable = tqdm(iterable)

    nevents = 0
    t0 = time.time()
    for arrays in iterable:
        df = awkward1_arrays_to_dataframe(arrays)
        nevents += len(df)
        yield df
    t1 = time.time()
    if progress:
        print(f"Processed {nevents} in {t1-t0:.2f}s ({1e-6*nevents/(t1-t0):.2f}MHz)")

class ChunkDataFrame(pd.DataFrame):
    filename = None
    treename = None
    entry_start = None
    entry_stop = None
    tree = None

    _metadata = ["filename", "treename", "entry_start", "entry_stop", "tree"]

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
        array = self.tree[column].array(
            entry_start=self.entry_start, entry_stop=self.entry_stop
        )
        array = array_to_fletcher_or_numpy(array)
        self[column] = array

    def _possibly_cache(self, key):
        is_str = isinstance(key, (str))
        if is_str:
            key = [key]
        else:
            is_list = isinstance(key, (tuple, list))
            if not is_list:
                return

        for column in key:
            if column in self.columns.values:
                continue
            self._add_column(column)

    def __getitem__(self, key):
        self._possibly_cache(key)
        return super().__getitem__(key)

    # # Messes up notebook repr for some reason
    # def __getattr__(self, name):
    #     self._possibly_cache(name)
    #     return super().__getattr__(name)
