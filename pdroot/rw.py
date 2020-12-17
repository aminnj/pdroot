import numpy as np
import pandas as pd
import uproot # uproot3
import awkward # awkward0
from tqdm.auto import tqdm


def read_root(filename, treename="t", columns=None, progress=False, **kwargs):
    """
    Read ROOT file containing one TTree into pandas DataFrame.
    Thin wrapper around `uproot.iterate`.

    filename: filename(s)/file pattern(s)
    treename: name of input TTree
    progress: show tqdm progress bar?
    columns: list of columns ("branches") to read (default of `None` reads all)
    **kwargs: extra kwargs to pass to `uproot.iterate`

    Passing `entrysteps=[(0, 10)]` will read the first 10 rows, for example.
    """
    # entrysteps of None iterates by basket to match `dataframe_to_ttree`
    iterable = uproot.iterate(
        filename,
        treename,
        columns,
        entrysteps=kwargs.pop("entrysteps", None),
        outputtype=dict,
        namedecode="ascii",
        **kwargs,
    )
    if progress:
        iterable = tqdm(iterable)
    f = uproot.open(filename)
    categorical_columns = [k.decode().split("_",1)[1].rsplit(";",1)[0] for k in f.keys() if k.startswith(b"categories_")]
    def to_df(chunk):
        for column in list(chunk.keys()):
            vals = chunk[column]
            if (vals.dtype == "object") and (column+"_strn" in chunk.keys()):
                del chunk[column+"_strn"]
                chunk[column] = awkward.StringArray.fromjagged(vals.astype("uint8"))
            elif column in categorical_columns:
                sep = "<!SEP!>"
                categories = np.array(f[f"categories_{column}"].decode().split(sep))
                chunk[column] = categories[vals]
        return pd.DataFrame(chunk)
    df = pd.concat(map(to_df, iterable), ignore_index=True, sort=True)
    return df


def to_root(
    df, filename, treename="t", chunksize=1e6, compression=uproot.LZ4(1), progress=False
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
    string_branches = []
    category_branches = []
    for bname, dtype in df.dtypes.items():
        if (dtype == "object"):
            if type(df.iloc[0][bname]) in [str]:
                tree_dtypes[bname] = uproot.newbranch(np.dtype(">i2"), size=bname+"_strn", compression=uproot.ZLIB(6))
                string_branches.append(bname)
            else:
                raise Exception(f"Don't know what kind of branch {bname} is.")
        elif (str(dtype) == "category"):
            tree_dtypes[bname] = np.int8
            category_branches.append(bname)
        else:
            tree_dtypes[bname] = dtype
    with uproot.recreate(filename, compression=compression) as f:
        t = uproot.newtree(tree_dtypes)
        f[treename] = t
        for bname in category_branches:
            sep = "<!SEP!>"
            f[f"categories_{bname}"] = sep.join(df[bname].cat.categories)
        chunksize = int(chunksize)
        iterable = range(0, len(df), chunksize)
        if progress:
            iterable = tqdm(iterable)
        for i in iterable:
            chunk = df.iloc[i : i + chunksize]
            basket = dict()
            for column in chunk.columns:
                if column in string_branches:
                    arr = chunk[column].values.astype("str")
                    jagged = awkward.StringArray.fromnumpy(arr)._content
                    jagged = jagged[jagged != 0]
                    basket[column] = jagged
                    basket[column+"_strn"] = jagged.counts
                elif column in category_branches:
                    basket[column] = chunk[column].cat.codes.values.astype(np.int8)
                else:
                    basket[column] = chunk[column].values
            f[treename].extend(basket)

