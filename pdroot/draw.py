import time
import numpy as np
import pandas as pd

import warnings

warnings.simplefilter("ignore", category=FutureWarning)
import uproot3
import uproot4

import awkward1

warnings.resetwarnings()

from yahist import Hist1D, Hist2D

from .parse import variables_in_expr, nops_in_expr, to_ak_expr

def tree_draw(df, varexp, sel="", **kwargs):
    """
    1d and 2d drawing function that supports jagged columns
    returns hist
    """
    array = tree_draw_to_array(df, varexp, sel)
    if np.ndim(array) == 1:
        return Hist1D(array, **kwargs)
    elif np.ndim(array) == 2:
        return Hist2D(array, **kwargs)

def tree_draw_to_array(df, varexp, sel=""):
    """
    1d and 2d drawing function that supports jagged columns
    returns array
    """
    colnames = variables_in_expr(f"{varexp}${sel}")
    for colname in colnames:
        locals()[colname] = df[colname].ak(1)
    locals()["ak"] = awkward1

    if sel:
        globalmask = eval(to_ak_expr(sel))

    dims = []
    for ve in varexp.split(":"):

        vals = eval(to_ak_expr(ve))
        if sel:
            vals = vals[globalmask]

        if vals.ndim > 1:
            vals = awkward1.flatten(vals)

        vals = awkward1.to_numpy(vals)
        dims.append(vals)

    def has_mask(vals):
        if isinstance(vals, np.ma.masked_array) or (
            hasattr(vals, "layout")
            and isinstance(vals.layout, awkward1.layout.ByteMaskedArray)
        ):
            return True
        return False

    if len(dims) == 1:

        vals = dims[0]
        if has_mask(vals):
            mask = vals.mask
            if np.ndim(mask) != 0:
                vals = vals.data[~mask]
            else:
                vals = vals.data

    # could be simplified
    # if one of the dimensions has a mask, we want to apply the OR
    # of the two to make sure their lengths will be equal
    if len(dims) == 2:
        x, y = dims
        if (has_mask(x) and np.ndim(x.mask) != 0) or (has_mask(y) and np.ndim(y.mask) != 0):
            mask = x.mask | y.mask
            x = x.data[~mask]
            y = y.data[~mask]
        else:
            if has_mask(x):
                x = x.data
            if has_mask(y):
                y = y.data

        vals = np.c_[x, y]

    return vals


def iter_draw(
    path, varexp, sel="", treepath="t", progress=False, entrysteps="50MB", **kwargs
):
    """
    Same as `tree_draw`, except this requires an additional first argument for the path/pattern
    of input root files. Tree name is specified via `treepath`.
    Iterates over the files in chunks of `entrysteps` (as per `uproot.iterate`), reading
    only the branches deemed necessary according to `variables_in_expr`.
    """
    hists = []
    t0 = time.time()
    nevents = 0
    edges = None
    branches = variables_in_expr(varexp + ":" + sel)
    iterable = enumerate(
        uproot3.iterate(
            path,
            treepath=treepath,
            entrysteps=entrysteps,
            branches=branches,
            namedecode="ascii",
            outputtype=pd.DataFrame,
        )
    )
    if progress:
        from tqdm.auto import tqdm

        iterable = tqdm(iterable)
    for i, df in iterable:
        nevents += len(df)
        if i >= 1 and "bins" not in kwargs:
            kwargs["bins"] = edges
        h = df.draw(varexp, sel, **kwargs)
        edges = h.edges
        hists.append(h)
    h = sum(hists)
    t1 = time.time()
    if progress:
        print(f"Processed {nevents} in {t1-t0:.2f}s ({1e-6*nevents/(t1-t0):.2f}MHz)")
    return h
