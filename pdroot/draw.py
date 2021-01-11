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

from .query import hacky_query_eval
from .parse import variables_in_expr, nops_in_expr, to_ak_expr


def tree_draw(df, varexp, sel="", **kwargs):
    """
    1d and 2d drawing function that supports jagged columns
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

        # make a concrete array if there is a mask
        if isinstance(vals, np.ma.masked_array) or (
            hasattr(vals, "layout")
            and isinstance(vals.layout, awkward1.layout.ByteMaskedArray)
        ):
            mask = vals.mask
            if np.ndim(mask) == 0:
                vals = vals.data
            else:
                vals = vals.data[~vals.mask]

        dims.append(vals)

    if len(dims) == 1:
        return Hist1D(dims[0], **kwargs)
    else:
        return Hist2D(np.c_[dims[0], dims[1]], **kwargs)
    return dims


def tree_draw_old(df, varexp, sel="", **kwargs):
    """
    Does not support jagged columns
    """
    from tokenize import tokenize, NAME, OP
    from io import BytesIO

    fast = nops_in_expr(f"{varexp}${sel}") < 5
    twodim = ":" in varexp
    if fast and not twodim:
        raw = hacky_query_eval(df, varexp, sel)
        # performance hit for float16's. copy to float32 first.
        if raw.dtype in ["float16"]:
            vals = raw.astype("float32")
        else:
            vals = raw
        if "weights" in kwargs:
            raise Exception("`weights` kwarg not supported for `fast=True` (yet?)")
        return Hist1D(vals, **kwargs)
    if not sel:
        mask = slice(None)
    else:
        mask = df.eval(sel)
    if twodim:
        assert varexp.count(":") == 1
        varexp1, varexp2 = varexp.split(":")
        vals = np.c_[df[mask].eval(varexp1), df[mask].eval(varexp2)]
        return Hist2D(vals, **kwargs)
    else:
        vals = df[mask].eval(varexp)
        return Hist1D(vals, **kwargs)


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
