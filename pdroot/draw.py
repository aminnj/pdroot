import time
import numpy as np
import pandas as pd

import warnings

warnings.simplefilter("ignore", category=FutureWarning)
import uproot3
import uproot4

import awkward1

from tqdm.auto import tqdm

warnings.resetwarnings()

from yahist import Hist1D, Hist2D

from .readwrite import awkward1_arrays_to_dataframe, iter_chunks
from .parse import variables_in_expr, to_ak_expr, split_expr_on_free_colon


def _array_ndim(array):
    if hasattr(array, "ndim"):
        return array.ndim
    return np.ndim(array)

def _has_mask(vals):
    if isinstance(vals, np.ma.masked_array) or (
        hasattr(vals, "layout")
        and isinstance(vals.layout, awkward1.layout.ByteMaskedArray)
    ):
        return True
    return False

def _tree_draw_to_array(df, varexp, sel="", weights="", env=dict()):

    varexp_exprs = [to_ak_expr(expr) for expr in split_expr_on_free_colon(varexp)]
    weights_expr = to_ak_expr(weights)
    sel_expr = to_ak_expr(sel)

    colnames = variables_in_expr(f"{varexp}${sel}${weights}")
    loc = {"ak": awkward1, "np": np, "pd": pd}
    for colname in colnames:
        loc[colname] = df[colname].ak(1)
    loc.update(env)

    if sel:
        globalmask = eval(sel_expr, dict(), loc)

    vweights = None

    def expr_to_vals(expr):
        vals = eval(expr, dict(), loc)

        # if varexp is a simple constant, broadcast it to an array
        if _array_ndim(vals) == 0:
            vals = vals * np.ones(len(df))

        if sel:
            vals = vals[globalmask]

        if _array_ndim(vals) > 1:
            vals = awkward1.flatten(vals)

        vals = awkward1.to_numpy(vals)
        return vals

    dims = []
    for expr in varexp_exprs:
        vals = expr_to_vals(expr)
        dims.append(vals)

    if weights:
        vweights = expr_to_vals(weights_expr)

    mask = None

    if len(dims) == 1:
        x = dims[0]
        if (_has_mask(x) and np.ndim(x.mask) != 0):
            mask = x.mask
            x = x.data[~mask]
        elif _has_mask(x):
            x = x.data
        vals = x

    # if one of the dimensions has a mask, we want to apply the OR
    # of the two to make sure their lengths will be equal
    if len(dims) == 2:
        x, y = dims
        if (_has_mask(x) and np.ndim(x.mask) != 0) or (
            _has_mask(y) and np.ndim(y.mask) != 0
        ):
            mask = x.mask | y.mask
            x = x.data[~mask]
            y = y.data[~mask]
        elif _has_mask(x):
            x = x.data
        elif _has_mask(y):
            y = y.data
        vals = np.c_[x, y]

    if weights:
        if mask is not None:
            vweights = vweights[~mask]

    return vals, vweights


def tree_draw(df, varexp, sel="", weights="", to_array=False, env=dict(), **kwargs):
    """
    Draws a 1D or 2D histogram (or an array) from a pandas DataFrame.

    varexp: expression to draw
    sel: selection expression
    weights: weight expression
    to_array: return an array if True, otherwise return a `yahist.Hist1D` or `yahist.Hist2D`
    env: dictionary of additional symbols needed to parse the expressions

    >>> df.draw("Jet_pt", "MET_pt>40")
    >>> df.draw("Jet_pt", "MET_pt>40", to_array=True)
    >>> df.draw("Jet_pt[0]:Jet_pt[1]")
    >>> df.draw("1", "length(Jet_pt)>2")
    >>> df.draw("1", "length(Jet_pt[abs(Jet_eta)<2.4])>2")
    >>> df.draw("sum(-2.4<Jet_eta<2.4 and Jet_pt>25)")
    """
    array, vweights = _tree_draw_to_array(df, varexp, sel, weights, env)
    if to_array:
        if weights:
            return array, vweights
        return array

    if weights:
        kwargs["weights"] = vweights
    if np.ndim(array) == 1:
        return Hist1D(array, **kwargs)
    elif np.ndim(array) == 2:
        return Hist2D(array, **kwargs)


def iter_draw(
    path,
    varexp,
    sel="",
    treename="t",
    bins=None,
    progress=True,
    step_size="50MB",
    **kwargs,
):
    """
    Loop over specified ROOT files in `path` in chunks, making histograms and returning their sum.
    Tree name is specified via `treename`.
    Iterates over the files in chunks of `step_size` (as per `uproot4.iterate`), reading
    only the branches deemed necessary according to `pdroot.parse.variables_in_expr`.
    """
    columns = variables_in_expr(f"{varexp}${sel}")

    opts = dict()
    if bins:
        opts["bins"] = bins
    opts.update(kwargs)

    hists = []
    for df in iter_chunks(path, treename=treename, progress=progress, step_size=step_size, columns=columns):
        h = df.draw(varexp, sel, **opts)
        if "bins" not in opts:
            opts["bins"] = h.edges
        hists.append(h)
    h = sum(hists)
    return h
