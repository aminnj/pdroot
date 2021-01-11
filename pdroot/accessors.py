import awkward0
import awkward1
import pyarrow
import pandas as pd
import uproot3_methods
import numpy as np
import numba

from .readwrite import ChunkDataFrame


def pandas_series_to_awkward(series, version=0):
    values = series.values
    if "fletcher" not in str(values.dtype).lower():
        if version == 1:
            return awkward1.from_numpy(values)
        else:
            return np.array(values, copy=False)

    array_arrow = values.data

    # if array_arrow.offset != 0:
    #     # this means we have done a slice, and from_arrow will
    #     # not see the slice, instead returning the beginning of the
    #     # array/buffer, so we explicitly do a .take() to fix it
    #     array_arrow = array_arrow.take(np.arange(len(array_arrow)))
    #     import warnings

    #     warnings.warn(
    #         "Made a copy to handle slicing an Arrow array with awkward array. This code can be removed when https://github.com/scikit-hep/awkward-1.0/pull/625 is merged."
    #     )

    if version == 0:
        array = awkward0.fromarrow(array_arrow)
    elif version == 1:
        array = awkward1.from_arrow(array_arrow)
    else:
        raise RuntimeError(
            "What version of awkward do you want? Specify `version=0` or `1`."
        )
    return array


@pd.api.extensions.register_series_accessor("ak")
class AwkwardArrayAccessor:
    def __init__(self, obj):
        self._obj = obj

    def __call__(self, version=0):
        # version=0 returns an awkward0 array, version1 returns awkward1
        return pandas_series_to_awkward(self._obj, version=version)


@pd.api.extensions.register_dataframe_accessor("ak")
class AwkwardArraysAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self, version=0):
        df = self._obj
        if version == 0:
            return awkward0.Table(
                dict((c, df[c].ak(version=version)) for c in df.columns)
            )
        elif version == 1:
            return awkward1.from_arrow(pyarrow.Table.from_pandas(df))
        else:
            raise RuntimeError(
                "What version of awkward do you want? Specify `version=0` or `1`."
            )


@pd.api.extensions.register_dataframe_accessor("p4")
class LorentzVectorAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self, which):
        components = [f"{which}_{x}" for x in ["pt", "eta", "phi", "mass"]]
        if not isinstance(self._obj, ChunkDataFrame):
            missing_columns = set(components) - set(self._obj.columns)
            if len(missing_columns):
                raise AttributeError("Missing columns: {}".format(missing_columns))
        arrays = (self._obj[c].ak(version=0) for c in components)
        return uproot3_methods.TLorentzVectorArray.from_ptetaphim(*arrays)


@numba.njit()
def max_jit(vals):
    return -np.inf if len(vals) == 0 else np.max(vals)


@numba.njit()
def min_jit(vals):
    return +np.inf if len(vals) == 0 else np.max(vals)


@numba.njit()
def sum_jit(vals):
    return np.sum(vals)


@numba.njit()
def mean_jit(vals):
    return np.nan if len(vals) == 0 else np.mean(vals)


@numba.njit()
def jagged_map(func, content, offsets):
    result = np.zeros(len(offsets) - 1, dtype=content.dtype)
    for cursor_off in range(0, len(offsets) - 1):
        start = offsets[cursor_off]
        stop = offsets[cursor_off + 1]
        accum = func(content[start:stop])
        result[cursor_off] = accum
    return result


@pd.api.extensions.register_series_accessor("nb")
class NumbaListArrayAccessor:
    def __init__(self, obj):
        self._obj = obj
        self._validate()

    def _validate(self):
        good = False
        try:
            good = self._obj.values.dtype.type is list
        except:
            pass
        if not good:
            raise Exception("Only works for ListArray")

    def map(self, operation):
        array = self._obj.values.data
        content = array.values.to_numpy()
        offsets = array.offsets.to_numpy()
        funcs = {
            "min": min_jit,
            "max": max_jit,
            "mean": mean_jit,
            "sum": sum_jit,
        }
        func = funcs.get(operation)
        if func is None:
            raise Exception(
                f"Unknown operation '{operation}'. Must be one of {funcs.keys()}"
            )
        return jagged_map(func, content, offsets)
