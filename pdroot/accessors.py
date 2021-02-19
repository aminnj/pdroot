import awkward0
import awkward1
import pyarrow
import pandas as pd
import uproot3_methods
import numpy as np

from .readwrite import ChunkDataFrame


def pandas_series_to_awkward(series, version=0):
    values = series.values
    if "fletcher" not in str(values.dtype).lower():
        if version == 1:
            return awkward1.from_numpy(values)
        else:
            return np.array(values, copy=False)

    array_arrow = values.data

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
