import pytest

import numpy as np
import pandas as pd

from pdroot import to_root, read_root, iter_chunks, ChunkDataFrame
import fletcher
import awkward0
import awkward1


def test_numerical_columns():
    N = 1000
    df1 = pd.DataFrame(
        dict(
            b1=np.random.normal(3.0, 0.1, N),
            b2=np.random.random(N),
            b3=np.random.random(N),
            b4=np.random.randint(0, 5, N),
        )
    )
    to_root(df1, ".test.root")
    df2 = read_root(".test.root")
    np.testing.assert_allclose(df1, df2)


def test_pandas_injection():
    df1 = pd.DataFrame(dict(b1=np.random.random(100),))
    df1.to_root(".test.root")
    df2 = pd.read_root(".test.root")
    np.testing.assert_allclose(df1, df2)


def test_jagged():
    x_in = fletcher.FletcherContinuousArray([[1.0, 2.0], [], [3.0, 4.0, 5.0]])
    df = pd.DataFrame(dict(x=x_in))
    df.to_root(".test.root", compression_jagged=None)
    x_out = pd.read_root(".test.root")["x"].values
    v_in = list(map(list, x_in.data))
    v_out = list(map(list, x_out.data))
    assert v_in == v_out


def test_awkward_accessor():
    x = fletcher.FletcherContinuousArray([[1.0, 2.0], [], [3.0, 4.0, 5.0]])
    y = np.zeros(len(x), dtype=float)
    df = pd.DataFrame(dict(x=x, y=y))
    df.to_root(".test.root", compression_jagged=None)
    df = pd.read_root(".test.root")
    assert df["x"].ak(0).sum().tolist() == [3.0, 0.0, 12.0]
    assert awkward1.sum(df["x"], axis=-1).tolist() == [3.0, 0.0, 12.0]


def test_p4_accessor():
    N = 10
    df = pd.DataFrame(
        dict(
            Jet_pt=np.zeros(N) + 50.0,
            Jet_eta=np.zeros(N) + 1.1,
            Jet_phi=np.zeros(N) - 1.2,
            Jet_mass=np.zeros(N) + 10.0,
        )
    )
    np.testing.assert_allclose(df.p4("Jet").pt, df["Jet_pt"])


def test_chunkdataframe():
    x = fletcher.FletcherContinuousArray(100 * [[1.0, 2.0], [], [3.0, 4.0, 5.0]])
    y = np.zeros(len(x), dtype=float)
    df = pd.DataFrame(dict(x=x, y=y))
    df.to_root(".test.root", compression_jagged=None)
    df = ChunkDataFrame(
        filename=".test.root", treename="t", entry_start=0, entry_stop=10
    )
    assert "x" not in df.columns
    assert len(df["x"]) == 10
    assert "x" in df.columns

def test_iter_chunks():
    N = 1000
    df1 = pd.DataFrame(
        dict(
            b1=np.random.normal(3.0, 0.1, N),
            b2=np.random.random(N),
            b3=np.random.random(N),
            b4=np.random.randint(0, 5, N),
        )
    )
    to_root(df1, ".test.root")

    columns = ["b1", "b2"]
    chunks = list(iter_chunks(".test.root", columns=columns, progress=False, step_size=N//10))
    assert len(chunks) == 10
    assert sum(map(len, chunks)) == N
    assert len(chunks[0].columns) == len(columns)

if __name__ == "__main__":
    pytest.main(["--capture=no", __file__])
