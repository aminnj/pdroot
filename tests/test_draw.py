from pdroot import tree_draw
from pdroot.readwrite import awkward1_arrays_to_dataframe
from pdroot.draw import tree_draw_to_array

import numpy as np
import pandas as pd

import awkward1

import pytest


@pytest.fixture(scope="module")
def df_flat():
    np.random.seed(42)
    N = 1000
    return pd.DataFrame(
        dict(a=np.random.random(N), b=np.random.random(N), c=np.random.random(N))
    )


@pytest.fixture(scope="module")
def df_jagged():
    a = awkward1.Array(
        dict(
            Jet_pt=[[42.0, 15.0, 10.5], [], [11.5], [50.0, 5.0]],
            Jet_eta=[[-2.2, 0.4, 0.5], [], [1.5], [-0.1, -3.0]],
            MET_pt=[46.5, 30.0, 82.0, 8.9],
        )
    )
    # For ease of visualization:
    """
    |    | Jet_pt           | Jet_eta          |   MET_pt |
    |----|------------------|------------------|----------|
    |  0 | [42.  15.  10.5] | [-2.2  0.4  0.5] |     46.5 |
    |  1 | []               | []               |     30   |
    |  2 | [11.5]           | [1.5]            |     82   |
    |  3 | [50.  5.]        | [-0.1 -3. ]      |      8.9 |
    """
    return awkward1_arrays_to_dataframe(a)


def drawclose(df, varexp, sel, y):
    if df is None:
        df = df_jagged()
    x = tree_draw_to_array(df, varexp, sel)
    x = np.array(x)
    y = np.array(y)
    assert x.shape == y.shape
    np.testing.assert_allclose(x, y)


def test_draw_1d(df_flat):
    df = df_flat
    c1 = tree_draw(df, "a+b", "(a<b<c) and (a<0.5)", bins="10,0,2").counts
    sel = df.eval("(a<b<c) and (a<0.5)")
    c2, _ = np.histogram(df.eval("a+b")[sel], bins=np.linspace(0, 2, 11))
    np.testing.assert_allclose(c1, c2)


def test_draw_2d(df_flat):
    df = df_flat
    varexp = "(a+b):b"
    selstr = "(a<0.5)"
    c1 = tree_draw(df, varexp, selstr, bins="10,0,2").counts
    sel = df.eval(selstr).values
    x = df.eval(varexp.split(":")[0]).values[sel]
    y = df.eval(varexp.split(":")[1]).values[sel]
    c2, _, _ = np.histogram2d(x, y, bins=[10, 10], range=[[0, 2], [0, 2]])
    c2 = c2.T
    np.testing.assert_allclose(c1, c2)


def test_pandas_injection(df_flat):
    df = df_flat
    h = df.draw("a")
    assert h.integral == 1000.0


def test_jitdraw_1d(df_flat):
    df = df_flat
    bins = np.linspace(0, 2, 11)

    # one function, has jit cost
    c1 = df.jitdraw("a", "b>0.5", bins=bins).counts
    c2, _ = np.histogram(df["a"][df["b"] > 0.5], bins=bins)
    np.testing.assert_allclose(c1, c2)

    # different function
    c1 = df.jitdraw("b", "c>0.5", bins=bins).counts
    c2, _ = np.histogram(df["b"][df["c"] > 0.5], bins=bins)
    np.testing.assert_allclose(c1, c2)

    # first function, no jit cost
    c1 = df.jitdraw("a", "b>0.5", bins=bins).counts
    c2, _ = np.histogram(df["a"][df["b"] > 0.5], bins=bins)
    np.testing.assert_allclose(c1, c2)


def test_broadcasting(df_jagged):
    df = df_jagged
    drawclose(df, "Jet_pt", "", [42.0, 15.0, 10.5, 11.5, 50.0, 5.0])


def test_draw_to_hist1d(df_jagged):
    df = df_jagged
    assert df.draw("Jet_pt").integral == 6
    assert df.draw("MET_pt").integral == 4


def test_draw_to_hist2d(df_jagged):
    df = df_jagged
    h = df.draw("Jet_pt:Jet_eta", "MET_pt > 40.")
    assert np.ndim(h.counts) == 2
    assert h.integral == 4


def test_broadcasting(df_jagged):
    df = df_jagged
    drawclose(df, "Jet_pt", "", [42.0, 15.0, 10.5, 11.5, 50.0, 5.0])
    drawclose(df, "MET_pt", "", [46.5, 30.0, 82.0, 8.9])
    drawclose(df, "Jet_pt", "Jet_pt > 40", [42.0, 50.0])
    drawclose(df, "MET_pt", "MET_pt > 40", [46.5, 82.0])
    drawclose(df, "Jet_pt", "MET_pt > 40", [42, 15, 10.5, 11.5])
    drawclose(df, "Jet_pt", "abs(Jet_eta) > 1 and MET_pt > 10", [42.0, 11.5])


def test_2d(df_jagged):
    df = df_jagged
    drawclose(
        df,
        "Jet_pt:Jet_eta",
        "MET_pt > 40.",
        [[42, -2.2], [15, 0.4], [10.5, 0.5], [11.5, 1.5]],
    )


def test_correct_value_picked_out(df_jagged):
    df = df_jagged
    drawclose(df, "Jet_eta", "Jet_pt > 40 and MET_pt > 40", [-2.2])
    drawclose(df, "Jet_eta + 1", "Jet_pt > 40 and MET_pt > 40", [-1.2])


def test_reductions(df_jagged):
    df = df_jagged
    drawclose(df, "max(abs(Jet_eta))", "", [2.2, 1.5, 3.0])
    drawclose(df, "max(abs(Jet_eta))", "MET_pt > 80", [1.5])
    drawclose(df, "min(Jet_pt)", "", [10.5, 11.5, 5.0])
    drawclose(
        df, "mean(Jet_pt)", "", [1.0 / 3 * (42 + 15 + 10.5), 11.5, 0.5 * (50 + 5)]
    )
    drawclose(df, "length(Jet_pt)", "", [3, 0, 1, 2])
    drawclose(df, "length(Jet_pt)", "MET_pt < 10", [2])
    drawclose(df, "sum(Jet_pt)", "MET_pt < 10", [50 + 5])


def test_indexing(df_jagged):
    df = df_jagged
    drawclose(df, "Jet_pt[Jet_pt>25]", "", [42, 50])
    drawclose(df, "Jet_pt[2]", "", [10.5])
    drawclose(df, "Jet_pt[0]:Jet_pt[1]", "", [[42, 15], [50, 5]])
    drawclose(df, "Jet_pt[0]:Jet_pt[1]", "MET_pt > 40", [[42, 15]])


def test_indexing_reduction(df_jagged):
    df = df_jagged
    drawclose(df, "sum(Jet_pt[abs(Jet_eta)<2.0])", "", [15 + 10.5, 0.0, 11.5, 50.0])


def test_counting(df_jagged):
    df = df_jagged
    drawclose(df, "sum(Jet_pt>10)", "MET_pt>40", [3, 1])


def test_mathematical_operations(df_jagged):
    df = df_jagged
    drawclose(df, "np.exp(sum(Jet_pt>10))", "MET_pt>40", [np.exp(3), np.exp(1)])


def test_comparisons(df_jagged):
    df = df_jagged
    drawclose(df, "Jet_pt", "(14. < Jet_pt < 16.)", [15])
    drawclose(df, "Jet_pt", "(14. < Jet_pt) and (Jet_pt < 16.)", [15])
    drawclose(df, "Jet_pt", "(14. < Jet_pt) & (Jet_pt < 16.)", [15])


def test_negation(df_jagged):
    df = df_jagged
    drawclose(df, "Jet_pt", "not(14. < Jet_pt < 16.)", [42, 10.5, 11.5, 50, 5])
    drawclose(df, "Jet_pt", "~(14. < Jet_pt < 16.)", [42, 10.5, 11.5, 50, 5])
    drawclose(df, "not MET_pt>40", "", [False, True, False, True])


def test_slicing(df_jagged):
    df = df_jagged
    drawclose(df, "sum(Jet_pt[:2])", "", [42 + 15, 0, 11.5, 50 + 5])
    drawclose(df, "sum(Jet_pt[2:3])", "MET_pt > 40", [10.5, 0.0])
    drawclose(df, "sum(Jet_pt[(Jet_pt>40) and abs(Jet_eta)<2.4])", "", [42, 0, 0, 50])
    drawclose(
        df, "sum(Jet_pt[(Jet_pt>40) and abs(Jet_eta)<2.4])", "MET_pt > 40", [42, 0]
    )


def test_nested(df_jagged):
    df = df_jagged
    drawclose(
        df, "length(Jet_pt) == 2 and sum(Jet_pt) > 40", "", [False, False, False, True]
    )
    drawclose(
        df,
        "(MET_pt>40) and sum((Jet_pt>40) and (abs(Jet_eta)<2.4)) >= 1",
        "",
        [True, False, False, False],
    )


def test_booleans(df_jagged):
    df = df_jagged
    drawclose(df, "(MET_pt>30) or False", "", [True, False, True, False])
    drawclose(df, "(MET_pt>30) and False", "", [False, False, False, False])
    drawclose(df, "(MET_pt>30) or True", "", [True, True, True, True])
    drawclose(df, "(MET_pt>30) and True", "", [True, False, True, False])


if __name__ == "__main__":
    pytest.main(["--capture=no", __file__])
