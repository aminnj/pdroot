from pdroot.draw import tree_draw, iter_draw
from pdroot.readwrite import awkward1_arrays_to_dataframe

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
            eventWeight=[-1.0, 0.0, 2.0, 2.0],
        )
    )
    df = awkward1_arrays_to_dataframe(a)
    df["sr"] = ["foo","bar","baz","foo"]
    # For ease of visualization:
    """
    |    | Jet_pt           | Jet_eta          |   MET_pt |   sr | eventWeight |
    |----|------------------|------------------|----------|------|-------------|
    |  0 | [42.  15.  10.5] | [-2.2  0.4  0.5] |     46.5 |  foo |        -1.0 |
    |  1 | []               | []               |     30   |  bar |         0.0 |
    |  2 | [11.5]           | [1.5]            |     82   |  baz |         2.0 |
    |  3 | [50.  5.]        | [-0.1 -3. ]      |      8.9 |  foo |         2.0 |
    """
    return df


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


def test_draw_to_hist1d(df_jagged):
    df = df_jagged
    assert df.draw("Jet_pt").integral == 6
    assert df.draw("MET_pt").integral == 4


def test_draw_to_hist2d(df_jagged):
    df = df_jagged
    h = df.draw("Jet_pt:Jet_eta", "MET_pt > 40.")
    assert np.ndim(h.counts) == 2
    assert h.integral == 4


cases_noweights = [
    ("Jet_pt", "", [42.0, 15.0, 10.5, 11.5, 50.0, 5.0]),
    ("Jet_pt", "abs(Jet_eta) > 1 and MET_pt > 10", [42.0, 11.5]),
    ("Jet_pt", "MET_pt > 40", [42, 15, 10.5, 11.5]),
    ("MET_pt", "MET_pt > 40", [46.5, 82.0]),
    ("Jet_pt", "Jet_pt > 40", [42.0, 50.0]),
    ("MET_pt", "", [46.5, 30.0, 82.0, 8.9]),
    ("Jet_pt", "", [42.0, 15.0, 10.5, 11.5, 50.0, 5.0]),
    ("Jet_eta + 1", "Jet_pt > 40 and MET_pt > 40", [-1.2]),
    ("Jet_eta", "Jet_pt > 40 and MET_pt > 40", [-2.2]),
    ("sum(Jet_pt)", "MET_pt < 10", [50 + 5]),
    ("length(Jet_pt)", "MET_pt < 10", [2]),
    ("len(Jet_pt)", "MET_pt < 10", [2]),
    ("length(Jet_pt)", "", [3, 0, 1, 2]),
    ("mean(Jet_pt)", "", [1.0 / 3 * (42 + 15 + 10.5), 11.5, 0.5 * (50 + 5)]),
    ("min(Jet_pt)", "", [10.5, 11.5, 5.0]),
    ("max(abs(Jet_eta))", "MET_pt > 80", [1.5]),
    ("max(abs(Jet_eta))", "", [2.2, 1.5, 3.0]),
    ("Jet_pt[0]:Jet_pt[1]", "MET_pt > 40", ([42], [15])),
    ("Jet_pt[0]:Jet_pt[1]", "", ([42, 50], [15, 5])),
    ("Jet_pt[2]", "", [10.5]),
    ("Jet_pt[-1]", "", [10.5, 11.5, 5.0]),
    ("Jet_pt[Jet_pt>25]", "", [42, 50]),
    ("sum(Jet_pt[abs(Jet_eta)<2.0])", "", [15 + 10.5, 0.0, 11.5, 50.0]),
    ("sum(Jet_pt>10)", "MET_pt>40", [3, 1]),
    ("np.exp(sum(Jet_pt>10))", "MET_pt>40", [np.exp(3), np.exp(1)]),
    ("Jet_pt", "(14. < Jet_pt) & (Jet_pt < 16.)", [15]),
    ("Jet_pt", "(14. < Jet_pt) and (Jet_pt < 16.)", [15]),
    ("Jet_pt", "(14. < Jet_pt < 16.)", [15]),
    ("not MET_pt>40", "", [False, True, False, True]),
    ("Jet_pt", "~(14. < Jet_pt < 16.)", [42, 10.5, 11.5, 50, 5]),
    ("Jet_pt", "not(14. < Jet_pt < 16.)", [42, 10.5, 11.5, 50, 5]),
    ("(MET_pt>30) and True", "", [True, False, True, False]),
    ("(MET_pt>30) or True", "", [True, True, True, True]),
    ("(MET_pt>30) and False", "", [False, False, False, False]),
    ("(MET_pt>30) or False", "", [True, False, True, False]),
    ("length(Jet_pt) == 2 and sum(Jet_pt) > 40", "", [False, False, False, True]),
    ("len(Jet_pt) == 2 and sum(Jet_pt) > 40", "", [False, False, False, True]),
    ("sum(Jet_pt[(Jet_pt>40) and abs(Jet_eta)<2.4])", "MET_pt > 40", [42, 0]),
    ("sum(Jet_pt[(Jet_pt>40) and abs(Jet_eta)<2.4])", "", [42, 0, 0, 50]),
    ("sum(Jet_pt[2:3])", "MET_pt > 40", [10.5, 0.0]),
    ("sum(Jet_pt[:2])", "", [42 + 15, 0, 11.5, 50 + 5]),
    ("Jet_pt[Jet_pt>40][0] + Jet_eta[2]", "", [42.5]),
    ("min(min(Jet_pt), min(Jet_eta))", "MET_pt > 40", [-2.2, 1.5]),
    ("min(min(Jet_pt), min(Jet_eta))", "", [-2.2, 1.5, -3]),
    ("max(min(Jet_pt), min(Jet_eta))", "", [10.5, 11.5, 5]),
    ("min(MET_pt, 50)", "", [46.5, 30, 50, 8.9]),
    ("min(MET_pt, MET_pt+1)", "", [46.5, 30, 82, 8.9]),
    ("max(MET_pt, MET_pt+1)", "", [47.5, 31, 83, 9.9]),
    ("max(min(Jet_pt), MET_pt*2)", "", [93, 164, 17.8]),
    ("10*(1>0) + MET_pt", "MET_pt>40", [56.5, 92]),
    ("sum(-0.8<Jet_eta<0.8 and Jet_pt>25)", "", [0, 0, 0, 1]),
    ("1", "MET_pt>40", [1, 1]),
    ("10", "", [10, 10, 10, 10]),
    ("Jet_eta[argmin(Jet_pt)]", "", [0.5, 1.5, -3.0]),
    ("Jet_eta[argmax(2*Jet_pt)]", "", [-2.2, 1.5, -0.1]),
    ("Jet_eta[argmax(2*Jet_pt)]", "", [-2.2, 1.5, -0.1]),
    ("0.5*(Jet_pt[0] + Jet_pt[-1])", "length(Jet_pt)>=2", [26.25, 27.5]),
    ("MET_pt", "Jet_pt>12", [46.5, 46.5, 8.9]),
    ("Jet_pt:Jet_eta", "MET_pt > 40.", ([42, 15, 10.5, 11.5], [-2.2, 0.4, 0.5, 1.5]),),
    (
        "(MET_pt>40) and sum((Jet_pt>40) and (abs(Jet_eta)<2.4)) >= 1",
        "",
        [True, False, False, False],
    ),
    ("MET_pt", "(sr in ['foo','bar']) and MET_pt>10", [46.5, 30],),
]


@pytest.mark.parametrize("varexp,sel,expected", cases_noweights)
def test_draw(df_jagged, varexp, sel, expected):
    x = tree_draw(df_jagged, varexp, sel, to_array=True)
    x = np.array(x)
    y = np.array(expected)
    np.testing.assert_allclose(x, y)


cases_weights = [
    (
        "Jet_pt",
        "",
        "Jet_pt",
        [42.0, 15.0, 10.5, 11.5, 50.0, 5.0],
        [42.0, 15.0, 10.5, 11.5, 50.0, 5.0],
    ),
    (
        "Jet_pt",
        "abs(Jet_eta) > 1 and MET_pt > 10",
        "Jet_eta*2",
        [42.0, 11.5],
        [-4.4, 3.0],
    ),
    ("MET_pt", "MET_pt > 40", "eventWeight", [46.5, 82], [-1, 2]),
    ("length(Jet_pt)", "", "eventWeight", [3, 0, 1, 2], [-1, 0, 2, 2]),
    ("length(Jet_pt)", "MET_pt < 10", "eventWeight", [2], [2]),
    ("Jet_pt[0]:Jet_pt[1]", "", "eventWeight", ([42, 50], [15, 5]), [-1, 2]),
    ("Jet_pt[0]", "MET_pt>40", "length(Jet_pt)", [42.0, 11.5], [3, 1]),
    ("Jet_pt[0]", "", "length(Jet_pt)", [42.0, 11.5, 50], [3, 1, 2]),
]


@pytest.mark.parametrize("varexp,sel,weights,expected,expectedweights", cases_weights)
def test_draw_weights(df_jagged, varexp, sel, weights, expected, expectedweights):
    x, vweights = tree_draw(df_jagged, varexp, sel, weights, to_array=True)
    x = np.array(x)
    x_exp = np.array(expected)
    vweights_exp = np.array(expectedweights)
    np.testing.assert_allclose(x, x_exp)
    np.testing.assert_allclose(vweights, vweights_exp)


def test_draw_custom_func(df_jagged):
    df = df_jagged

    def myfunc(x, y):
        return x + y

    x = df.draw(
        "myfunc(Jet_pt,Jet_eta)", "MET_pt > 40.", to_array=True, env=dict(myfunc=myfunc)
    )
    x_exp = np.array([39.8, 15.4, 11, 13])
    np.testing.assert_allclose(x, x_exp)


# def test_aliases(df_jagged):
#     df = df_jagged
#     x = df.draw("sum((Jet_pt>40) and abs(Jet_eta)<2.4)", "MET_pt>40", to_array=True)
#     aliases = {
#             "njets": "sum((Jet_pt>40) and abs(Jet_eta)<2.4)",
#             "highmet": "MET_pt>40",
#             }
#     y = df.draw("njets", "highmet", aliases=aliases, to_array=True)
#     np.testing.assert_allclose(x, y)


def test_iterdraw():
    treename = "tree"
    filename = ".test.root"
    varexp = "a"
    sel = "b>c"
    df = pd.DataFrame(np.random.normal(0, 1, (1000, 4)), columns=list("abcd"))
    df.to_root(filename, treename=treename)
    h = iter_draw(
        filename, varexp, sel=sel, treename=treename, step_size=500, progress=False
    )
    assert h.integral == df.eval(sel).sum()


if __name__ == "__main__":
    pytest.main(["--capture=no", __file__])
