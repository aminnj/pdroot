import unittest

import numpy as np
import pandas as pd

from pdroot import tree_draw


def make_df(N):
    np.random.seed(42)
    return pd.DataFrame(
        dict(a=np.random.random(N), b=np.random.random(N), c=np.random.random(N),)
    )


class DrawTest(unittest.TestCase):
    def test_draw_1d(self):
        df = make_df(1000)
        c1 = tree_draw(df, "a+b", "(a<b<c) and (a<0.5)", bins="10,0,2").counts
        sel = df.eval("(a<b<c) and (a<0.5)")
        c2, _ = np.histogram(df.eval("a+b")[sel], bins=np.linspace(0, 2, 11))
        self.assertTrue(np.allclose(c1, c2))

    def test_draw_2d(self):
        df = make_df(1000)
        varexp = "(a+b):b"
        selstr = "(a<0.5)"
        c1 = tree_draw(df, varexp, selstr, bins="10,0,2").counts
        sel = df.eval(selstr).values
        x = df.eval(varexp.split(":")[0]).values[sel]
        y = df.eval(varexp.split(":")[1]).values[sel]
        c2, _, _ = np.histogram2d(x, y, bins=[10, 10], range=[[0, 2], [0, 2]])
        c2 = c2.T
        self.assertTrue(np.allclose(c1, c2))

    def test_pandas_injection(self):
        df = make_df(1000)
        h = df.draw("a")
        self.assertEqual(h.integral, 1000.0)

    def test_jitdraw_1d(self):
        df = make_df(1000)
        bins = np.linspace(0, 2, 11)
        c1 = df.jitdraw("a", "b>0.5", bins=bins).counts
        c2, _ = np.histogram(df["a"][df["b"] > 0.5], bins=bins)
        self.assertTrue(np.allclose(c1, c2))


if __name__ == "__main__":
    unittest.main()
