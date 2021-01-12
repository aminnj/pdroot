import unittest

from pdroot import tree_draw
from pdroot.readwrite import awkward1_arrays_to_dataframe
from pdroot.draw import tree_draw_to_array

import numpy as np
import pandas as pd

import awkward1

class FlatDrawTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        N = 1000
        self.df = pd.DataFrame(
            dict(a=np.random.random(N), b=np.random.random(N), c=np.random.random(N),)
        )

    def test_draw_1d(self):
        df = self.df
        c1 = tree_draw(df, "a+b", "(a<b) and (b<c) and (a<0.5)", bins="10,0,2").counts
        sel = df.eval("(a<b) and (b<c) and (a<0.5)")
        c2, _ = np.histogram(df.eval("a+b")[sel], bins=np.linspace(0, 2, 11))
        self.assertTrue(np.allclose(c1, c2))

    def test_draw_2d(self):
        df = self.df
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
        df = self.df
        h = df.draw("a")
        self.assertEqual(h.integral, 1000.0)

    def test_jitdraw_1d(self):
        np.random.seed(42)
        df = self.df
        bins = np.linspace(0, 2, 11)

        # one function, has jit cost
        c1 = df.jitdraw("a", "b>0.5", bins=bins).counts
        c2, _ = np.histogram(df["a"][df["b"] > 0.5], bins=bins)
        self.assertTrue(np.allclose(c1, c2))

        # different function
        c1 = df.jitdraw("b", "c>0.5", bins=bins).counts
        c2, _ = np.histogram(df["b"][df["c"] > 0.5], bins=bins)
        self.assertTrue(np.allclose(c1, c2))

        # first function, no jit cost
        c1 = df.jitdraw("a", "b>0.5", bins=bins).counts
        c2, _ = np.histogram(df["a"][df["b"] > 0.5], bins=bins)
        self.assertTrue(np.allclose(c1, c2))


class DrawJaggedTest(unittest.TestCase):

    def drawclose(self, varexp, sel, y):
        x = tree_draw_to_array(self.df, varexp, sel)
        x = np.array(x)
        y = np.array(y)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(np.allclose(x, y))

    def setUp(self):
        a = awkward1.Array(
            dict(
                Jet_pt=[[42.0, 15.0, 10.5], [], [11.5], [50.0, 5.0]],
                Jet_eta=[[-2.2, 0.4, 0.5], [], [1.5], [-0.1, -3.0]],
                MET_pt=[46.5, 30.0, 82.0, 8.9],
            )
        )
        self.df = awkward1_arrays_to_dataframe(a)

    def test_draw_to_hist1d(self):
        df = self.df
        self.assertEqual(df.draw("Jet_pt").integral, 6)
        self.assertEqual(df.draw("MET_pt").integral, 4)

    def test_draw_to_hist2d(self):
        df = self.df
        h = df.draw("Jet_pt:Jet_eta", "MET_pt > 40.")
        self.assertEqual(np.ndim(h.counts), 2)
        self.assertEqual(h.integral, 4)

    def test_broadcasting(self):
        self.drawclose("Jet_pt", "", [42.0, 15.0, 10.5, 11.5, 50.0, 5.0])
        self.drawclose("MET_pt", "", [46.5, 30.0, 82.0, 8.9])
        self.drawclose("Jet_pt", "Jet_pt > 40", [42.0, 50.0])
        self.drawclose("MET_pt", "MET_pt > 40", [46.5, 82.0])
        self.drawclose("Jet_pt", "MET_pt > 40", [42, 15, 10.5, 11.5])
        self.drawclose("Jet_pt", "abs(Jet_eta) > 1 and MET_pt > 10", [42.0, 11.5])

    def test_2d(self):
        self.drawclose("Jet_pt:Jet_eta", "MET_pt > 40.", [[42,-2.2],[15,0.4],[10.5,0.5],[11.5,1.5]])

    def test_correct_value_picked_out(self):
        self.drawclose("Jet_eta", "Jet_pt > 40 and MET_pt > 40", [-2.2])
        self.drawclose("Jet_eta + 1", "Jet_pt > 40 and MET_pt > 40", [-1.2])

    def test_reductions(self):
        self.drawclose("max(abs(Jet_eta))", "", [2.2, 1.5, 3.0])
        self.drawclose("max(abs(Jet_eta))", "MET_pt > 80", [1.5])
        self.drawclose("min(Jet_pt)", "", [10.5, 11.5, 5.0])
        self.drawclose("mean(Jet_pt)", "", [1./3*(42+15+10.5), 11.5, 0.5*(50+5)])
        self.drawclose("length(Jet_pt)", "", [3, 0, 1, 2])
        self.drawclose("length(Jet_pt)", "MET_pt < 10", [2])
        self.drawclose("sum(Jet_pt)", "MET_pt < 10", [50+5])

    def test_indexing(self):
        self.drawclose("Jet_pt[Jet_pt>25]", "", [42, 50])
        self.drawclose("Jet_pt[2]", "", [10.5])
        self.drawclose("Jet_pt[0]:Jet_pt[1]", "", [[42,15], [50,5]])
        self.drawclose("Jet_pt[0]:Jet_pt[1]", "MET_pt > 40", [[42,15]])

    def test_indexing_reduction(self):
        self.drawclose("sum(Jet_pt[abs(Jet_eta)<2.0])", "", [15+10.5, 0.0, 11.5, 50.])

    def test_counting(self):
        self.drawclose("sum(Jet_pt>10)", "MET_pt>40", [3, 1])

    def test_mathematical_operations(self):
        self.drawclose("np.exp(sum(Jet_pt>10))", "MET_pt>40", [np.exp(3), np.exp(1)])

    def test_comparisons(self):
        self.drawclose("Jet_pt", "(14. < Jet_pt < 16.)", [15])
        self.drawclose("Jet_pt", "(14. < Jet_pt) and (Jet_pt < 16.)", [15])
        self.drawclose("Jet_pt", "(14. < Jet_pt) & (Jet_pt < 16.)", [15])

    def test_negation(self):
        self.drawclose("Jet_pt", "not(14. < Jet_pt < 16.)", [42, 10.5, 11.5, 50, 5])
        self.drawclose("Jet_pt", "~(14. < Jet_pt < 16.)", [42, 10.5, 11.5, 50, 5])



if __name__ == "__main__":
    unittest.main()
