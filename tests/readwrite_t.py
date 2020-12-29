import unittest

import numpy as np
import pandas as pd

from pdroot import to_root, read_root, ChunkDataFrame
import fletcher
import awkward0
import awkward1


class ReadWriteTest(unittest.TestCase):
    def test_numerical_columns(self):
        N = 1000
        df1 = pd.DataFrame(
            dict(
                b1=np.random.normal(3.0, 0.1, N),
                b2=np.random.random(N),
                b3=np.random.random(N),
                b4=np.random.randint(0, 5, N),
            )
        )
        to_root(df1, "test.root")
        df2 = read_root("test.root")
        self.assertTrue(np.allclose(df1, df2))

    def test_pandas_injection(self):
        df1 = pd.DataFrame(dict(b1=np.random.random(100),))
        df1.to_root("test.root")
        df2 = pd.read_root("test.root")
        self.assertTrue(np.allclose(df1, df2))

    def test_jagged(self):
        x_in = fletcher.FletcherContinuousArray([[1.0, 2.0], [], [3.0, 4.0, 5.0]])
        df = pd.DataFrame(dict(x=x_in))
        df.to_root("test.root")
        x_out = pd.read_root("test.root")["x"].values
        self.assertEqual(x_in.data, x_out.data)

    def test_awkward_accessor(self):
        x = fletcher.FletcherContinuousArray([[1.0, 2.0], [], [3.0, 4.0, 5.0]])
        y = np.zeros(len(x), dtype=float)
        df = pd.DataFrame(dict(x=x, y=y))
        df.to_root("test.root")
        df = pd.read_root("test.root")
        self.assertEqual(df["x"].ak(0).sum().tolist(), [3.0, 0.0, 12.0])
        self.assertEqual(awkward1.sum(df["x"], axis=-1).tolist(), [3.0, 0.0, 12.0])

    def test_p4_accessor(self):
        N = 10
        df = pd.DataFrame(
            dict(
                Jet_pt=np.zeros(N) + 50.0,
                Jet_eta=np.zeros(N) + 1.1,
                Jet_phi=np.zeros(N) - 1.2,
                Jet_mass=np.zeros(N) + 10.0,
            )
        )
        self.assertTrue((df.p4("Jet").pt == df["Jet_pt"]).all())

    def test_chunkdataframe(self):
        x = fletcher.FletcherContinuousArray(100 * [[1.0, 2.0], [], [3.0, 4.0, 5.0]])
        y = np.zeros(len(x), dtype=float)
        df = pd.DataFrame(dict(x=x, y=y))
        df.to_root("test.root")

        df = ChunkDataFrame(
            filename="test.root", treename="t", entry_start=0, entry_stop=10
        )
        self.assertTrue("x" not in df.columns)
        self.assertEqual(len(df["x"]), 10)
        self.assertTrue("x" in df.columns)


if __name__ == "__main__":
    unittest.main()
