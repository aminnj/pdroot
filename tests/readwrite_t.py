import unittest

import numpy as np
import pandas as pd

from pdroot import to_root, read_root

class ReadWriteTest(unittest.TestCase):

    def test_numerical_columns(self):
        N = 1000
        df1 = pd.DataFrame(dict(
            b1=np.random.normal(3.0, 0.1, N),
            b2=np.random.random(N), 
            b3=np.random.random(N),
            b4=np.random.randint(0,5,N),
            ))
        to_root(df1, "test.root")
        df2 = read_root("test.root")
        self.assertTrue(np.allclose(df1, df2))

    def test_categorical_columns(self):
        N = 1000
        df1 = pd.DataFrame(dict(
            b1=np.random.choice(list("ABCD"), N),
            ))
        df1["b1"] = df1["b1"].astype("category")
        to_root(df1, "test.root")
        df2 = read_root("test.root")
        self.assertTrue((df1["b1"] == df2["b1"]).all())

    def test_string_columns(self):
        N = 1000
        df1 = pd.DataFrame(dict(
            b1=np.random.choice(["my", "string", "list"], N),
            ))
        to_root(df1, "test.root")
        df2 = read_root("test.root")
        self.assertTrue((df1["b1"] == df2["b1"]).all())

    def test_pandas_injection(self):
        df1 = pd.DataFrame(dict(
            b1=np.random.random(100), 
            ))
        df1.to_root("test.root")
        df2 = pd.read_root("test.root")
        self.assertTrue(np.allclose(df1, df2))



if __name__ == "__main__":
    unittest.main()
