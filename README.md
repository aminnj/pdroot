```python
import pdroot
```
...will add some functions to pandas namespace and for dataframes:

Write out dataframes to a ROOT file (including strings, though these are slow and you should `.astype("category")` if possible):
```python
import numpy as np
N = int(1e5)
df = pd.DataFrame(dict(mass=np.random.normal(3.0, 0.1, N), foo=np.random.random(N), bar=np.random.random(N)))
df.to_root("test.root")
```
Read ROOT files and specify certain columns and/or a range of rows. Let's just say that ROOT is the original `parquet` (just with different words: rows -> entries/events, rowgroups -> baskets, columns -> branches).
```python
df = pd.read_root("test*.root", columns=["mass", "foo"])
```
And, for those familiar with `TTree::Draw()`, you can draw directly from a dataframe. This will make a 1D histogram of `mass+0.1` for rows where `0.1<foo<0.2`. All kwargs after first two required args are passed to yahist's Hist1D(). See `pip install yahist` for details on the (nice and simple) histogram object.
```python
df.draw("mass+0.1", "0.1<foo<0.2", bins="200,0,10").plot(histtype="step")
```
or 2D with "x:y"
```python
df.draw("mass:foo+1", "0.1<foo<0.2").plot(logz=True)
```

Lastly, this will read root files and make a histogram in one pass (chunked reading and only branches that are needed)
```python
pdroot.iter_draw("*.root", "mass", "(foo>0.2)", bins="200,0,10")
```
