```python
import pdroot
```
...will add some functions to pandas namespace and for dataframes, which make it easier
to deal with ROOT files and making histograms. ROOT I/O uses the excellent [uproot](https://github.com/scikit-hep/uproot3) package.


* Write out dataframes to a ROOT file (including strings, though these are slow and you should `.astype("category")` if possible):
```python
import pandas as pd
import numpy as np
N = int(1e5)
df = pd.DataFrame(dict(
    mass=np.random.normal(3.0, 0.1, N),
    foo=np.random.random(N), 
    bar=np.random.random(N),
    ))
df.to_root("test.root")
```

* Read ROOT files and optionally specify certain columns and/or a range of rows.
```python
df = pd.read_root("test*.root", columns=["mass", "foo"])
```

* For those familiar with ROOT's `TTree::Draw()`, you can draw directly from a dataframe. This will make a 1D histogram of `mass+0.1` for rows where `0.1<foo<0.2`. 
All kwargs after first two required args are passed to a [yahist](https://github.com/aminnj/yahist) Hist1D().
```python
df.draw("mass+0.1", "0.1<foo<0.2", bins="200,0,10")
```
* 2D with "x:y"
```python
df.draw("mass:foo+1", "0.1<foo<0.2")
```

* Use numba to jit everything (about 5x faster than using pandas DataFrame `query`, `eval`, and `np.histogram`).
```python
df.jitdraw("mass:foo+1", "0.1<foo<0.2")
```

* Read root files and make a histogram in one pass (chunked reading and only branches that are needed)
```python
pdroot.iter_draw("*.root", "mass", "(foo>0.2)", bins="200,0,10")
```
