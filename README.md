## Installation
```
pip install pdroot
```

## Usage

This augments `pandas` making it easier to deal with ROOT files and make histograms:
```python
import pdroot
```

### Reading/writing ROOT files

```python
import pandas as pd
import numpy as np
N = int(1e5)
df = pd.DataFrame(dict(
    mass=np.random.normal(3.0, 0.1, N),
    foo=np.random.random(N), 
    bar=np.random.random(N),
    ))

# write out dataframe to a ROOT file:
df.to_root("test.root")

# read ROOT files and optionally specify certain columns and/or a range of rows.
df = pd.read_root("test.root", columns=["mass", "foo"], entry_stop=1000)
```

### Histogram drawing from DataFrames

For those familiar with ROOT's `TTree::Draw()`, you can compute a histogram directly from a dataframe.
All kwargs after first two required args are passed to a [yahist](https://github.com/aminnj/yahist) Hist1D().
```python
# expression string and a query string
df.draw("mass+0.1", "0.1<foo<0.2", bins="200,0,10")

# 2D with "x:y"
df.draw("mass:foo+1", "0.1<foo<0.2")

# use numba to jit a specialized function (5x faster than using `df.query`/`df.eval`/`np.histogram`).
df.jitdraw("mass+foo+1", "0.1<foo<0.2")
```

Read root files and make a histogram in one pass (chunked reading and only branches that are needed)
```python
pdroot.iter_draw("*.root", "mass", "(foo>0.2)", bins="200,0,10")
```

### Jagged arrays (e.g., in NanoAOD)

* One can read jagged arrays into DataFrames without converting to list of lists by using zero-copy conversions from `awkward1` to `arrow` 
and the [fletcher](https://github.com/xhochy/fletcher) pandas ExtensionArray. In other words, this is fast.

```python
df = pd.read_root("nano.root", columns=["/Electron_(pt|eta|phi|mass)$/", "MET_pt"])
df.head()
```
|    | Electron_eta              | Electron_mass             | Electron_phi          | Electron_pt           |   MET_pt |
|---:|:--------------------------|:--------------------------|:----------------------|:----------------------|---------:|
|  0 | []                        | []                        | []                    | []                    | 208.131  |
|  1 | [2.1259766]               | [0.12030029]              | [0.4970703]           | [121.077896]          |  96.3884 |
|  2 | [-1.1259766]              | [-0.00405121]             | [0.1531372]           | [12.117786]           | 284.988  |
|  3 | [0.17492676]              | [-0.04089355]             | [2.9018555]           | [178.91772]           |  26.7631 |
|  4 | [ 0.12136841 -1.8227539 ] | [-0.00730515 -0.00543594] | [1.4355469 1.3552246] | [19.721205 14.386331] |  48.4577 |

```python
>>> df.dtypes

Electron_eta     fletcher_continuous[list<item: float>]
Electron_mass    fletcher_continuous[list<item: float>]
Electron_phi     fletcher_continuous[list<item: float>]
Electron_pt      fletcher_continuous[list<item: float>]
MET_pt                                          float32
dtype: object
```

It's easy to get the awkward array from the fletcher columns:
```python
>>> df["Electron_pt"].ak() # or .ak(1) to get an `awkward1` array instead of the default `awkward0`

<JaggedArray [[] [121.077896] [12.117786] ... [5.583064] [48.620327 35.415432] []] at 0x0001199ba5f8>
```

And provided the four component branches are in the dataframe, one can do
```python
>>> df.p4("Electron").p

<JaggedArray [[] [514.605] [20.646055] ... [19.576536] [48.658344 35.758152] []] at 0x00012ad87358>
```
