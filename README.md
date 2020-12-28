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

# use numba to jit a specialized function 
# (can be an order of magnitude faster than df.query/df.eval/np.histogram).
df.jitdraw("mass+foo+1", "0.1<foo<0.2")
```

Read root files and make a histogram in one pass (chunked reading and only branches that are needed)
```python
pdroot.iter_draw("*.root", "mass", "(foo>0.2)", bins="200,0,10")
```

### Jagged arrays (e.g., in NanoAOD)

#### Manual reading

One can read jagged arrays into DataFrames without converting to (super slow) lists of lists by using zero-copy conversions from `awkward1` to `arrow` 
and the [fletcher](https://github.com/xhochy/fletcher) pandas ExtensionArray. In other words, after the arrays are read from the ROOT file,
making the dataframe is instantaneous.

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

It's easy to get the awkward array from the fletcher columns (also a zero-copy operation):
```python
>>> df["Electron_pt"].ak() # or .ak(1) to get an `awkward1` array instead of the default `awkward0`

<JaggedArray [[] [121.077896] [12.117786] ... [48.620327 35.415432] []] at 0x0001199ba5f8>
```

And provided the four component branches are in the dataframe, one can do
```python
>>> df.p4("Electron").p

<JaggedArray [[] [514.605] [20.646055] ... [48.658344 35.758152] []] at 0x00012ad87358>
```

#### Lazy chunked reading

`ChunkDataFrame` subclasses `pd.DataFrame` and lazily reads from a chunk of a file (or a whole one).
```python
df = pdroot.ChunkDataFrame(filename="nano.root", entry_start=0, entry_stop=100e3)

pt = df["Jet_pt"].ak()
df["ht"] = pt[pt > 40].sum()

df.head()
```

|    | Jet_pt                                                                 |      ht |
|---:|:-----------------------------------------------------------------------|--------:|
|  0 | [270.75       85.5        39.90625    29.71875    27.453125   20.53125 18.234375   15.4140625] | 356.25  |
|  1 | [145.75     144.375     64.6875    59.1875    25.875     17.546875]    | 414     |
|  2 | [343.5       91.5       63.5625    57.15625   29.984375]               | 555.719 |
|  3 | [192.625    108.125     56.40625   55.75      33.40625   24.140625  21.625     21.3125    17.25      16.75      16.125   ]   | 412.906 |
|  4 | [105.4375    85.6875    73.        54.5       53.875     40.78125    29.328125  24.484375  23.34375 ]  | 413.281 |
