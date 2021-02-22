## Installation
```
pip install pdroot
```

### Reading/writing ROOT files

```python
import pandas as pd
import numpy as np
import pdroot # powers up pandas

df = pd.DataFrame(dict(foo=np.random.random(100), bar=np.random.random(100)))

# write out dataframe to a ROOT file
df.to_root("test.root")

# read ROOT files and optionally specify certain columns and/or a range of rows
df = pd.read_root("test.root", columns=["foo"], entry_start=0, entry_stop=50)
```

### Histogram drawing from DataFrames

For those familiar with ROOT's `TTree::Draw()`, you can compute a histogram directly from a dataframe.
All kwargs after first two required args are passed to a [yahist](https://github.com/aminnj/yahist) Hist1D().
"Jagged" branches are also supported (see below).
```python
# expression string and a query/selection string
df.draw("mass+0.1", "0.1 < foo < 0.2", bins="200,0,10")

# 2D with "x:y"
df.draw("mass:foo+1", "0.1 < foo < 0.2")
```

### Jagged arrays (e.g., in NanoAOD)

#### Manual reading

One can read jagged arrays into regular DataFrames without converting to (super slow) lists of lists by using zero-copy conversions from `awkward1` to `arrow` 
and the [fletcher](https://github.com/xhochy/fletcher) pandas ExtensionArray.

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

It's easy to get the awkward array from the fletcher columns (also a zero-copy operation):
```python
>>> df["Electron_pt"].ak() 
# or ak(1) to get an `awkward1` array instead of the default `awkward0`

<JaggedArray [[] [121.077896] [12.117786] ... [48.620327 35.415432] []] at 0x0001199ba5f8>
```

Provided four component branches (`*_{pt,eta,phi,mass}`) are in the dataframe, one can do
```python
>>> df.p4("Electron").p

<JaggedArray [[] [514.605] [20.646055] ... [48.658344 35.758152] []] at 0x00012ad87358>
```

#### Drawing/evaluating expressions and queries

Drawing from a DataFrame handles jagged columns via [awkward-array](https://github.com/scikit-hep/awkward-1.0) and AST transformations.
```python
# supports reduction operators:
#    min/max/sum/mean/length/len/argmin/margmax
#    (ROOT's Length$ -> length, etc)
df.draw("length(Jet_pt)")
df.draw("sum(Jet_pt>10)", "MET_pt>40", bins="5,-0.5,4.5")
df.draw("max(abs(Jet_eta))")
df.draw("Jet_eta[argmax(Jet_pt)]")

# combine event-level and object-level selection
df.draw("Jet_pt", "abs(Jet_eta) > 1.0 and MET_pt > 10")

# 2D
df.draw("Jet_pt:Jet_eta", "MET_pt > 40.")

# indexing; imagine you are operating row-by-row, so Jet_pt[0], not Jet_pt[:,0]
df.draw("Jet_pt[0]:Jet_eta[0]", "MET_pt > 10")

# combine reduction operators with fancy indexing
df.draw("sum(Jet_pt[abs(Jet_eta)<2.0])", bins="100,0,100")

# use the underlying array before a histogram is created
df["ht"] = df.draw("sum(Jet_pt[Jet_pt>40])", to_array=True)
```

The expression parsing can be explored via
```python
>>> pdroot.parse.to_ak_expr("sum(Jet_pt[:2])") # sum of first/leading two jet pTs

'ak.sum(ak.pad_none(Jet_pt, 3, clip=True)[:, :2], axis=-1)'
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

