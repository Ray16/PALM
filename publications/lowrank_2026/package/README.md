# lowrank-split

Low-rank similarity splitting with specified fold sizes. Dense similarity is
approximated by Nyström factors; integer powers and optional capped groups
provide controls over global and local overlap. Lower similarity is not a
guarantee of a realistic deployment test.

```python
from lowrank import split, SplitSpec

result = split("lowrank", feature_data,
               SplitSpec([8, 1, 1], ["train", "val", "test"], seed=0),
               score_exact_leakage=False)
```

Default low-rank fold counts are exact; ratios rounding to an empty fold raise.
Set `allowed_size_imbalance` explicitly to allow a corridor. Group preservation
is conditional: caps, missing graph edges and capacity repair can split related
entities. Diagnostics retain `broken_groups` rather than claim hard exclusion.
Optional exact leakage diagnostics can dominate runtime and are disabled above.

Install with `pip install .`; `.[molecules]` adds RDKit and `.[ann]` adds the
approximate neighbour backend. PyTorch is required; it uses CUDA when visible
and CPU otherwise. The NN metric defaults to CPU and accepts an explicit device.

All support modules are under `lowrank.splitters`, `lowrank.datasets`,
`lowrank.features` and `lowrank.eval`. The wheel installs no top-level package
named `datasets`, `features`, `eval` or `splitters`.

Dataset loaders are optional research helpers. Public raw datasets and the
manuscript's frozen reproduction sources are distributed separately; this
package does not bundle input datasets or author publication identifiers.
